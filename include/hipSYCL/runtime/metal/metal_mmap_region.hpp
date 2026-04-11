/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_METAL_MMAP_REGION_HPP
#define HIPSYCL_METAL_MMAP_REGION_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

namespace hipsycl {
namespace rt {

// ---------------------------------------------------------------------------
// metal_mmap_region
//
// Manages a single large anonymous mmap region that backs all USM allocations
// on a Metal device.  Because every sub-allocation comes from the same
// contiguous virtual-address range, Apple Silicon's IOMMU maps the whole
// region into the GPU address space with a constant offset:
//
//   delta = gpu_virtual_addr - cpu_virtual_addr  (same for every byte in region)
//
// This lets the pointer-translation kernel pass use one global delta for all
// USM pointers, regardless of how many independent malloc_shared calls the
// application makes.
//
// Sub-allocation strategy: sorted free-list + bump-pointer fallback.
//
//   Invariants:
//     - [base, cursor) has been handed out at least once (may contain free blocks)
//     - [cursor, base+capacity) has never been touched
//     - free_blocks holds non-overlapping, non-adjacent ranges within [base, cursor)
//       sorted by start address; adjacent free blocks are always coalesced
//
//   alloc(size, align):
//     1. Walk free_blocks (first-fit) looking for a block that can satisfy
//        the aligned request.  Alignment padding may leave a small "left
//        remnant" at the start of the free block; the remainder after the
//        allocation is a "right remnant".  Both remnants, if non-zero, are
//        reinserted into free_blocks.
//     2. If no free block is large enough, advance the cursor by
//        (align - 1 + size) bytes, rounding the allocation start up to the
//        requested alignment.  Any padding gap before the aligned start is
//        added to free_blocks so it can be reclaimed later.
//
//   free(ptr, size):
//     1. Insert {ptr, size} into free_blocks.
//     2. Coalesce with the immediately preceding block if it ends exactly at
//        ptr, and with the immediately following block if it starts exactly at
//        ptr+size.  This keeps free_blocks compact and prevents fragmentation.
// ---------------------------------------------------------------------------

class metal_mmap_region {
public:
  // Allocates a contiguous anonymous virtual-address range of `capacity` bytes.
  // The region is NOT pre-faulted; physical pages are committed on first touch.
  explicit metal_mmap_region(std::size_t capacity);

  // Releases the entire mmap region.  All outstanding MTL::Buffer objects that
  // were created over sub-ranges must have been released before this point.
  ~metal_mmap_region();

  // Non-copyable, non-movable (the mmap address is baked into Metal buffers).
  metal_mmap_region(const metal_mmap_region &) = delete;
  metal_mmap_region &operator=(const metal_mmap_region &) = delete;

  // Returns a pointer to an unused sub-range of at least `size` bytes, aligned
  // to `align` bytes (must be a power of two, >= 1).
  // Returns nullptr if the region is exhausted.
  void *alloc(std::size_t size, std::size_t align);

  // Reserve a specific address within the region for use as a USM allocation.
  // Used by Turner approach 2 retry: after a delta mismatch the caller
  // computes corrected_cpu = observed_gpu - _delta and calls this to place
  // the allocation exactly there.
  //   - Removes any free-list block that overlaps [addr, addr+size).
  //   - If addr+size is past the bump cursor, advances cursor to addr+size.
  // Returns addr on success, nullptr if addr is outside the region bounds.
  void *alloc_at(void *addr, std::size_t size);

  // Returns the sub-range [ptr, ptr+size) to the free pool.
  // The caller must ensure the corresponding MTL::Buffer has already been
  // released before calling free().
  void free(void *ptr, std::size_t size);

  // Base address of the mmap region (constant after construction).
  void *base() const { return _base; }

  // Total capacity of the region in bytes.
  std::size_t capacity() const { return _capacity; }

  // Number of bytes between base() and the bump cursor (high-water mark).
  std::size_t used_extent() const;

private:
  void       *_base     = nullptr;
  std::size_t _capacity = 0;
  // Bump cursor: next untouched byte beyond the high-water mark.
  uintptr_t   _cursor   = 0;

  // Free blocks within [base, cursor), sorted by start address.
  // Invariant: no two adjacent entries — they are always merged on free().
  std::map<uintptr_t, std::size_t> _free_blocks; // start → size

  mutable std::mutex _mutex;
};

} // namespace rt
} // namespace hipsycl

#endif // HIPSYCL_METAL_MMAP_REGION_HPP
