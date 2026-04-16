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
#ifndef HIPSYCL_METAL_SLAB_ALLOCATOR_HPP
#define HIPSYCL_METAL_SLAB_ALLOCATOR_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace hipsycl {
namespace rt {

// ---------------------------------------------------------------------------
// metal_usm_alloc_type
//
// Defined at namespace scope so it can be shared between metal_allocator and
// metal_slab_allocator without circular header dependencies.
// ---------------------------------------------------------------------------
enum class metal_usm_alloc_type {
  shared    = 0,
  device    = 1,
  host      = 2,
  undefined = 3,
};

// ---------------------------------------------------------------------------
// metal_slab_allocator
//
// Pure bitmap manager for "small" USM allocations.  It knows nothing about
// Metal buffers, mmap regions, or USM deltas — those are all managed by
// metal_allocator.  The slab allocator ONLY:
//
//   • Tracks which slots are free / used within a registered slab.
//   • Allocates the next free slot from any suitable registered slab.
//   • Marks a slot free on deallocation; reports when the slab is empty.
//
// Slab Metal buffers are owned by metal_allocator and registered in its
// _ptr_to_block map.  Because they appear in _ptr_to_block with their full
// stride as the block size, metal_allocator::get_usm_block() finds any slot
// within the slab automatically via its existing upper_bound range-lookup —
// no separate get_block/owns methods are needed here.
//
// Threading
// ---------
// All methods are called with the parent metal_allocator::_mutex already
// held.  This class has NO internal mutex.
// ---------------------------------------------------------------------------
class metal_slab_allocator {
public:
  // -----------------------------------------------------------------------
  // Tuning knobs
  // -----------------------------------------------------------------------

  // Allocations <= kSmallAllocThresholdPages * page_size go through the slab.
  static constexpr std::size_t kSmallAllocThresholdPages = 4;

  // Size of each slab buffer in bytes.
  // Must stay within the range validated by the GPU-VA stride formula
  // (alloc_demo.cpp tests sizes up to 1 MiB → use 1 MiB).
  static constexpr std::size_t kSlabBytes = 1ULL << 20; // 1 MiB

  // Smallest slot size class.
  static constexpr std::size_t kMinSlotBytes = 64;

  // -----------------------------------------------------------------------
  // Public API (all called with parent mutex held)
  // -----------------------------------------------------------------------

  // Round `size` up to the nearest power-of-two size class >= kMinSlotBytes.
  std::size_t size_class(std::size_t size) const noexcept;

  // True iff size <= kSmallAllocThresholdPages * page_size.
  bool is_small(std::size_t size, std::size_t page_size) const noexcept;

  // Register a newly created slab.
  // base:      CPU base address of the slab buffer.
  // slot_size: byte size of each slot (a size-class value).
  // num_slots: number of slots in the slab.
  // type:      USM allocation type for all slots in this slab.
  void register_slab(void *base, std::size_t slot_size,
                     std::size_t num_slots, metal_usm_alloc_type type);

  // Allocate the first free slot of the given slot_size and alloc type.
  // Returns the CPU address of the slot, or nullptr if no slab has a free slot
  // of that size and type.
  void *try_alloc_slot(std::size_t slot_size, metal_usm_alloc_type type);

  // Mark a slot free.
  // base:      CPU base of the owning slab (obtained from the _ptr_to_block
  //            range-lookup done by raw_free before calling here).
  // Returns true if the slab is now completely free (caller should destroy it).
  bool free_slot(void *slot_ptr, void *slab_base);

  // Remove slab metadata (call after destroying the Metal buffer and freeing
  // the backing mmap pages).
  void remove_slab(void *base);

private:
  struct SlabMeta {
    std::size_t           slot_size  = 0;
    std::size_t           num_slots  = 0;
    std::size_t           free_count = 0;
    metal_usm_alloc_type  alloc_type = metal_usm_alloc_type::undefined;
    // Bitmap: bit=1 means free.  Indexed as word[idx/64] bit[idx%64].
    std::vector<uint64_t> bitmap;
  };

  std::map<uintptr_t, SlabMeta> _slabs; // cpu_base → metadata

  static std::size_t find_free_slot(const SlabMeta &s) noexcept;
  static void mark_used(SlabMeta &s, std::size_t idx) noexcept;
  static void mark_free(SlabMeta &s, std::size_t idx) noexcept;
};

} // namespace rt
} // namespace hipsycl

#endif // HIPSYCL_METAL_SLAB_ALLOCATOR_HPP
