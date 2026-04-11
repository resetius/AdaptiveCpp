/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

#include "hipSYCL/runtime/metal/metal_mmap_region.hpp"
#include "hipSYCL/common/debug.hpp"

#include <sys/mman.h>   // mmap, munmap
#include <sys/sysctl.h> // sysctlbyname (hw.memsize)
#include <cassert>
#include <stdexcept>

namespace hipsycl {
namespace rt {

// ---------------------------------------------------------------------------
// Helper: round `v` up to the nearest multiple of `align` (power-of-two).
// ---------------------------------------------------------------------------
static uintptr_t align_up(uintptr_t v, std::size_t align) {
  return (v + align - 1) & ~(uintptr_t)(align - 1);
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

metal_mmap_region::metal_mmap_region(std::size_t capacity)
    : _capacity{capacity} {
  // Reserve a contiguous virtual-address range.  MAP_ANONYMOUS means no
  // backing file; pages are zero-filled on first access (demand paging).
  void *ptr = ::mmap(nullptr, capacity,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     /*fd=*/-1, /*offset=*/0);
  if (ptr == MAP_FAILED)
    throw std::runtime_error("metal_mmap_region: mmap failed");

  _base = ptr;
  // Start the bump cursor from the MIDDLE of the reservation (Turner approach
  // 2: https://tallendev.github.io/assets/papers/sc21.pdf).
  // This leaves VA room for corrected_cpu addresses in both directions.
  _cursor = reinterpret_cast<uintptr_t>(ptr) + capacity / 2;
  HIPSYCL_DEBUG_INFO << "metal_mmap_region: reserved " << (capacity >> 20)
                     << " MiB at " << ptr << " (cursor starts at middle)\n";
}

metal_mmap_region::~metal_mmap_region() {
  if (_base)
    ::munmap(_base, _capacity);
}

// ---------------------------------------------------------------------------
// alloc
// ---------------------------------------------------------------------------

void *metal_mmap_region::alloc(std::size_t size, std::size_t align) {
  if (size == 0) return nullptr;
  if (align == 0) align = 1;

  std::lock_guard<std::mutex> lock{_mutex};

  // ------------------------------------------------------------------
  // Phase 1: search the free-list (first-fit with alignment).
  //
  // For each free block [block_start, block_start+block_size):
  //   aligned_start = align_up(block_start, align)
  //   If aligned_start + size <= block_start + block_size the block fits.
  //
  //   We carve out [aligned_start, aligned_start+size).
  //   Left remnant  = [block_start,        aligned_start)   if non-empty
  //   Right remnant = [aligned_start+size, block_start+block_size) if non-empty
  //   Both remnants go back into _free_blocks.
  // ------------------------------------------------------------------
  for (auto it = _free_blocks.begin(); it != _free_blocks.end(); ++it) {
    uintptr_t   block_start = it->first;
    std::size_t block_size  = it->second;
    uintptr_t   block_end   = block_start + block_size;

    uintptr_t aligned_start = align_up(block_start, align);
    if (aligned_start + size > block_end)
      continue; // does not fit

    // Remove this free block; we will reinsert remnants below.
    _free_blocks.erase(it);

    // Left remnant: padding bytes before the aligned start.
    if (aligned_start > block_start)
      _free_blocks[block_start] = aligned_start - block_start;

    // Right remnant: tail bytes after the allocation.
    uintptr_t alloc_end = aligned_start + size;
    if (alloc_end < block_end)
      _free_blocks[alloc_end] = block_end - alloc_end;

    return reinterpret_cast<void *>(aligned_start);
  }

  // ------------------------------------------------------------------
  // Phase 2: no suitable free block — advance the bump cursor.
  //
  //   aligned_start = align_up(cursor, align)
  //   new_cursor    = aligned_start + size
  //
  //   If there is a gap between the old cursor and aligned_start (because
  //   of alignment padding), that gap is a free block and is added to
  //   _free_blocks so it can be reclaimed by a future smaller allocation.
  // ------------------------------------------------------------------
  uintptr_t   aligned_start = align_up(_cursor, align);
  uintptr_t   new_cursor    = aligned_start + size;
  uintptr_t   region_end    = reinterpret_cast<uintptr_t>(_base) + _capacity;

  if (new_cursor > region_end) {
    HIPSYCL_DEBUG_WARNING << "metal_mmap_region: region exhausted\n";
    return nullptr;
  }

  // Any alignment gap between old cursor and aligned_start becomes free.
  if (aligned_start > _cursor)
    _free_blocks[_cursor] = aligned_start - _cursor;

  _cursor = new_cursor;
  return reinterpret_cast<void *>(aligned_start);
}

// ---------------------------------------------------------------------------
// alloc_at — Turner approach 2 retry helper
// ---------------------------------------------------------------------------

void *metal_mmap_region::alloc_at(void *addr, std::size_t size) {
  if (!addr || size == 0) return nullptr;

  uintptr_t start      = reinterpret_cast<uintptr_t>(addr);
  uintptr_t end        = start + size;
  uintptr_t region_beg = reinterpret_cast<uintptr_t>(_base);
  uintptr_t region_end = region_beg + _capacity;

  if (start < region_beg || end > region_end) {
    HIPSYCL_DEBUG_WARNING << "metal_mmap_region::alloc_at: "
                             "address out of region bounds\n";
    return nullptr;
  }

  std::lock_guard<std::mutex> lock{_mutex};

  // If addr+size extends past the bump cursor, advance the cursor so that
  // subsequent alloc() calls never overlap these pages.
  if (end > _cursor)
    _cursor = end;

  // Remove any free-list entries that overlap [start, end).
  // Typically corrected_cpu is a fresh address, but the retry may
  // happen to land on a recently-freed block.
  auto it = _free_blocks.lower_bound(start);
  if (it != _free_blocks.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second > start)
      it = prev;
  }
  while (it != _free_blocks.end() && it->first < end) {
    uintptr_t   bs = it->first;
    uintptr_t   be = bs + it->second;
    it = _free_blocks.erase(it);
    if (bs < start) _free_blocks[bs]  = start - bs;
    if (be > end)   _free_blocks[end] = be - end;
  }

  return addr;
}

// ---------------------------------------------------------------------------
// free
// ---------------------------------------------------------------------------

void metal_mmap_region::free(void *ptr, std::size_t size) {
  if (!ptr || size == 0) return;

  uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t end   = start + size;

  std::lock_guard<std::mutex> lock{_mutex};

  // ------------------------------------------------------------------
  // Insert the block, then coalesce with neighbours.
  // ------------------------------------------------------------------
  auto [inserted_it, _] = _free_blocks.emplace(start, size);

  // Try to merge with the next free block.
  auto next_it = std::next(inserted_it);
  if (next_it != _free_blocks.end() && next_it->first == end) {
    inserted_it->second += next_it->second;
    _free_blocks.erase(next_it);
    end = inserted_it->first + inserted_it->second;
  }

  // Try to merge with the previous free block.
  if (inserted_it != _free_blocks.begin()) {
    auto prev_it = std::prev(inserted_it);
    if (prev_it->first + prev_it->second == start) {
      prev_it->second += inserted_it->second;
      _free_blocks.erase(inserted_it);
    }
  }
}

// ---------------------------------------------------------------------------
// used_extent
// ---------------------------------------------------------------------------

std::size_t metal_mmap_region::used_extent() const {
  std::lock_guard<std::mutex> lock{_mutex};
  return _cursor - reinterpret_cast<uintptr_t>(_base);
}

} // namespace rt
} // namespace hipsycl
