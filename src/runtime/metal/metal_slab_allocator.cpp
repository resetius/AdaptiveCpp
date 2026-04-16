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

#include "hipSYCL/runtime/metal/metal_slab_allocator.hpp"
#include "hipSYCL/common/debug.hpp"

#include <cassert>
#include <cstring>

namespace hipsycl {
namespace rt {

// ---------------------------------------------------------------------------
// size_class / is_small
// ---------------------------------------------------------------------------

std::size_t metal_slab_allocator::size_class(std::size_t size) const noexcept {
  std::size_t sc = kMinSlotBytes;
  while (sc < size)
    sc *= 2;
  return sc;
}

bool metal_slab_allocator::is_small(std::size_t size,
                                    std::size_t page_size) const noexcept {
  return size <= kSmallAllocThresholdPages * page_size;
}

// ---------------------------------------------------------------------------
// Bitmap helpers
// ---------------------------------------------------------------------------

std::size_t metal_slab_allocator::find_free_slot(const SlabMeta &s) noexcept {
  for (std::size_t w = 0; w < s.bitmap.size(); ++w) {
    if (s.bitmap[w] != 0) {
      int bit = __builtin_ctzll(s.bitmap[w]);
      std::size_t idx = w * 64 + static_cast<std::size_t>(bit);
      if (idx < s.num_slots)
        return idx;
    }
  }
  return SIZE_MAX;
}

void metal_slab_allocator::mark_used(SlabMeta &s, std::size_t idx) noexcept {
  s.bitmap[idx / 64] &= ~(uint64_t(1) << (idx % 64));
}

void metal_slab_allocator::mark_free(SlabMeta &s, std::size_t idx) noexcept {
  s.bitmap[idx / 64] |= uint64_t(1) << (idx % 64);
}

// ---------------------------------------------------------------------------
// register_slab
// ---------------------------------------------------------------------------

void metal_slab_allocator::register_slab(void       *base,
                                         std::size_t slot_size,
                                         std::size_t num_slots,
                                         metal_usm_alloc_type type) {
  const std::size_t bitmap_words = (num_slots + 63) / 64;
  SlabMeta meta;
  meta.slot_size  = slot_size;
  meta.num_slots  = num_slots;
  meta.free_count = num_slots;
  meta.alloc_type = type;
  meta.bitmap.assign(bitmap_words, ~uint64_t(0));

  // Clear stray bits in the last word beyond num_slots.
  if (num_slots % 64 != 0) {
    meta.bitmap.back() = ~uint64_t(0) >> (64 - (num_slots % 64));
  }

  _slabs[reinterpret_cast<uintptr_t>(base)] = std::move(meta);

  HIPSYCL_DEBUG_INFO << "metal_slab_allocator: registered slab slot_size="
                     << slot_size << " num_slots=" << num_slots
                     << " cpu=" << base << "\n";
}

// ---------------------------------------------------------------------------
// try_alloc_slot
// ---------------------------------------------------------------------------

void *metal_slab_allocator::try_alloc_slot(std::size_t slot_size,
                                            metal_usm_alloc_type type) {
  for (auto &[base, meta] : _slabs) {
    if (meta.slot_size != slot_size || meta.alloc_type != type ||
        meta.free_count == 0)
      continue;
    std::size_t idx = find_free_slot(meta);
    if (idx == SIZE_MAX)
      continue;
    mark_used(meta, idx);
    meta.free_count--;
    return reinterpret_cast<void *>(base + idx * slot_size);
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// free_slot
// ---------------------------------------------------------------------------

bool metal_slab_allocator::free_slot(void *slot_ptr, void *slab_base) {
  auto it = _slabs.find(reinterpret_cast<uintptr_t>(slab_base));
  if (it == _slabs.end())
    return false;

  SlabMeta &meta      = it->second;
  uintptr_t base_addr = reinterpret_cast<uintptr_t>(slab_base);
  uintptr_t slot_addr = reinterpret_cast<uintptr_t>(slot_ptr);

  std::size_t slot_idx = (slot_addr - base_addr) / meta.slot_size;
  mark_free(meta, slot_idx);
  meta.free_count++;

  return meta.free_count == meta.num_slots;
}

// ---------------------------------------------------------------------------
// remove_slab
// ---------------------------------------------------------------------------

void metal_slab_allocator::remove_slab(void *base) {
  _slabs.erase(reinterpret_cast<uintptr_t>(base));
}

} // namespace rt
} // namespace hipsycl
