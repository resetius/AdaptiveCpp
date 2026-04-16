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
#ifndef HIPSYCL_METAL_ALLOCATOR_HPP
#define HIPSYCL_METAL_ALLOCATOR_HPP

#include "../allocator.hpp"
#include "../hints.hpp"
#include "metal_mmap_region.hpp"
#include "metal_slab_allocator.hpp"

#include <map>
#include <memory>

namespace MTL {

class Device;
class Buffer;

} // namespace MTL

namespace hipsycl {
namespace rt {

class metal_allocator : public backend_allocator
{
public:
  // usm_alloc_type is defined at namespace scope in metal_slab_allocator.hpp
  // and aliased here for backward compatibility.
  using usm_alloc_type = metal_usm_alloc_type;

  metal_allocator(MTL::Device* device, const device_id &id);
  ~metal_allocator();

  virtual void* raw_allocate(size_t min_alignment, size_t size_bytes,
                             const allocation_hints &hints = {}) override;

  virtual void *
  raw_allocate_optimized_host(size_t min_alignment, size_t bytes,
                              const allocation_hints &hints = {}) override;

  virtual void raw_free(void *mem) override;

  virtual void *raw_allocate_usm(size_t bytes,
                                 const allocation_hints &hints = {}) override;
  virtual bool is_usm_accessible_from(backend_descriptor b) const override;

  virtual result query_pointer(const void *ptr,
                               pointer_info &out) const override;

  virtual result mem_advise(const void *addr, std::size_t num_bytes,
                            int advise) const override;

  virtual device_id get_device() const override;

  // Returns the Metal buffer and offset for a given USM pointer.
  // Works for both regular allocations and slab slots: slab buffers are
  // registered in _ptr_to_block and found via range-lookup automatically.
  std::tuple<MTL::Buffer*, size_t, usm_alloc_type> get_usm_block(const void* ptr) const;

  // GPU-to-host address delta for all shared/host USM allocations.
  // Valid after the first shared allocation; returns 0 before that.
  int64_t usm_delta() const;

private:
  MTL::Device* _device = nullptr;
  device_id _device_id;

  // Backing mmap region shared by all shared/host USM allocations.
  // Device-only (StorageModePrivate) allocations do NOT use this region.
  std::unique_ptr<metal_mmap_region> _region;

  // Turner approach 2 (https://tallendev.github.io/assets/papers/sc21.pdf):
  //
  // delta = gpuAddress - hostAddress.  Set on the first shared/host allocation
  // and kept constant for all subsequent allocations via the retry mechanism.
  int64_t  _usm_delta        = 0;
  bool     _usm_delta_valid  = false;

  struct usm_block {
    MTL::Buffer*   buffer;
    size_t         size;       // bytes reserved in the mmap region (stride, not user size)
    usm_alloc_type alloc_type;
    bool           is_slab = false; // true if this block is managed by _slab_alloc
  };
  std::map<void*, usm_block> _ptr_to_block;
  mutable std::mutex _mutex;

  // Bitmap-only slab manager for small USM allocations.
  // Slab Metal buffers are registered in _ptr_to_block with is_slab=true.
  // All methods are called with _mutex held.
  metal_slab_allocator _slab_alloc;

  // Shared implementation for raw_allocate_usm and raw_allocate_optimized_host
  // for large (non-slab) allocations.
  // opts is MTL::ResourceOptions (unsigned long) to avoid Metal headers here.
  void *alloc_from_region(std::size_t size_bytes,
                          unsigned long opts,
                          usm_alloc_type type);

  // Creates a slab Metal buffer, registers it in _ptr_to_block, and returns
  // the first slot.  Used for small allocations.
  void *alloc_from_slab(std::size_t size_bytes,
                        unsigned long opts,
                        usm_alloc_type type);
};



} // namespace rt
} // namespace hipsycl

#endif
