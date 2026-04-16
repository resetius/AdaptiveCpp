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

#include "hipSYCL/runtime/metal/metal_allocator.hpp"

#include <Metal/Metal.hpp>

namespace hipsycl {
namespace rt {

metal_allocator::metal_allocator(MTL::Device* device, const device_id &id)
  : _device{device}, _device_id{id}
{}

metal_allocator::~metal_allocator() = default;

void* metal_allocator::raw_allocate(
  size_t min_alignment, size_t size_bytes,
  const allocation_hints &hints)
{
  auto block = get_block(size_bytes, alloc_type::device);
  return block.buffer ? reinterpret_cast<void*>(block.buffer->gpuAddress()) : nullptr;
}

void *metal_allocator::raw_allocate_usm(
  size_t size_bytes,
  const allocation_hints &hints)
{
  auto block = get_block(size_bytes, alloc_type::shared);
  return block.buffer ? block.buffer->contents() : nullptr;
}

void *
metal_allocator::raw_allocate_optimized_host(
  size_t min_alignment, size_t size_bytes,
  const allocation_hints &hints)
{
  auto block = get_block(size_bytes, alloc_type::host);
  return block.buffer ? block.buffer->contents() : nullptr;
}

void metal_allocator::raw_free(void *mem)
{
  if (!mem) return;

  std::lock_guard<std::mutex> lock{_mutex};
  auto it = _ptr_to_block.find(mem);
  if (it != _ptr_to_block.end()) {
    if(it->second.buffer) {
      it->second.buffer->release();
    } else {
      std::free(mem);
    }
    _ptr_to_block.erase(it);
  }
}

bool metal_allocator::is_usm_accessible_from(backend_descriptor b) const
{
  return b.id == backend_id::metal;
}

result metal_allocator::query_pointer(
  const void *ptr,
  pointer_info &out) const
{
  memset(&out, 0, sizeof(pointer_info));
  out.dev = _device_id;
  if (!ptr) {
    return make_error(__acpp_here(),
      error_info{"metal_allocator: Null pointer queried"});
  }
  auto block = get_block(ptr);
  if (block.alloc_type == alloc_type::undefined) {
    return make_error(__acpp_here(),
      error_info{"metal_allocator: Pointer is unknown"});
  }
  if (block.alloc_type == alloc_type::host) {
    out.is_optimized_host = true;
    return make_success();
  }
  if (block.alloc_type == alloc_type::shared) {
    out.is_usm = true;
    return make_success();
  }

  return make_success();
}

result metal_allocator::mem_advise(
  const void *addr, std::size_t num_bytes,
  int advise) const
{
  return make_success();
}

device_id metal_allocator::get_device() const {
  return _device_id;
}

metal_allocator::block metal_allocator::get_block(const void* ptr) const {
  std::lock_guard<std::mutex> lock{_mutex};
  if (_ptr_to_block.empty()) {
    return {nullptr, 0, alloc_type::undefined};
  }
  auto it = _ptr_to_block.upper_bound(const_cast<void*>(ptr));
  if (it == _ptr_to_block.begin()) {
    return {nullptr, 0, alloc_type::undefined};
  }
  --it;
  const raw_block& block = it->second;
  size_t offset = static_cast<const char*>(ptr) -
          static_cast<const char*>(it->first);
  if (offset < block.buffer->length()) {
    return {block.buffer, offset, block.alloc_type, it};
  }
  return {nullptr, 0, alloc_type::undefined, _ptr_to_block.end()};
}

metal_allocator::block metal_allocator::get_block(size_t bytes, alloc_type alloc_type) {
  auto storage_mode = MTL::ResourceStorageModeShared;
  if (alloc_type == alloc_type::device) {
    storage_mode = MTL::ResourceStorageModePrivate;
  }
  auto buffer = _device->newBuffer(bytes, storage_mode);
  void* ptr = alloc_type == alloc_type::device ? (char*)buffer->gpuAddress() : buffer->contents();
  auto block = raw_block {
    .buffer = buffer,
    .alloc_type = alloc_type
  };
  std::lock_guard<std::mutex> lock{_mutex};
  auto [position, _] = _ptr_to_block.emplace(ptr, block);
  return {block.buffer, 0, alloc_type, position};
}

void metal_allocator::free_block(block block) {
  if (block.buffer && block.position != _ptr_to_block.end() && block.offset == 0) {
    block.buffer->release();
    std::lock_guard<std::mutex> lock{_mutex};
    _ptr_to_block.erase(block.position);
  }
}


} // namespace rt
} // namespace hipsycl
