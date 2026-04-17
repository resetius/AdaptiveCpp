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
#include <iostream>

namespace hipsycl {
namespace rt {

static constexpr std::array<size_t, 6> slab_classes = {1024, 1024*4, 1024*16, 1024*64, 1024*256, 1024*1024};
static constexpr size_t slab_buffer_size = 1024 * 1024 * 64;

struct metal_slab_meta {
  size_t _class; // slab page in bytes
  size_t _free_slots;
  std::vector<uint64_t> _free_mask;

  metal_slab_meta(size_t cls)
    : _class(cls)
    , _free_slots(slab_buffer_size / _class)
    , _free_mask((slab_buffer_size / _class + 63) / 64, ~0ull)
  { }

  bool is_full() const {
    return _free_slots == 0;
  }

  bool is_empty() const {
    return _free_slots == slab_buffer_size / _class;
  }

  size_t alloc_offset() {
    for (size_t i = 0; i < _free_mask.size(); ++i) {
      if (_free_mask[i]) {
        size_t bit_pos = __builtin_ctzll(_free_mask[i]);
        _free_mask[i] &= ~(1ull << bit_pos);
        --_free_slots;
        return (i * 64 + bit_pos) * _class;
      }
    }
    return -1;
  }

  void free_offset(size_t offset) {
    size_t bit_idx = offset / _class;
    size_t mask_idx = bit_idx / 64;
    size_t bit_pos = bit_idx % 64;
    _free_mask[mask_idx] |= (1ull << bit_pos);
    ++_free_slots;
  }
};

metal_allocator::metal_allocator(MTL::Device* device, const device_id &id)
  : _device{device}, _device_id{id}
{}

metal_allocator::~metal_allocator() = default;

void* metal_allocator::raw_allocate(
  size_t min_alignment, size_t size_bytes,
  const allocation_hints &hints)
{
  auto block = get_block(size_bytes, alloc_type::device);
  return block.buffer ? reinterpret_cast<char*>(block.buffer->gpuAddress()) + block.offset : nullptr;
}

void *metal_allocator::raw_allocate_usm(
  size_t size_bytes,
  const allocation_hints &hints)
{
  auto block = get_block(size_bytes, alloc_type::shared);
  return block.buffer ? reinterpret_cast<char*>(block.buffer->contents()) + block.offset : nullptr;
}

void *
metal_allocator::raw_allocate_optimized_host(
  size_t min_alignment, size_t size_bytes,
  const allocation_hints &hints)
{
  auto block = get_block(size_bytes, alloc_type::host);
  return block.buffer ? reinterpret_cast<char*>(block.buffer->contents()) + block.offset : nullptr;
}

void metal_allocator::raw_free(void *mem)
{
  if (!mem) return;
  auto block = get_block(mem);
  if (block.alloc_type != alloc_type::undefined) {
    free_block(block);
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

metal_allocator::storage_type::const_iterator metal_allocator::allocate_block(size_t bytes, alloc_type alloc_type) {
  std::lock_guard<std::mutex> lock{_mutex};
  return allocate_block_unlocked(bytes, alloc_type);
}

metal_allocator::storage_type::const_iterator metal_allocator::allocate_block_unlocked(size_t bytes, alloc_type alloc_type) {
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
  auto [position, _] = _ptr_to_block.emplace(ptr, std::move(block));
  return position;
}

metal_allocator::storage_type::const_iterator metal_allocator::get_slab_unlocked(size_t slab_class, alloc_type alloc_type) {
  auto it = _slab_blocks.find({slab_class, alloc_type});
  if (it != _slab_blocks.end()) {
    return *it->second.begin();
  }
  return _ptr_to_block.end();
}

metal_allocator::storage_type::const_iterator metal_allocator::allocate_slab_unlocked(size_t slab_class, alloc_type alloc_type) {
  auto block = allocate_block_unlocked(slab_buffer_size, alloc_type);
  if (block == _ptr_to_block.end()) {
    return _ptr_to_block.end();
  }
  auto slab_meta = std::make_unique<metal_slab_meta>(slab_class);
  block->second.slab_meta = std::move(slab_meta);
  _slab_blocks[std::make_pair(slab_class, alloc_type)].emplace(block);
  return block;
}

metal_allocator::storage_type::const_iterator metal_allocator::get_or_allocate_slab_unlocked(size_t slab_class, alloc_type alloc_type) {
  auto res = get_slab_unlocked(slab_class, alloc_type);
  if (res != _ptr_to_block.end()) {
    return res;
  }
  return allocate_slab_unlocked(slab_class, alloc_type);
}

metal_allocator::block metal_allocator::get_block(size_t bytes, alloc_type alloc_type, bool allow_slab) {
  if (allow_slab && bytes <= slab_classes.back()) {
    // slab allocation
    size_t slab_class = 0;
    while (slab_class < slab_classes.size() && slab_classes[slab_class] < bytes) {
      ++slab_class;
    }
    if (slab_class == slab_classes.size()) {
      return {nullptr, 0, alloc_type::undefined, _ptr_to_block.end()};
    }
    slab_class = slab_classes[slab_class];

    std::lock_guard<std::mutex> lock{_mutex};
    auto slab_it = get_or_allocate_slab_unlocked(slab_class, alloc_type);
    if (slab_it == _ptr_to_block.end()) {
      return {nullptr, 0, alloc_type::undefined, _ptr_to_block.end()};
    }
    auto& slab_meta = slab_it->second.slab_meta;
    size_t offset = slab_meta->alloc_offset();
    if (slab_meta->is_full()) {
      _slab_full_blocks[std::make_pair(slab_class, alloc_type)].emplace(slab_it);
      auto slab_blocks_it = _slab_blocks.find(std::make_pair(slab_class, alloc_type));
      if (slab_blocks_it != _slab_blocks.end()) {
        slab_blocks_it->second.erase(slab_it);
        if (slab_blocks_it->second.empty()) {
          _slab_blocks.erase(slab_blocks_it);
        }
      }
    }
    return {slab_it->second.buffer, offset, alloc_type, slab_it};
  } else {
    auto it = allocate_block(bytes, alloc_type);
    if (it == _ptr_to_block.end()) {
      return {nullptr, 0, alloc_type::undefined, it};
    }
    return {it->second.buffer, 0, alloc_type, it};
  }
}

void metal_allocator::free_block(block block) {
  if (block.buffer && block.position != _ptr_to_block.end()) {
    std::lock_guard<std::mutex> lock{_mutex};
    if (block.position->second.slab_meta) {
      auto& slab_meta = block.position->second.slab_meta;
      if (slab_meta->is_full()) {
        _slab_blocks[std::make_pair(slab_meta->_class, block.alloc_type)].emplace(block.position);
        auto full_blocks_it = _slab_full_blocks.find(std::make_pair(slab_meta->_class, block.alloc_type));
        if (full_blocks_it != _slab_full_blocks.end()) {
          full_blocks_it->second.erase(block.position);
          if (full_blocks_it->second.empty()) {
            _slab_full_blocks.erase(full_blocks_it);
          }
        }
      }
      slab_meta->free_offset(block.offset);
      if (slab_meta->is_empty()) {
        auto slab_blocks_it = _slab_blocks.find(std::make_pair(slab_meta->_class, block.alloc_type));
        if (slab_blocks_it != _slab_blocks.end()) {
          slab_blocks_it->second.erase(block.position);
          if (slab_blocks_it->second.empty()) {
            _slab_blocks.erase(slab_blocks_it);
          }
        }
        block.buffer->release();
        _ptr_to_block.erase(block.position);
      }
    } else {
      block.buffer->release();
      _ptr_to_block.erase(block.position);
    }
  }
}


} // namespace rt
} // namespace hipsycl
