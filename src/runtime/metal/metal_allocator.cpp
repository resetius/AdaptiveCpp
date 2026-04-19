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
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <iostream>

namespace hipsycl {
namespace rt {

namespace {
  uintptr_t align_up(uintptr_t v, size_t align) {
    return (v + align - 1) & ~(uintptr_t)(align - 1);
  }

  size_t get_total_ram() {
    uint64_t mem = 0;
    std::size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0)
      return static_cast<std::size_t>(mem);
    return 8ULL << 30; // fallback: 8 GiB
  }

  size_t metal_gpu_stride(size_t sz, size_t page_size) {
    const size_t two_pages = 2 * page_size;
    return (sz + page_size + two_pages - 1) & ~(two_pages - 1);
  }
}

static constexpr std::array<size_t, 6> slab_classes = {1024, 1024*4, 1024*16, 1024*64, 1024*256, 1024*1024};
static constexpr size_t slab_buffer_size = 1024 * 1024 * 64;
static constexpr double mmap_region_size_fraction = 1.0;

struct metal_mmap_region {
  metal_mmap_region(size_t capacity, size_t alignment)
    : _mmap_size(capacity), _capacity(capacity), _alignment(alignment)
  {
    void* ptr = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (ptr == MAP_FAILED) {
      throw std::runtime_error("metal_mmap_region: mmap failed");
    }

    _base = static_cast<char*>(ptr);
    _current = reinterpret_cast<char*>(align_up(reinterpret_cast<uintptr_t>(_base), _alignment));
    _capacity = capacity - (_current - _base);
  }

  ~metal_mmap_region() {
    if (_base) {
      munmap(_base, _mmap_size); // use original mmap size, not the modified _capacity
    }
  }

  // size must be a multiple of _alignment (page-aligned)
  void* alloc(size_t size) {
    if (size == 0) {
      return nullptr;
    }
    size = align_up(size, _alignment);

    for (auto it = _free_blocks.begin(); it != _free_blocks.end(); ++it) {
      if (it->second >= size) {
        void* ptr = it->first;
        size_t remaining = it->second - size;
        _free_blocks.erase(it);
        if (remaining > 0) {
          _free_blocks.emplace(static_cast<char*>(ptr) + size, remaining);
        }
        return ptr;
      }
    }

    if (_current + size > _base + _capacity) {
      return nullptr;
    }

    void* ptr = _current;
    _current += size;
    return ptr;
  }

  // assume addr already aligned
  void* alloc_at(void* addr, size_t size) {
    if (size == 0) {
      return nullptr;
    }

    uintptr_t aligned_addr = align_up(reinterpret_cast<uintptr_t>(addr), _alignment);
    if (aligned_addr != reinterpret_cast<uintptr_t>(addr)) {
      return nullptr;
    }

    // 1. check _current
    if (addr >= _current) {
      if (static_cast<char*>(addr) + size > _base + _capacity) {
        return nullptr;
      }
      if (static_cast<char*>(addr) - _current > 0) {
        _free_blocks.emplace(_current, static_cast<char*>(addr) - _current);
      }
      _current = static_cast<char*>(addr) + size;
      return addr;
    }

    // 2. check free blocks
    auto it = _free_blocks.upper_bound(addr);
    if (it == _free_blocks.begin()) {
      return nullptr;
    }
    --it;
    char* ptr = static_cast<char*>(it->first);
    if (ptr + size <= static_cast<char*>(addr)) {
      return nullptr;
    }
    if (size - (static_cast<char*>(addr) - ptr) > it->second) {
      return nullptr;
    }
    size_t remaining = (ptr + it->second) - (static_cast<char*>(addr) + size);
    _free_blocks.erase(it);
    if (remaining > 0) {
      _free_blocks.emplace(static_cast<char*>(addr) + size, remaining);
    }
    if (ptr < static_cast<char*>(addr)) {
      _free_blocks.emplace(ptr, static_cast<char*>(addr) - ptr);
    }
    return addr;
  }

  void free(void* ptr, size_t size) {
    if (!ptr || size == 0) {
      return;
    }

    void* end = static_cast<char*>(ptr) + size;

    auto [it, _] = _free_blocks.emplace(ptr, size);

    // try to merge with next block
    auto next = std::next(it);
    if (next != _free_blocks.end() && next->first == end) {
      it->second += next->second;
      _free_blocks.erase(next);
    }

    // try to merge with previous block
    if (it != _free_blocks.begin()) {
      auto prev = std::prev(it);
      if (static_cast<char*>(prev->first) + prev->second == it->first) {
        prev->second += it->second;
        _free_blocks.erase(it);
      }
    }

    while (!_free_blocks.empty()) {
      auto last = std::prev(_free_blocks.end());
      if (static_cast<char*>(last->first) + last->second == _current) {
        _current = static_cast<char*>(last->first);
        _free_blocks.erase(last);
      } else {
        break;
      }
    }
  }

  char* _base;
  char* _current;
  size_t _mmap_size;   // original mmap size for munmap
  size_t _capacity;
  size_t _alignment;

  std::map<void*, size_t> _free_blocks;
};

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
  : _device{device}, _device_id{id}, _mmap_region(std::make_unique<metal_mmap_region>(static_cast<size_t>(get_total_ram() * mmap_region_size_fraction), getpagesize()))
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
  MTL::Buffer* buffer;
  void* ptr;
  size_t mmap_size = 0;

  if (alloc_type == alloc_type::device) {
    buffer = _device->newBuffer(bytes, MTL::ResourceStorageModePrivate);
    if (!buffer) return _ptr_to_block.end();
    ptr = reinterpret_cast<char*>(buffer->gpuAddress());
  } else {
    const size_t page_size = static_cast<size_t>(getpagesize());
    const size_t aligned = align_up(bytes, page_size);
    const size_t stride  = metal_gpu_stride(aligned, page_size);
    void* region_ptr = _mmap_region->alloc(stride);
    if (!region_ptr) {
      return _ptr_to_block.end();
    }
    buffer = _device->newBuffer(
      region_ptr, aligned, MTL::ResourceStorageModeShared,
      ^(void*, NS::UInteger) {
        std::cerr << "[alloc] deallocator fired: region_ptr=" << region_ptr << " stride=" << stride << "\n";
        std::lock_guard<std::mutex> lock{_mutex};
        _mmap_region->free(region_ptr, stride);
      });
    if (!buffer) {
      _mmap_region->free(region_ptr, stride);
      return _ptr_to_block.end();
    }
    mmap_size = stride;
    ptr = buffer->contents();
    std::cerr << "[alloc] new shared buffer: contents=" << ptr
              << " gpuAddr=" << reinterpret_cast<void*>(buffer->gpuAddress())
              << " aligned=" << aligned << " stride=" << stride << "\n";
  }

  auto block = raw_block {
    .buffer = buffer,
    .alloc_type = alloc_type,
    .mmap_size = mmap_size
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
  MTL::Buffer* buffer = nullptr;
  void* key = nullptr;
  bool is_slab = false;
  size_t mmap_sz = 0;
  auto atype = block.alloc_type;
  if (block.buffer && block.position != _ptr_to_block.end()) {
    std::lock_guard<std::mutex> lock{_mutex};
    key = block.position->first;
    is_slab = (block.position->second.slab_meta != nullptr);
    mmap_sz = block.position->second.mmap_size;
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
        buffer = block.buffer;
        _ptr_to_block.erase(block.position);
      }
    } else {
      buffer = block.buffer;
      _ptr_to_block.erase(block.position);
    }
  }
  if (buffer) {
    std::cerr << "[free] releasing buffer: key=" << key
              << " is_slab=" << is_slab
              << " contents=" << (atype != metal_allocator::alloc_type::device ? buffer->contents() : nullptr)
              << " gpuAddr=" << reinterpret_cast<void*>(buffer->gpuAddress())
              << " mmap_size=" << mmap_sz << "\n";
    // Release outside the mutex: for mmap-backed buffers the deallocator fires
    // when Metal's retain count hits 0 (after the GPU command completes) and
    // it tries to acquire _mutex to free the mmap region — so we must not hold
    // the lock here to avoid deadlock.
    buffer->release();
  }
}


} // namespace rt
} // namespace hipsycl
