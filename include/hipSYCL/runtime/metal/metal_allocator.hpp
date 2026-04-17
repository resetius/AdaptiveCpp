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

#include <map>
#include <unordered_set>

namespace MTL {

class Device;
class Buffer;

} // namespace MTL

namespace hipsycl {
namespace rt {

template <class T, class Deleter>
class unique_resource {
public:
  unique_resource() = default;

  unique_resource(T value, Deleter deleter)
      : value_(value), deleter_(std::move(deleter)), has_value_(true)
  {}

  unique_resource(const unique_resource&) = delete;
  unique_resource& operator=(const unique_resource&) = delete;

  unique_resource(unique_resource&& other) noexcept
      : value_(std::move(other.value_))
      , deleter_(std::move(other.deleter_))
      , has_value_(std::exchange(other.has_value_, false))
  {}

  unique_resource& operator=(unique_resource&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::move(other.value_);
      deleter_ = std::move(other.deleter_);
      has_value_ = std::exchange(other.has_value_, false);
    }
    return *this;
  }

  ~unique_resource() {
    reset();
  }

  T& get() noexcept { return value_; }
  const T& get() const noexcept { return value_; }
  const T* operator->() const noexcept { return &value_; }
  T* operator->() noexcept { return &value_; }

  void reset() noexcept {
    if (has_value_) {
      deleter_(value_);
      has_value_ = false;
    }
  }

  T release() noexcept {
    has_value_ = false;
    return std::move(value_);
  }

private:
  T value_{};
  Deleter deleter_{};
  bool has_value_ = false;
};

struct metal_slab_meta;

class metal_allocator : public backend_allocator
{
  struct raw_block;
  using storage_type = std::map<void*, raw_block>;

public:
  enum class alloc_type {
    shared = 0,
    device = 1,
    host = 2,
    undefined = 3
  };

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

  struct block {
    MTL::Buffer* buffer;
    size_t offset;
    alloc_type alloc_type;
    storage_type::const_iterator position;
  };
  // Returns the Metal buffer and offset for a given USM pointer
  block get_block(const void* ptr) const;
  // Returns buffer for temporary allocations used by metal_query
  block get_block(size_t bytes, alloc_type alloc_type, bool allow_slab = true);
  void free_block(block block);

  struct block_deleter {
    metal_allocator* owner = nullptr;

    void operator()(block b) const {
      owner->free_block(b);
    }
  };

  using owned_block = unique_resource<block, block_deleter>;

  owned_block get_owned_block(size_t bytes, alloc_type alloc_type, bool allow_slab = true) {
    auto b = get_block(bytes, alloc_type, allow_slab);
    return unique_resource<block, block_deleter>(b, block_deleter{this});
  }

private:
  storage_type::const_iterator allocate_block(size_t bytes, alloc_type alloc_type);

  storage_type::const_iterator allocate_block_unlocked(size_t bytes, alloc_type alloc_type);
  storage_type::const_iterator get_slab_unlocked(size_t slab_class, alloc_type alloc_type);
  storage_type::const_iterator allocate_slab_unlocked(size_t slab_class, alloc_type alloc_type);
  storage_type::const_iterator get_or_allocate_slab_unlocked(size_t slab_class, alloc_type alloc_type);

  MTL::Device* _device = nullptr;
  device_id _device_id;

  mutable std::mutex _mutex;

  struct raw_block {
    MTL::Buffer* buffer;
    alloc_type alloc_type;
    mutable std::unique_ptr<metal_slab_meta> slab_meta;
  };
  storage_type _ptr_to_block;
  struct iterator_hash {
    size_t operator()(const storage_type::const_iterator& it) const {
      return std::hash<void*>()(it->first);
    }
  };
  std::map<std::pair<size_t, alloc_type>, std::unordered_set<storage_type::const_iterator, iterator_hash>> _slab_blocks;
  std::map<std::pair<size_t, alloc_type>, std::unordered_set<storage_type::const_iterator, iterator_hash>> _slab_full_blocks;
};



} // namespace rt
} // namespace hipsycl

#endif
