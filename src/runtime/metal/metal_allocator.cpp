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
#include "hipSYCL/common/debug.hpp"

#include <Metal/Metal.hpp>

#include <sys/sysctl.h> // sysctlbyname (hw.memsize)

namespace hipsycl {
namespace rt {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Returns total physical RAM in bytes via sysctl.
static std::size_t get_total_ram() {
  uint64_t mem = 0;
  std::size_t len = sizeof(mem);
  if (::sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0)
    return static_cast<std::size_t>(mem);
  return 8ULL << 30; // fallback: 8 GiB
}

// Page size used for USM alignment (Metal requires page-aligned noCopy buffers).
static constexpr std::size_t kPageSize = 4096;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

metal_allocator::metal_allocator(MTL::Device *device, const device_id &id)
    : _device{device}, _device_id{id} {
  // Reserve half of physical RAM as the USM backing region.
  // The region uses PROT_NONE initially; pages are committed on demand.
  // The cursor starts from the middle of the region (Turner approach 2:
  // https://tallendev.github.io/assets/papers/sc21.pdf).
  std::size_t region_size = get_total_ram() / 2;
  _region = std::make_unique<metal_mmap_region>(region_size);
  //std::cerr << "[metal_allocator] region: " << (region_size >> 20)
  //          << " MiB at " << _region->base() << "\n";
}

metal_allocator::~metal_allocator() {
  // Release all outstanding Metal buffers before the mmap region is unmapped.
  std::lock_guard<std::mutex> lock{_mutex};
  for (auto &[ptr, block] : _ptr_to_block)
    if (block.buffer)
      block.buffer->release();
  _ptr_to_block.clear();
}

// ---------------------------------------------------------------------------
// Shared/host USM allocation — backed by the mmap region.
//
// Procedure:
//   1. Sub-allocate from _region aligned to gpu_page_size (Turner approach 2,
//      https://tallendev.github.io/assets/papers/sc21.pdf): rounding both size
//      and address to the GPU VA granularity makes CPU stride == GPU stride, so
//      delta = gpuAddress - hostAddress is the same for every allocation.
//   2. Create a MTL::Buffer over the sub-range using newBufferWithBytesNoCopy.
//      The deallocator block is a no-op: memory lifetime is managed by _region.
//   3. Record the host pointer → {buffer, size, type} in _ptr_to_block.
//      The global _usm_delta was already set in the constructor from the probe.
// ---------------------------------------------------------------------------

void *metal_allocator::alloc_from_region(std::size_t size_bytes,
                                         unsigned long opts,
                                         usm_alloc_type type) {
  // Round size up to kPageSize so Metal's noCopy path gets page-aligned ranges.
  std::size_t size = ((size_bytes + kPageSize - 1) / kPageSize) * kPageSize;

  // Grab _usm_delta / _usm_delta_valid under the lock once so we can use them
  // outside the lock (they are written at most once, before being read).
  int64_t  expected_delta = 0;
  bool     delta_valid    = false;
  {
    std::lock_guard<std::mutex> lk{_mutex};
    expected_delta = _usm_delta;
    delta_valid    = _usm_delta_valid;
  }

  // -------------------------------------------------------------------------
  // Turner approach 2 (https://tallendev.github.io/assets/papers/sc21.pdf):
  //
  //   Each newBufferWithBytesNoCopy call advances Metal's internal GPU VA
  //   cursor by an unpredictable stride (depends on Metal version, size,
  //   alignment).  The delta (gpuAddress - hostAddress) therefore varies per
  //   allocation.
  //
  //   Fix: after the first allocation establishes `_usm_delta`, every
  //   subsequent allocation checks whether the new buffer's delta matches.
  //   On mismatch:
  //     1. Release the mismatched buffer — Metal immediately puts its GPU VA
  //        back at the head of its free list.
  //     2. Compute corrected_cpu = observed_gpu - _usm_delta.
  //        Because Metal reuses the just-freed GPU VA for the very next
  //        newBuffer call, creating a buffer at corrected_cpu will receive
  //        observed_gpu and delta == _usm_delta exactly.
  //     3. commit_at(corrected_cpu) and retry.
  // -------------------------------------------------------------------------

  // Step 1: pick a candidate address from the region.
  void *host_ptr = _region->alloc(size, kPageSize);
  if (!host_ptr)
    return nullptr;

  // Empty deallocator: Metal must not free the mmap pages.
  auto make_buf = [&](void *ptr) -> MTL::Buffer * {
    return _device->newBuffer(
        ptr, size, static_cast<MTL::ResourceOptions>(opts),
        ^(void *, NS::UInteger) { /* no-op: _region owns the memory */ });
  };

  MTL::Buffer *buf = make_buf(host_ptr);
  if (!buf) {
    _region->free(host_ptr, size);
    return nullptr;
  }

  uint64_t gpu_addr  = buf->gpuAddress();
  uint64_t host_addr = reinterpret_cast<uint64_t>(host_ptr);
  int64_t  delta     = static_cast<int64_t>(gpu_addr - host_addr);

  //std::cerr << "[alloc_from_region]"
  //          << " host=0x" << std::hex << host_addr
  //          << " gpu=0x"  << gpu_addr
  //          << " delta=0x" << static_cast<uint64_t>(delta)
  //          << std::dec << "\n";

  // Step 2: retry on delta mismatch.
  if (delta_valid && delta != expected_delta) {
    // Release the mismatched buffer so Metal reuses its GPU VA slot.
    buf->release();
    buf = nullptr;
    _region->free(host_ptr, size);
    host_ptr = nullptr;

    // corrected_cpu: the CPU address such that newBuffer will assign
    // observed_gpu (the just-freed VA) → delta == expected_delta.
    auto corrected_cpu = static_cast<uintptr_t>(
        static_cast<int64_t>(gpu_addr) - expected_delta);

    //std::cerr << "[alloc_from_region] delta mismatch — retrying at corrected"
    //          << " cpu=0x" << std::hex << corrected_cpu
    //          << " (expected delta=0x" << static_cast<uint64_t>(expected_delta)
    //          << ")" << std::dec << "\n";

    void *corrected = _region->alloc_at(reinterpret_cast<void *>(corrected_cpu),
                                        size);
    if (!corrected)
      return nullptr;

    host_ptr = corrected;
    buf      = make_buf(host_ptr);
    if (!buf) {
      _region->free(host_ptr, size);
      return nullptr;
    }

    gpu_addr  = buf->gpuAddress();
    host_addr = reinterpret_cast<uint64_t>(host_ptr);
    delta     = static_cast<int64_t>(gpu_addr - host_addr);

    //std::cerr << "[alloc_from_region] retry result:"
    //          << " host=0x" << std::hex << host_addr
    //          << " gpu=0x"  << gpu_addr
    //          << " delta=0x" << static_cast<uint64_t>(delta)
    //          << std::dec << "\n";
  }

  std::lock_guard<std::mutex> lock{_mutex};

  if (!_usm_delta_valid) {
    _usm_delta       = delta;
    _usm_delta_valid = true;
    HIPSYCL_DEBUG_INFO << "metal_allocator: USM delta = " << delta
                       << " (gpu=0x" << std::hex << gpu_addr
                       << " host=0x" << host_addr << std::dec << ")\n";
  }

  _ptr_to_block[host_ptr] = {buf, size, type};
  return host_ptr;
}

// ---------------------------------------------------------------------------
// Public allocation entry points
// ---------------------------------------------------------------------------

void *metal_allocator::raw_allocate(std::size_t min_alignment,
                                    std::size_t size_bytes,
                                    const allocation_hints &) {
  // Device-only (StorageModePrivate) allocations do not need CPU access and
  // therefore cannot use the noCopy path.  Keep the original approach.
  auto *buf = _device->newBuffer(size_bytes, MTL::ResourceStorageModePrivate);
  void *gpu_ptr = reinterpret_cast<void *>(buf->gpuAddress());
  std::lock_guard<std::mutex> lock{_mutex};
  _ptr_to_block[gpu_ptr] = {buf, size_bytes, usm_alloc_type::device};
  return gpu_ptr;
}

void *metal_allocator::raw_allocate_usm(std::size_t size_bytes,
                                        const allocation_hints &) {
  return alloc_from_region(size_bytes, MTL::ResourceStorageModeShared,
                           usm_alloc_type::shared);
}

void *metal_allocator::raw_allocate_optimized_host(std::size_t,
                                                   std::size_t size_bytes,
                                                   const allocation_hints &) {
  return alloc_from_region(size_bytes, MTL::ResourceStorageModeShared,
                           usm_alloc_type::host);
}

// ---------------------------------------------------------------------------
// Free
//
// 1. Release the MTL::Buffer — drops the GPU-address mapping.
//    Because the deallocator was a no-op, Metal does NOT touch the mmap pages.
// 2. Return the mmap pages to the region free-list via region->free().
// ---------------------------------------------------------------------------

void metal_allocator::raw_free(void *mem) {
  if (!mem) return;

  MTL::Buffer *buf  = nullptr;
  std::size_t  size = 0;
  bool         from_region = false;

  {
    std::lock_guard<std::mutex> lock{_mutex};
    auto it = _ptr_to_block.find(mem);
    if (it == _ptr_to_block.end())
      return;

    buf  = it->second.buffer;
    size = it->second.size;
    from_region = (it->second.alloc_type != usm_alloc_type::device);
    _ptr_to_block.erase(it);
  }

  if (buf)
    buf->release();       // drop GPU mapping (mmap pages stay alive)

  if (from_region)
    _region->free(mem, size); // return pages to free-list
}

// ---------------------------------------------------------------------------
// Remaining interface (unchanged)
// ---------------------------------------------------------------------------

bool metal_allocator::is_usm_accessible_from(backend_descriptor b) const {
  return b.id == backend_id::metal;
}

result metal_allocator::query_pointer(const void *ptr,
                                      pointer_info &out) const {
  memset(&out, 0, sizeof(pointer_info));
  out.dev = _device_id;
  if (!ptr)
    return make_error(__acpp_here(),
                      error_info{"metal_allocator: Null pointer queried"});

  auto [buffer, offset, alloc_type] = get_usm_block(ptr);
  if (alloc_type == usm_alloc_type::undefined)
    return make_error(__acpp_here(),
                      error_info{"metal_allocator: Pointer is unknown"});

  if (alloc_type == usm_alloc_type::host)
    out.is_optimized_host = true;
  if (alloc_type == usm_alloc_type::shared)
    out.is_usm = true;

  return make_success();
}

result metal_allocator::mem_advise(const void *, std::size_t, int) const {
  return make_success();
}

device_id metal_allocator::get_device() const { return _device_id; }

std::tuple<MTL::Buffer *, size_t, metal_allocator::usm_alloc_type>
metal_allocator::get_usm_block(const void *ptr) const {
  std::lock_guard<std::mutex> lock{_mutex};
  if (_ptr_to_block.empty())
    return {nullptr, 0, usm_alloc_type::undefined};

  auto it = _ptr_to_block.upper_bound(const_cast<void *>(ptr));
  if (it == _ptr_to_block.begin())
    return {nullptr, 0, usm_alloc_type::undefined};
  --it;

  const usm_block &block = it->second;
  std::size_t offset =
      static_cast<const char *>(ptr) - static_cast<const char *>(it->first);
  if (offset < block.size)
    return {block.buffer, offset, block.alloc_type};

  return {nullptr, 0, usm_alloc_type::undefined};
}

int64_t metal_allocator::usm_delta() const {
  std::lock_guard<std::mutex> lock{_mutex};
  return _usm_delta_valid ? _usm_delta : 0;
}

} // namespace rt
} // namespace hipsycl
