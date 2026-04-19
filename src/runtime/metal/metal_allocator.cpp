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
#include "hipSYCL/runtime/metal/metal_slab_allocator.hpp"
#include "hipSYCL/common/debug.hpp"

#include <Metal/Metal.hpp>

#include <sys/sysctl.h> // sysctlbyname (hw.memsize)
#include <unistd.h>    // getpagesize()

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

// Page size for USM alignment.  Metal's newBufferWithBytesNoCopy requires the
// pointer and length to be aligned to the system page size.  On Apple Silicon
// (arm64 macOS) getpagesize() returns 16384; on x86_64 macOS it returns 4096.
// Using the runtime value avoids silent size mismatches.
static std::size_t get_page_size() {
  static const std::size_t kPageSize = static_cast<std::size_t>(::getpagesize());
  return kPageSize;
}

// Metal advances its GPU VA cursor by round_up(sz + page, 2*page) for each
// newBufferWithBytesNoCopy call — it always inserts at least one guard page
// and aligns the stride to a 2-page boundary.  Advancing the CPU bump cursor
// by the same stride keeps delta = gpuAddress - hostAddress constant across
// allocations without any Turner retry.
static std::size_t metal_gpu_stride(std::size_t sz, std::size_t page_size) {
  const std::size_t two_pages = 2 * page_size;
  return (sz + page_size + two_pages - 1) & ~(two_pages - 1);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

metal_allocator::metal_allocator(MTL::Device *device, const device_id &id)
    : _device{device}, _device_id{id} {
  // Reserve half of physical RAM as the USM backing region (PROT_READ|PROT_WRITE,
  // demand-paged).  The cursor starts from the middle of the region so that
  // Turner approach 2 corrected_cpu addresses have room in both directions.
  // See: https://tallendev.github.io/assets/papers/sc21.pdf
  std::size_t region_size = get_total_ram() / 2;
  _region = std::make_unique<metal_mmap_region>(region_size);
  HIPSYCL_DEBUG_INFO << "metal_allocator: reserved " << (region_size >> 20)
                     << " MiB USM region at " << _region->base() << "\n";
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
// Core buffer allocation helper — allocates one Metal buffer backed by the
// mmap region, using Turner approach 2 to keep delta constant.
//
// Returns {host_ptr, buf, stride} on success; {nullptr, nullptr, 0} on failure.
// Registers nothing in _ptr_to_block — caller does that.
// Called WITHOUT _mutex held.
// ---------------------------------------------------------------------------

struct AllocResult {
  void        *host_ptr;
  MTL::Buffer *buf;
  std::size_t  stride;
};

static AllocResult
alloc_region_buffer(MTL::Device *device, metal_mmap_region *region,
                    int64_t &usm_delta, bool &usm_delta_valid,
                    std::mutex &mutex,
                    std::size_t size_bytes, unsigned long opts) {
  const std::size_t page_size = get_page_size();
  // Round size up to the system page size so Metal's noCopy path gets
  // page-aligned ranges (required by newBufferWithBytesNoCopy).
  std::size_t size = ((size_bytes + page_size - 1) / page_size) * page_size;

  // CPU region stride mirrors Metal's GPU VA stride so that delta stays
  // constant by construction.
  std::size_t stride = metal_gpu_stride(size, page_size);

  // Snapshot delta under the lock.
  int64_t expected_delta = 0;
  bool    delta_valid    = false;
  {
    std::lock_guard<std::mutex> lk{mutex};
    expected_delta = usm_delta;
    delta_valid    = usm_delta_valid;
  }

  // Step 1: pick a candidate address from the region.
  void *host_ptr = region->alloc(stride, page_size);
  if (!host_ptr)
    return {nullptr, nullptr, 0};

  auto make_buf = [&](void *ptr) -> MTL::Buffer * {
    auto buffer = device->newBuffer(
        ptr, size, static_cast<MTL::ResourceOptions>(opts),
        ^(void *, NS::UInteger) {
          std::cerr << "Releasing Metal buffer at " << ptr << "\n";
          /* no-op: _region owns the memory */
    });

    std::cerr << "Allocated Metal buffer " << ptr << " (size=" << size << " stride=" << stride
              << ") with GPU address 0x" << std::hex
              << (buffer ? buffer->gpuAddress() : 0) << std::dec << "\n";
    return buffer;
  };

  MTL::Buffer *buf = make_buf(host_ptr);
  if (!buf) {
    region->free(host_ptr, stride);
    return {nullptr, nullptr, 0};
  }

  uint64_t gpu_addr  = buf->gpuAddress();
  uint64_t host_addr = reinterpret_cast<uint64_t>(host_ptr);
  int64_t  delta     = static_cast<int64_t>(gpu_addr - host_addr);

  // Step 2: Turner approach 2 retry on delta mismatch.
  if (delta_valid && delta != expected_delta) {
    buf->release();
    buf = nullptr;
    // Burn the candidate VA range (do not return to free-list).
    host_ptr = nullptr;

    auto corrected_cpu = static_cast<uintptr_t>(
        static_cast<int64_t>(gpu_addr) - expected_delta);

    HIPSYCL_DEBUG_INFO << "metal_allocator: delta mismatch by "
                       << (delta - expected_delta) / static_cast<int64_t>(page_size)
                       << " pages, retrying at corrected cpu=0x" << std::hex
                       << corrected_cpu << " (got delta=0x"
                       << static_cast<uint64_t>(delta) << " expected=0x"
                       << static_cast<uint64_t>(expected_delta) << ")"
                       << std::dec << "\n";

    void *corrected = region->alloc_at(reinterpret_cast<void *>(corrected_cpu),
                                       stride);
    if (!corrected)
      return {nullptr, nullptr, 0};

    host_ptr = corrected;
    buf      = make_buf(host_ptr);
    if (!buf) {
      region->free(host_ptr, stride);
      return {nullptr, nullptr, 0};
    }

    gpu_addr  = buf->gpuAddress();
    host_addr = reinterpret_cast<uint64_t>(host_ptr);
    delta     = static_cast<int64_t>(gpu_addr - host_addr);

    if (delta != expected_delta) {
      HIPSYCL_DEBUG_WARNING
          << "metal_allocator: retry delta mismatch (got 0x" << std::hex
          << static_cast<uint64_t>(delta) << ", expected 0x"
          << static_cast<uint64_t>(expected_delta) << ") — pointer "
          << "translation may be incorrect for this allocation\n" << std::dec;
    }
  }

  // Record delta on first allocation.
  {
    std::lock_guard<std::mutex> lk{mutex};
    if (!usm_delta_valid) {
      usm_delta       = delta;
      usm_delta_valid = true;
      HIPSYCL_DEBUG_INFO << "metal_allocator: USM delta = " << delta
                         << " (gpu=0x" << std::hex << gpu_addr
                         << " host=0x" << host_addr << std::dec << ")\n";
    }
  }

  return {host_ptr, buf, stride};
}

// ---------------------------------------------------------------------------
// alloc_from_region — large (non-slab) allocations
// ---------------------------------------------------------------------------

void *metal_allocator::alloc_from_region(std::size_t size_bytes,
                                         unsigned long opts,
                                         usm_alloc_type type) {
  auto [host_ptr, buf, stride] =
      alloc_region_buffer(_device, _region.get(), _usm_delta, _usm_delta_valid,
                          _mutex, size_bytes, opts);
  if (!host_ptr)
    return nullptr;

  std::lock_guard<std::mutex> lock{_mutex};
  // Store stride (not size) so raw_free returns the full region extent
  // including guard pages back to the free list.
  _ptr_to_block[host_ptr] = {buf, stride, type, /*is_slab=*/false};
  return host_ptr;
}

// ---------------------------------------------------------------------------
// alloc_from_slab — small allocations
//
// Tries to satisfy the request from an existing slab first.  If no slab has a
// free slot of the right size class, allocates a new 1-MiB Metal buffer (with
// Turner retry), registers it in _ptr_to_block with is_slab=true, calls
// _slab_alloc.register_slab(), and then allocates the first slot.
// ---------------------------------------------------------------------------

void *metal_allocator::alloc_from_slab(std::size_t size_bytes,
                                       unsigned long opts,
                                       usm_alloc_type type) {
  const std::size_t page_size = get_page_size();
  const std::size_t sc        = _slab_alloc.size_class(size_bytes);

  // Fast path: existing slab has a free slot of matching type.
  {
    std::lock_guard<std::mutex> lock{_mutex};
    void *slot = _slab_alloc.try_alloc_slot(sc, type);
    if (slot)
      return slot;
  }

  // Slow path: allocate a new slab buffer (kSlabBytes).
  auto [host_ptr, buf, stride] =
      alloc_region_buffer(_device, _region.get(), _usm_delta, _usm_delta_valid,
                          _mutex, metal_slab_allocator::kSlabBytes, opts);
  if (!host_ptr)
    return nullptr;

  const std::size_t num_slots = metal_slab_allocator::kSlabBytes / sc;

  std::lock_guard<std::mutex> lock{_mutex};
  _ptr_to_block[host_ptr] = {buf, stride, type, /*is_slab=*/true};
  _slab_alloc.register_slab(host_ptr, sc, num_slots, type);

  void *slot = _slab_alloc.try_alloc_slot(sc, type);
  // Must succeed: we just registered a fresh slab.
  return slot;
}

// ---------------------------------------------------------------------------
// Public allocation entry points
// ---------------------------------------------------------------------------

void *metal_allocator::raw_allocate(std::size_t min_alignment,
                                    std::size_t size_bytes,
                                    const allocation_hints &) {
  // Device-only (StorageModePrivate) allocations do not need CPU access and
  // therefore cannot use the noCopy path.
  auto *buf = _device->newBuffer(size_bytes, MTL::ResourceStorageModePrivate);
  if (!buf)
    return nullptr;
  void *gpu_ptr = reinterpret_cast<void *>(buf->gpuAddress());
  std::lock_guard<std::mutex> lock{_mutex};
  _ptr_to_block[gpu_ptr] = {buf, size_bytes, usm_alloc_type::device, /*is_slab=*/false};
  return gpu_ptr;
}

void *metal_allocator::raw_allocate_usm(std::size_t size_bytes,
                                        const allocation_hints &) {
  if (_slab_alloc.is_small(size_bytes, get_page_size()))
    return alloc_from_slab(size_bytes, MTL::ResourceStorageModeShared,
                           usm_alloc_type::shared);
  return alloc_from_region(size_bytes, MTL::ResourceStorageModeShared,
                           usm_alloc_type::shared);
}

void *metal_allocator::raw_allocate_optimized_host(std::size_t,
                                                   std::size_t size_bytes,
                                                   const allocation_hints &) {
  if (_slab_alloc.is_small(size_bytes, get_page_size()))
    return alloc_from_slab(size_bytes, MTL::ResourceStorageModeShared,
                           usm_alloc_type::host);
  return alloc_from_region(size_bytes, MTL::ResourceStorageModeShared,
                           usm_alloc_type::host);
}

// ---------------------------------------------------------------------------
// Free
//
// For slab slots:
//   1. Range-lookup to find the slab block.
//   2. free_slot() in the bitmap; if the slab is now empty, release the Metal
//      buffer, return the mmap pages, remove the slab metadata.
//
// For regular blocks:
//   1. Release the MTL::Buffer — drops GPU-address mapping.
//   2. Return the mmap pages to the region free-list.
// ---------------------------------------------------------------------------

void metal_allocator::raw_free(void *mem) {
  if (!mem) return;

  MTL::Buffer *buf         = nullptr;
  std::size_t  size        = 0;
  bool         from_region = false;

  std::lock_guard<std::mutex> lock{_mutex};

  // First try exact lookup (works for regular non-slab blocks).
  auto exact_it = _ptr_to_block.find(mem);
  if (exact_it != _ptr_to_block.end() && !exact_it->second.is_slab) {
    buf         = exact_it->second.buffer;
    size        = exact_it->second.size;
    from_region = (exact_it->second.alloc_type != usm_alloc_type::device);
    _ptr_to_block.erase(exact_it);
    // Release outside the conditional below.
  } else {
    // Range-lookup for slab slots (or unknown pointer).
    if (_ptr_to_block.empty())
      return;
    auto it = _ptr_to_block.upper_bound(mem);
    if (it == _ptr_to_block.begin())
      return;
    --it;

    usm_block &block    = it->second;
    void      *slab_base = it->first;
    std::size_t offset  =
        static_cast<char *>(mem) - static_cast<char *>(slab_base);
    if (offset >= block.size)
      return; // pointer not in this block

    if (!block.is_slab)
      return; // should not happen (exact_it above would have matched)

    bool empty = _slab_alloc.free_slot(mem, slab_base);
    if (!empty)
      return; // slab still has live slots

    // Slab is now completely free — tear it down.
    buf  = block.buffer;
    size = block.size;
    from_region = true;
    mem  = slab_base; // return the slab base, not the slot address
    _slab_alloc.remove_slab(slab_base);
    _ptr_to_block.erase(it);
  }

  // Drop GPU mapping (mmap pages stay alive because deallocator was no-op).
  if (buf) {
    std::cerr << "Call Releasing Metal buffer at " << buf->contents() << "\n";
    buf->release();
  }

  // Return mmap pages to the free-list.
  if (from_region) {
     std::cerr << "Returning mmap pages to free-list at " << mem << " (size=" << size << ")\n";
    _region->free(mem, size);
  }
}

// ---------------------------------------------------------------------------
// Remaining interface
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

// get_usm_block: pure range-lookup in _ptr_to_block.
// Slab buffers are registered there with their full stride as block size,
// so any slot address within a slab falls inside the block naturally.
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
