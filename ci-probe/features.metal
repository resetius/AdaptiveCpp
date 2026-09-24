// TEMPORARY: narrows down which MSL construct the paravirtual GPU backend
// compiler rejects. Each kernel adds one construct on top of the previous one.
#include <metal_stdlib>
using namespace metal;

device uint* constant prog_scope_ptr [[buffer(2)]];

kernel void plain(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  out[tid] = tid;
}

kernel void vote_mask(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  out[tid] = (uint)(ulong)(simd_vote::vote_t)simd_active_threads_mask();
}

kernel void vote_ballot(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  out[tid] = (uint)(ulong)(simd_vote::vote_t)simd_ballot(tid % 2 == 0);
}

kernel void program_scope_pointer(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  out[tid] = prog_scope_ptr[tid & 1023];
}

kernel void lock_loop(device atomic_uint* locks [[buffer(0)]],
                      device atomic_uint* counter [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
  volatile bool done = false;
  ulong active = (ulong)(simd_vote::vote_t)simd_active_threads_mask();
  ulong done_active = 0;
  while (active != done_active) {
    if (!done) {
      uint expected = 0;
      if (atomic_compare_exchange_weak_explicit(&locks[0], &expected, 1u,
                                               memory_order_relaxed, memory_order_relaxed)) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        uint low = atomic_load_explicit(&counter[0], memory_order_relaxed);
        atomic_store_explicit(&counter[0], low + 1, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_store_explicit(&locks[0], 0u, memory_order_relaxed);
        done = true;
      }
    }
    done_active = (ulong)(simd_vote::vote_t)simd_ballot(done);
  }
}

kernel void lock_loop_prog_scope(device atomic_uint* counter [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]]) {
  device atomic_uint* locks = (device atomic_uint*)prog_scope_ptr;
  volatile bool done = false;
  ulong active = (ulong)(simd_vote::vote_t)simd_active_threads_mask();
  ulong done_active = 0;
  while (active != done_active) {
    if (!done) {
      uint expected = 0;
      if (atomic_compare_exchange_weak_explicit(&locks[0], &expected, 1u,
                                               memory_order_relaxed, memory_order_relaxed)) {
        uint low = atomic_load_explicit(&counter[0], memory_order_relaxed);
        atomic_store_explicit(&counter[0], low + 1, memory_order_relaxed);
        atomic_store_explicit(&locks[0], 0u, memory_order_relaxed);
        done = true;
      }
    }
    done_active = (ulong)(simd_vote::vote_t)simd_ballot(done);
  }
}
