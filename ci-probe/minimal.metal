// TEMPORARY: how small can the failing case get?
#include <metal_stdlib>
using namespace metal;

// no loop at all, just a volatile 64-bit local
kernel void volatile_ulong_no_loop(device ulong* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  volatile ulong v = tid;
  v = v + 1;
  out[tid] = v;
}

kernel void volatile_uint_no_loop(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  volatile uint v = tid;
  v = v + 1;
  out[tid] = v;
}

kernel void plain_ulong_no_loop(device ulong* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  ulong v = tid;
  v = v + 1;
  out[tid] = v;
}

// the voting loop with the old value in two volatile 32-bit halves
kernel void shape_two_halves(device atomic_uint* locks [[buffer(0)]], device atomic_uint* counter [[buffer(1)]],
                             device ulong* out [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  volatile bool done = false;
  volatile uint old_low = 0;
  volatile uint old_high = 0;
  ulong active = (ulong)(simd_vote::vote_t)simd_active_threads_mask();
  ulong done_active = 0;
  while (active != done_active) {
    if (!done) {
      uint expected = 0;
      if (atomic_compare_exchange_weak_explicit(&locks[0], &expected, 1u,
                                               memory_order_relaxed, memory_order_relaxed)) {
        old_low = atomic_load_explicit(&counter[0], memory_order_relaxed);
        old_high = atomic_load_explicit(&counter[1], memory_order_relaxed);
        atomic_store_explicit(&counter[0], old_low + 1, memory_order_relaxed);
        atomic_store_explicit(&locks[0], 0u, memory_order_relaxed);
        done = true;
      }
    }
    done_active = (ulong)(simd_vote::vote_t)simd_ballot(done);
  }
  out[tid] = ((ulong)old_high << 32) | (ulong)old_low;
}

// the voting loop with a plain (non volatile) 64-bit old value
kernel void shape_plain_ulong(device atomic_uint* locks [[buffer(0)]], device atomic_uint* counter [[buffer(1)]],
                              device ulong* out [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  volatile bool done = false;
  ulong old = 0;
  ulong active = (ulong)(simd_vote::vote_t)simd_active_threads_mask();
  ulong done_active = 0;
  while (active != done_active) {
    if (!done) {
      uint expected = 0;
      if (atomic_compare_exchange_weak_explicit(&locks[0], &expected, 1u,
                                               memory_order_relaxed, memory_order_relaxed)) {
        uint low = atomic_load_explicit(&counter[0], memory_order_relaxed);
        uint high = atomic_load_explicit(&counter[1], memory_order_relaxed);
        old = ((ulong)high << 32) | (ulong)low;
        atomic_store_explicit(&counter[0], low + 1, memory_order_relaxed);
        atomic_store_explicit(&locks[0], 0u, memory_order_relaxed);
        done = true;
      }
    }
    done_active = (ulong)(simd_vote::vote_t)simd_ballot(done);
  }
  out[tid] = old;
}
