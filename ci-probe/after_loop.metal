// TEMPORARY: is the problem the code that follows the voting loop?
#include <metal_stdlib>
using namespace metal;

static void vote_lock(device atomic_uint* locks, device atomic_uint* counter, thread ulong* old_out) {
  volatile bool done = false;
  ulong active = (ulong)(simd_vote::vote_t)simd_active_threads_mask();
  ulong done_active = 0;
  ulong old = 0;
  while (active != done_active) {
    if (!done) {
      uint expected = 0;
      if (atomic_compare_exchange_weak_explicit(&locks[0], &expected, 1u,
                                               memory_order_relaxed, memory_order_relaxed)) {
        old = atomic_load_explicit(&counter[0], memory_order_relaxed);
        atomic_store_explicit(&counter[0], (uint)old + 1, memory_order_relaxed);
        atomic_store_explicit(&locks[0], 0u, memory_order_relaxed);
        done = true;
      }
    }
    done_active = (ulong)(simd_vote::vote_t)simd_ballot(done);
  }
  *old_out = old;
}

// control: nothing follows the loop
kernel void loop_only(device atomic_uint* locks [[buffer(0)]],
                      device atomic_uint* counter [[buffer(1)]]) {
  ulong old = 0;
  vote_lock(locks, counter, &old);
}

// a store that does not depend on the loop follows it
kernel void loop_then_store(device atomic_uint* locks [[buffer(0)]],
                            device atomic_uint* counter [[buffer(1)]],
                            device uint* out [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
  ulong old = 0;
  vote_lock(locks, counter, &old);
  out[tid] = 42;
}

// the value taken inside the loop is stored after it
kernel void loop_then_value(device atomic_uint* locks [[buffer(0)]],
                            device atomic_uint* counter [[buffer(1)]],
                            device ulong* out [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
  ulong old = 0;
  vote_lock(locks, counter, &old);
  out[tid] = old;
}
