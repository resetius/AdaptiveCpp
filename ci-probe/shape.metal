// TEMPORARY: hand written copy of the shape that the emitter produces for a
// 64-bit atomic load, with variations.
#include <metal_stdlib>
using namespace metal;

#define BODY(OLDTYPE, GUARD, READ_AFTER)                                        \
  volatile bool done = false;                                                   \
  volatile OLDTYPE old = 0;                                                     \
  ulong active = (ulong)(simd_vote::vote_t)simd_active_threads_mask();           \
  ulong done_active = 0;                                                         \
  if (GUARD) {                                                                   \
    while (true) {                                                               \
      if (!done) {                                                               \
        uint expected = 0;                                                       \
        if (atomic_compare_exchange_weak_explicit(&locks[0], &expected, 1u,      \
                                                 memory_order_relaxed,           \
                                                 memory_order_relaxed)) {        \
          uint low = atomic_load_explicit(&counter[0], memory_order_relaxed);    \
          uint high = atomic_load_explicit(&counter[1], memory_order_relaxed);   \
          old = (OLDTYPE)(((ulong)high << 32) | (ulong)low);                      \
          atomic_store_explicit(&counter[0], (uint)old + 1, memory_order_relaxed);\
          atomic_store_explicit(&locks[0], 0u, memory_order_relaxed);             \
          done = true;                                                            \
        }                                                                         \
      }                                                                           \
      done_active = (ulong)(simd_vote::vote_t)simd_ballot(done);                  \
      if (active == done_active) { break; } else { continue; }                    \
    }                                                                             \
  }                                                                               \
  if (READ_AFTER) out[tid] = (ulong)old;

kernel void shape_full(device atomic_uint* locks [[buffer(0)]], device atomic_uint* counter [[buffer(1)]],
                       device ulong* out [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  BODY(ulong, active != 0, true)
}

kernel void shape_noguard(device atomic_uint* locks [[buffer(0)]], device atomic_uint* counter [[buffer(1)]],
                          device ulong* out [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  BODY(ulong, true, true)
}

kernel void shape_uint_old(device atomic_uint* locks [[buffer(0)]], device atomic_uint* counter [[buffer(1)]],
                           device ulong* out [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  BODY(uint, active != 0, true)
}

kernel void shape_no_read_after(device atomic_uint* locks [[buffer(0)]], device atomic_uint* counter [[buffer(1)]],
                                device ulong* out [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  BODY(ulong, active != 0, false)
}
