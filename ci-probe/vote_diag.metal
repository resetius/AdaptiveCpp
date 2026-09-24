// TEMPORARY: what do the SIMD vote functions report on this GPU, and which
// loop condition converges? Every loop is capped, so a broken condition shows
// up as a cap hit instead of a hang.
#include <metal_stdlib>
using namespace metal;

constant uint cap = 4096;

kernel void vote_report(device uint* out [[buffer(2)]],
                        uint simd_size [[threads_per_simdgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint tid [[thread_position_in_grid]]) {
  ulong active = (ulong)(simd_vote::vote_t)simd_active_threads_mask();
  ulong all_true = (ulong)(simd_vote::vote_t)simd_ballot(true);
  ulong all_false = (ulong)(simd_vote::vote_t)simd_ballot(false);
  ulong even = (ulong)(simd_vote::vote_t)simd_ballot((lane & 1) == 0);
  if (tid == 0) {
    out[0] = simd_size;
    out[1] = (uint)active;        out[2] = (uint)(active >> 32);
    out[3] = (uint)all_true;      out[4] = (uint)(all_true >> 32);
    out[5] = (uint)all_false;     out[6] = (uint)(all_false >> 32);
    out[7] = (uint)even;          out[8] = (uint)(even >> 32);
    out[9] = simd_all(true) ? 1 : 0;
  }
}

#define LOCK_BODY(EXIT_CONDITION)                                              \
  uint used = 0;                                                               \
  volatile bool done = false;                                                  \
  ulong active = (ulong)(simd_vote::vote_t)simd_active_threads_mask();          \
  ulong done_active = 0;                                                       \
  (void)done_active;                                                           \
  while (!(EXIT_CONDITION) && used < cap) {                                    \
    ++used;                                                                    \
    if (!done) {                                                               \
      uint expected = 0;                                                       \
      if (atomic_compare_exchange_weak_explicit(&locks[0], &expected, 1u,       \
                                               memory_order_relaxed,            \
                                               memory_order_relaxed)) {         \
        uint low = atomic_load_explicit(&counter[0], memory_order_relaxed);      \
        atomic_store_explicit(&counter[0], low + 1, memory_order_relaxed);       \
        atomic_store_explicit(&locks[0], 0u, memory_order_relaxed);              \
        done = true;                                                             \
      }                                                                          \
    }                                                                            \
    done_active = (ulong)(simd_vote::vote_t)simd_ballot(done);                    \
  }                                                                              \
  iters[tid] = used;

// what we do now: compare the two votes
kernel void loop_equal(device atomic_uint* locks [[buffer(0)]], device atomic_uint* counter [[buffer(1)]],
                       device uint* iters [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  LOCK_BODY(active == done_active)
}

// ignore the bits of lanes that are not active
kernel void loop_masked(device atomic_uint* locks [[buffer(0)]], device atomic_uint* counter [[buffer(1)]],
                        device uint* iters [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  LOCK_BODY((done_active & active) == active)
}

// ask the hardware directly
kernel void loop_simd_all(device atomic_uint* locks [[buffer(0)]], device atomic_uint* counter [[buffer(1)]],
                          device uint* iters [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  LOCK_BODY(simd_all(done))
}
