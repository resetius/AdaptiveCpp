// TEMPORARY: are the native 64-bit atomic min and max usable on this GPU?
#include <metal_stdlib>
using namespace metal;

kernel void native_minmax(device atomic_ulong* obj [[buffer(0)]],
                          uint tid [[thread_position_in_grid]]) {
  atomic_min_explicit(&obj[0], (ulong)tid + 1, memory_order_relaxed);
  atomic_max_explicit(&obj[1], (ulong)tid + 1, memory_order_relaxed);
}
