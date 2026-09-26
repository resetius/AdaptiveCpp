// TEMPORARY: replica of the program scope prologue that the emitter generates,
// including the lock table pointer at buffer(2).
#include <metal_stdlib>
using namespace metal;

uint __simd_size [[threads_per_simdgroup]];
uint __simd_group_id [[simdgroup_index_in_threadgroup]];
uint __simd_lane_id [[thread_index_in_simdgroup]];

uint3 group_id [[threadgroup_position_in_grid]];
uint3 num_groups [[threadgroups_per_grid]];
uint3 local_id [[thread_position_in_threadgroup]];
uint3 local_size [[threads_per_threadgroup]];

threadgroup void * constant dynamic_local_memory [[threadgroup(0)]];
constant uint& constant dynamic_local_memory_size [[buffer(0)]];
constant long& constant addr_diff [[buffer(1)]];
device uint* constant lock_base [[buffer(2)]];

kernel void prologue_buf2_unused(device uint* out [[buffer(3)]]) {
  out[local_id.x] = dynamic_local_memory_size + (uint)addr_diff + local_size.x;
}

kernel void prologue_buf2_used(device uint* out [[buffer(3)]]) {
  out[local_id.x] = lock_base[0] + dynamic_local_memory_size;
}
