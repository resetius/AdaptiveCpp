// TEMPORARY: runs vote_diag.metal. Every loop in it is capped, so a condition
// that never converges is reported as a cap hit instead of hanging the GPU.
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:@"vote_diag.metal" encoding:NSUTF8StringEncoding error:&err];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_0;
    opts.optimizationLevel = MTLLibraryOptimizationLevelSize;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { printf("library FAILED: %s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    printf("device: %s\n", dev.name.UTF8String);

    const uint32_t n = argc > 1 ? (uint32_t)atoi(argv[1]) : 32;
    id<MTLBuffer> locks = [dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> counter = [dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> iters = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];

    NSArray<NSString*>* names = @[@"vote_report", @"loop_equal", @"loop_masked", @"loop_simd_all"];
    for (NSString* name in names) {
      id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
      if (!pso) { printf("%-14s pipeline FAILED: %s\n", name.UTF8String, err.localizedDescription.UTF8String); continue; }

      memset(locks.contents, 0, 8);
      memset(counter.contents, 0, 8);
      memset(iters.contents, 0, n * 4);

      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:locks offset:0 atIndex:0];
      [enc setBuffer:counter offset:0 atIndex:1];
      [enc setBuffer:iters offset:0 atIndex:2];
      [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(n, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) { printf("%-14s command buffer error: %s\n", name.UTF8String, cb.error.localizedDescription.UTF8String); continue; }

      if ([name isEqualToString:@"vote_report"]) {
        uint32_t* v = (uint32_t*)iters.contents;
        printf("simd_size=%u active=%08x%08x ballot(true)=%08x%08x ballot(false)=%08x%08x ballot(even)=%08x%08x simd_all(true)=%u\n",
               v[0], v[2], v[1], v[4], v[3], v[6], v[5], v[8], v[7], v[9]);
      } else {
        uint32_t* it = (uint32_t*)iters.contents;
        uint32_t max_it = 0, cap_hits = 0;
        for (uint32_t i = 0; i < n; ++i) { if (it[i] > max_it) max_it = it[i]; if (it[i] >= 4096) ++cap_hits; }
        printf("%-14s counter=%u (expected %u) max iterations=%u cap hits=%u\n", name.UTF8String,
               *(uint32_t*)counter.contents, n, max_it, cap_hits);
      }
    }
  }
  return 0;
}
