// TEMPORARY: runs native_minmax.metal and checks the results.
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device: %s, Apple7=%d Metal3=%d\n", dev.name.UTF8String,
           [dev supportsFamily:MTLGPUFamilyApple7], [dev supportsFamily:MTLGPUFamilyMetal3]);
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:@"native_minmax.metal" encoding:NSUTF8StringEncoding error:&err];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { printf("library FAILED: %s\n", err.localizedDescription.UTF8String); return 1; }
    printf("library: ok\n");
    id<MTLComputePipelineState> pso =
      [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"native_minmax"] error:&err];
    if (!pso) { printf("pipeline FAILED: %s\n", err.localizedDescription.UTF8String); return 1; }
    printf("pipeline: ok\n");

    const uint32_t n = argc > 1 ? (uint32_t)atoi(argv[1]) : 256;
    id<MTLBuffer> obj = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    unsigned long long* v = (unsigned long long*)obj.contents;
    v[0] = ~0ull;  // min target
    v[1] = 0;      // max target

    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:obj offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(n < 256 ? n : 256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { printf("command buffer error: %s\n", cb.error.localizedDescription.UTF8String); return 1; }

    printf("n=%u min=%llu (expected 1) %s, max=%llu (expected %u) %s\n", n,
           v[0], v[0] == 1 ? "OK" : "WRONG", v[1], n, v[1] == n ? "OK" : "WRONG");
  }
  return 0;
}
