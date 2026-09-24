// TEMPORARY: compiles features.metal and creates a pipeline per kernel, so the
// log says which construct the GPU backend compiler rejects.
#import <Metal/Metal.h>
#include <stdio.h>

int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("no Metal device\n"); return 1; }
    printf("device: %s\n", dev.name.UTF8String);
    printf("families: ");
    struct { const char* name; MTLGPUFamily family; } families[] = {
      {"Apple7", MTLGPUFamilyApple7}, {"Apple8", MTLGPUFamilyApple8},
      {"Apple9", MTLGPUFamilyApple9}, {"Mac2", MTLGPUFamilyMac2},
      {"Metal3", MTLGPUFamilyMetal3},
    };
    for (unsigned i = 0; i < sizeof(families)/sizeof(families[0]); ++i)
      printf("%s=%d ", families[i].name, [dev supportsFamily:families[i].family]);
    printf("\n");

    NSError* err = nil;
    NSString* path = argc > 1 ? [NSString stringWithUTF8String:argv[1]] : @"features.metal";
    NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
    if (!src) { printf("cannot read %s\n", path.UTF8String); return 1; }

    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { printf("library: FAILED: %s\n", err.description.UTF8String); return 1; }
    printf("library: ok\n");

    NSArray<NSString*>* names = @[@"plain", @"vote_mask", @"vote_ballot",
                                 @"program_scope_pointer", @"lock_loop",
                                 @"lock_loop_prog_scope"];
    for (NSString* name in names) {
      err = nil;
      id<MTLFunction> fn = [lib newFunctionWithName:name];
      if (!fn) { printf("%-22s no such function\n", name.UTF8String); continue; }
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
      printf("%-22s %s\n", name.UTF8String,
             pso ? "pipeline ok" : [NSString stringWithFormat:@"PIPELINE FAILED: %@", err].UTF8String);
    }
  }
  return 0;
}
