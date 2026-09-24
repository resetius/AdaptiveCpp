// TEMPORARY: compiles every .msl given on the command line and creates one
// pipeline per kernel in it, so the log says which kernel the GPU backend
// compiler rejects.
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

    for (int a = 1; a < argc; ++a) {
      NSString* path = [NSString stringWithUTF8String:argv[a]];
      printf("\n=== %s\n", path.lastPathComponent.UTF8String);
      NSError* err = nil;
      NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
      if (!src) { printf("cannot read file\n"); continue; }

      id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
      if (!lib) { printf("library: FAILED: %s\n", err.description.UTF8String); continue; }
      printf("library: ok, %lu function(s)\n", (unsigned long)lib.functionNames.count);

      for (NSString* name in lib.functionNames) {
        err = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn || fn.functionType != MTLFunctionTypeKernel) continue;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        printf("  %-40s %s\n", name.UTF8String,
               pso ? "pipeline ok" : [NSString stringWithFormat:@"PIPELINE FAILED: %@", err].UTF8String);
      }
    }
  }
  return 0;
}
