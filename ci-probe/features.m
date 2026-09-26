// TEMPORARY: compiles every .msl given on the command line and creates one
// pipeline per kernel in it, with the compile options of the runtime and with
// the default ones, so the log says which kernel the GPU backend rejects.
#import <Metal/Metal.h>
#include <stdio.h>

static void run_file(id<MTLDevice> dev, NSString* path, NSString* label,
                     MTLCompileOptions* opts) {
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
  if (!src) { printf("cannot read %s\n", path.UTF8String); return; }

  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
  if (!lib) { printf("  [%s] library FAILED: %s\n", label.UTF8String, err.description.UTF8String); return; }

  for (NSString* name in lib.functionNames) {
    err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn || fn.functionType != MTLFunctionTypeKernel) continue;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    printf("  [%s] %-34s %s\n", label.UTF8String,
           [name substringToIndex:MIN((NSUInteger)34, name.length)].UTF8String,
           pso ? "pipeline ok" : [NSString stringWithFormat:@"PIPELINE FAILED: %@", err.localizedDescription].UTF8String);
  }
}

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

    // what the runtime uses
    MTLCompileOptions* runtime_opts = [MTLCompileOptions new];
    runtime_opts.languageVersion = MTLLanguageVersion4_0;
    runtime_opts.optimizationLevel = MTLLibraryOptimizationLevelSize;

    for (int a = 1; a < argc; ++a) {
      NSString* path = [NSString stringWithUTF8String:argv[a]];
      printf("\n=== %s\n", path.lastPathComponent.UTF8String);
      run_file(dev, path, @"runtime", runtime_opts);
      run_file(dev, path, @"default", [MTLCompileOptions new]);
    }
  }
  return 0;
}
