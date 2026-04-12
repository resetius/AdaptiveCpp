/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

#include "hipSYCL/compiler/llvm-to-backend/metal/PointerTranslationAnnotationPass.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <llvm/ADT/SmallPtrSet.h>

using namespace llvm;

namespace hipsycl {
namespace compiler {

// ---------------------------------------------------------------------------
// Helper: walk GEPs and bitcasts back to the underlying alloca, if any.
// Returns nullptr if the base is not an alloca.
// ---------------------------------------------------------------------------
static AllocaInst *getBaseAlloca(Value *V) {
  while (V) {
    if (auto *AI = dyn_cast<AllocaInst>(V))
      return AI;
    if (auto *GEP = dyn_cast<GEPOperator>(V)) {
      V = GEP->getPointerOperand();
      continue;
    }
    if (auto *BC = dyn_cast<BitCastOperator>(V)) {
      V = BC->getOperand(0);
      continue;
    }
    break;
  }
  return nullptr;
}

PointerTranslationAnnotationPass::PointerTranslationAnnotationPass(unsigned GlobalAS)
    : GlobalAS(GlobalAS) {}

llvm::PreservedAnalyses
PointerTranslationAnnotationPass::run(Module &M, ModuleAnalysisManager &MAM) {
  LLVMContext &Ctx = M.getContext();
  // An empty metadata node — presence of the key is the signal, not content.
  MDNode *EmptyMD = MDNode::get(Ctx, {});

  unsigned LoadsAnnotated = 0;
  unsigned StoresAnnotated = 0;

  for (Function &F : M) {
    // -----------------------------------------------------------------------
    // Pre-scan: find AS5 allocas that were populated from global (AS1) memory
    // via llvm.memcpy.  These are the stack copies of pointer arrays produced
    // by `__builtin_memcpy(local_ptrs, device_ptrs, ...)` in kernel code.
    // Loads from such allocas deliver host addresses just like direct AS1 loads
    // and therefore require the same pointer translation.
    // -----------------------------------------------------------------------
    SmallPtrSet<AllocaInst *, 8> TaintedAllocas;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *MI = dyn_cast<MemCpyInst>(&I);
        if (!MI)
          continue;

        // Source must be in global address space (AS1).
        unsigned SrcAS = MI->getRawSource()->getType()->getPointerAddressSpace();
        if (SrcAS != GlobalAS)
          continue;

        // Destination must be in thread-private address space (AS5 = stack).
        unsigned DstAS = MI->getRawDest()->getType()->getPointerAddressSpace();
        if (DstAS != 5)
          continue;

        // Walk back to the underlying alloca and mark it as tainted.
        if (AllocaInst *AI = getBaseAlloca(MI->getRawDest()))
          TaintedAllocas.insert(AI);
      }
    }

    // -----------------------------------------------------------------------
    // Main annotation pass.
    // -----------------------------------------------------------------------
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {

        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          // Rule 1: load of a pointer value from global (device) memory.
          //
          // The loaded value is a host virtual address written by the CPU.
          // The GPU must add the per-allocation offset before dereferencing it.
          //
          // Conditions:
          //   (a) the result type is a pointer;
          //   (b) the source is in global address space (AS1),
          //       OR the source is in thread-private space (AS5) and the
          //       underlying alloca was populated by an llvm.memcpy from AS1.
          if (!LI->getType()->isPointerTy())
            continue;

          unsigned SrcAS =
              LI->getPointerOperand()->getType()->getPointerAddressSpace();
          bool FromDeviceMemory = (SrcAS == GlobalAS);
          bool FromMemcpyAlloca =
              (SrcAS == 5 &&
               TaintedAllocas.count(getBaseAlloca(LI->getPointerOperand())));

          if (!FromDeviceMemory && !FromMemcpyAlloca)
            continue;

          LI->setMetadata(MetalPtrLoadNeedsTranslationMD, EmptyMD);
          ++LoadsAnnotated;
          HIPSYCL_DEBUG_INFO << "Metal PtrTranslation: annotated load in "
                             << F.getName() << ": " << *LI << "\n";
          continue;
        }

        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          // Rule 2: store of a pointer value into global (device) memory.
          //
          // The value being stored is a GPU virtual address.  Before writing it
          // the GPU must subtract the per-allocation offset so that the CPU can
          // later read and follow it as a valid host virtual address.
          //
          // Conditions:
          //   (a) the value operand is a pointer — we are storing an address;
          //   (b) the pointer operand (destination) is in global address space.
          if (!SI->getValueOperand()->getType()->isPointerTy())
            continue;
          if (SI->getPointerOperand()->getType()->getPointerAddressSpace() !=
              GlobalAS)
            continue;

          SI->setMetadata(MetalPtrStoreNeedsTranslationMD, EmptyMD);
          ++StoresAnnotated;
          HIPSYCL_DEBUG_INFO << "Metal PtrTranslation: annotated store in "
                             << F.getName() << ": " << *SI << "\n";
          continue;
        }
      }
    }
  }

  HIPSYCL_DEBUG_INFO << "Metal PointerTranslationAnnotationPass: annotated "
                     << LoadsAnnotated << " load(s) and "
                     << StoresAnnotated << " store(s) for address translation.\n";

  // This pass only attaches metadata; the IR structure is unchanged.
  return (LoadsAnnotated + StoresAnnotated) > 0 ? PreservedAnalyses::none()
                                                : PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
