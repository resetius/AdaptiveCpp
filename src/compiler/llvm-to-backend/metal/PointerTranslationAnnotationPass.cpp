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
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

using namespace llvm;

namespace hipsycl {
namespace compiler {

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
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {

        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          // Rule 1: load of a pointer value from global (device) memory.
          //
          // The loaded value is a host virtual address written by the CPU.
          // The GPU must add the per-allocation offset before dereferencing it.
          //
          // Conditions:
          //   (a) the result type is a pointer — we are loading an address,
          //       not plain scalar/vector/struct data;
          //   (b) the pointer operand is in global address space — the
          //       container lives in device memory (threadgroup, thread, and
          //       constant pointers are excluded).
          if (!LI->getType()->isPointerTy())
            continue;
          if (LI->getPointerOperand()->getType()->getPointerAddressSpace() != GlobalAS)
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
          if (SI->getPointerOperand()->getType()->getPointerAddressSpace() != GlobalAS)
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
