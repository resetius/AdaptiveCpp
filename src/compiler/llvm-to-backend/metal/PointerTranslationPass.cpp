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

#include "hipSYCL/compiler/llvm-to-backend/metal/PointerTranslationPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/metal/PointerTranslationAnnotationPass.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>

using namespace llvm;

namespace hipsycl {
namespace compiler {

namespace {

static constexpr const char *SymbolIdFuncName = "__acpp_sscp_metal_symbol_id";
static constexpr const char *AddrDiffSymbolName =
    "__acpp_sscp_metal_gpu_to_host_addr_diff";
static constexpr const char *AddrDiffStrGlobalName =
    "__acpp_metal_addr_diff_sym_name";

// Returns (or creates) the __acpp_sscp_metal_symbol_id declaration.
// Signature: i64(ptr)
Function *getOrCreateSymbolIdDecl(Module &M) {
  if (Function *F = M.getFunction(SymbolIdFuncName))
    return F;

  LLVMContext &Ctx = M.getContext();
  FunctionType *FT =
      FunctionType::get(Type::getInt64Ty(Ctx), {PointerType::getUnqual(Ctx)},
                        /*isVarArg=*/false);
  Function *F = Function::Create(FT, GlobalValue::ExternalLinkage,
                                 SymbolIdFuncName, M);
  return F;
}

// Returns (or creates) the global string constant holding the MSL symbol name.
GlobalVariable *getOrCreateAddrDiffNameGlobal(Module &M) {
  if (GlobalVariable *GV = M.getNamedGlobal(AddrDiffStrGlobalName))
    return GV;

  LLVMContext &Ctx = M.getContext();
  Constant *Str = ConstantDataArray::getString(Ctx, AddrDiffSymbolName,
                                               /*AddNull=*/true);
  auto *GV = new GlobalVariable(M, Str->getType(), /*isConstant=*/true,
                                GlobalValue::PrivateLinkage, Str,
                                AddrDiffStrGlobalName);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return GV;
}

// Inserts a delta call at the start of the entry block and returns it.
// If one was already inserted (same function, same pass run), returns it.
CallInst *getOrInsertDeltaInEntryBlock(Function &F, Function *SymbolIdFn,
                                       GlobalVariable *AddrDiffNameGV) {
  BasicBlock &Entry = F.getEntryBlock();
  // Check if we already inserted for this function.
  // Must match both the callee AND the specific argument (the addr-diff name),
  // because parallel_for kernels already contain symbol_id calls for
  // group_id / local_id built-ins that use the same function.
  for (Instruction &I : Entry) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (CI->getCalledFunction() == SymbolIdFn &&
          CI->getArgOperand(0) == AddrDiffNameGV)
        return CI;
    }
  }

  IRBuilder<> B(&*Entry.getFirstInsertionPt());
  CallInst *Delta = B.CreateCall(SymbolIdFn, {AddrDiffNameGV}, "gpu_host_delta");
  return Delta;
}

} // anonymous namespace

PointerTranslationPass::PointerTranslationPass(unsigned GlobalAS)
    : GlobalAS(GlobalAS) {}

PreservedAnalyses PointerTranslationPass::run(Module &M,
                                              ModuleAnalysisManager &MAM) {
  Function *SymbolIdFn = getOrCreateSymbolIdDecl(M);
  GlobalVariable *AddrDiffNameGV = getOrCreateAddrDiffNameGlobal(M);

  unsigned LoadsTranslated = 0;
  unsigned StoresTranslated = 0;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // Collect annotated instructions first to avoid iterator invalidation.
    SmallVector<LoadInst *, 16> AnnotatedLoads;
    SmallVector<StoreInst *, 16> AnnotatedStores;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->getMetadata(MetalPtrLoadNeedsTranslationMD))
            AnnotatedLoads.push_back(LI);
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (SI->getMetadata(MetalPtrStoreNeedsTranslationMD))
            AnnotatedStores.push_back(SI);
        }
      }
    }

    if (AnnotatedLoads.empty() && AnnotatedStores.empty())
      continue;

    // Hoist the delta call to the entry block once per function.
    CallInst *Delta =
        getOrInsertDeltaInEntryBlock(F, SymbolIdFn, AddrDiffNameGV);

    // Rewrite loads: add delta to the loaded pointer, preserving null.
    // If the loaded host pointer is null it must stay null after translation:
    // a naive GEP would turn 0 into (0 + delta) which is non-null and wrong.
    for (LoadInst *LI : AnnotatedLoads) {
      IRBuilder<> B(LI->getNextNode());
      auto *PtrTy = cast<PointerType>(LI->getType());
      Value *NullPtr = ConstantPointerNull::get(PtrTy);
      Value *IsNull  = B.CreateICmpEQ(LI, NullPtr, "ptr_loaded_is_null");
      Value *Gep     = B.CreateGEP(Type::getInt8Ty(M.getContext()),
                                   LI, Delta, "ptr_gpu_raw");
      // Select: keep null as-is, adjust non-null host→GPU.
      Value *Adjusted = B.CreateSelect(IsNull, LI, Gep, "ptr_gpu_adjusted");
      LI->replaceUsesWithIf(Adjusted, [&](Use &U) {
        return U.getUser() != IsNull && U.getUser() != Gep &&
               U.getUser() != Adjusted;
      });
      LI->setMetadata(MetalPtrLoadNeedsTranslationMD, nullptr);
      ++LoadsTranslated;
      HIPSYCL_DEBUG_INFO << "Metal PointerTranslationPass: translated load in "
                         << F.getName() << "\n";
    }

    // Rewrite stores: subtract delta from the pointer value before storing,
    // preserving null.
    for (StoreInst *SI : AnnotatedStores) {
      IRBuilder<> B(SI);
      Value *Val    = SI->getValueOperand();
      auto *PtrTy   = cast<PointerType>(Val->getType());
      Value *NullPtr = ConstantPointerNull::get(PtrTy);
      Value *IsNull  = B.CreateICmpEQ(Val, NullPtr, "ptr_stored_is_null");
      Value *NegDelta = B.CreateNeg(Delta, "neg_gpu_host_delta");
      Value *Gep     = B.CreateGEP(Type::getInt8Ty(M.getContext()),
                                   Val, NegDelta, "ptr_host_raw");
      // Select: keep null as-is, adjust non-null GPU→host.
      Value *Adjusted = B.CreateSelect(IsNull, NullPtr, Gep, "ptr_host_adjusted");
      SI->setOperand(0, Adjusted);
      SI->setMetadata(MetalPtrStoreNeedsTranslationMD, nullptr);
      ++StoresTranslated;
      HIPSYCL_DEBUG_INFO << "Metal PointerTranslationPass: translated store in "
                         << F.getName() << "\n";
    }
  }

  HIPSYCL_DEBUG_INFO << "Metal PointerTranslationPass: translated "
                     << LoadsTranslated << " load(s) and "
                     << StoresTranslated << " store(s).\n";

  return (LoadsTranslated + StoresTranslated) > 0 ? PreservedAnalyses::none()
                                                  : PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
