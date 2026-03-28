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



#include "hipSYCL/compiler/sscp/DynamicFunctionSupport.hpp"
#include "hipSYCL/compiler/utils/ProcessFunctionAnnotationsPass.hpp"

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Instructions.h>

namespace hipsycl::compiler {

namespace {

// Resolve a function pointer argument at a call site to a set of direct
// llvm::Function* values.  Handles two cases:
//
//   1. Direct: the operand is already an llvm::Function* — trivial.
//
//   2. Itanium C1/C2 wrapper pattern (macOS):
//      On Linux, C1 == alias(C2), so users of the annotated C2 see direct
//      function pointers from call sites in user code.
//      On macOS, C1 is a thin wrapper that stores its arguments to allocas
//      (O0 ABI) and calls C2 with the loaded values:
//
//        main → C1(this, &myfunction1)         direct Function* ✓
//                 store &myfunction1 → alloca
//                 %v = load alloca
//               C1 → C2(this, %v)              LoadInst, not Function* ✗
//
//      To resolve: LoadInst → AllocaInst → StoreInst → Argument of C1
//                          → all callers of C1 at that argument position.
//
// Returns true and populates Result on success; false if the indirection
// cannot be resolved (genuine user error).
bool resolveFunctionPtrOperand(llvm::Value* V,
                               llvm::SmallPtrSetImpl<llvm::Function*>& Result) {
  // Case 1: direct function reference.
  if (auto* F = llvm::dyn_cast<llvm::Function>(V)) {
    Result.insert(F);
    return true;
  }

  // Case 2: load from alloca → stored argument → callers of the wrapper.
  auto* LI = llvm::dyn_cast<llvm::LoadInst>(V);
  if (!LI || !llvm::isa<llvm::AllocaInst>(LI->getPointerOperand()))
    return false;

  // The alloca must have a single store whose value is a function argument.
  llvm::Argument* WrapperArg = nullptr;
  for (auto* U : LI->getPointerOperand()->users()) {
    auto* SI = llvm::dyn_cast<llvm::StoreInst>(U);
    if (!SI)
      continue;
    auto* Arg = llvm::dyn_cast<llvm::Argument>(SI->getValueOperand());
    if (!Arg || WrapperArg) // not an arg, or more than one store
      return false;
    WrapperArg = Arg;
  }
  if (!WrapperArg)
    return false;

  // Collect direct function pointers from every call site to the wrapper.
  llvm::Function* WrapperF = WrapperArg->getParent();
  unsigned ArgNo = WrapperArg->getArgNo();
  bool foundAny = false;
  for (auto* U : WrapperF->users()) {
    auto* CB = llvm::dyn_cast<llvm::CallBase>(U);
    if (!CB || ArgNo >= CB->arg_size())
      continue;
    auto* F = llvm::dyn_cast<llvm::Function>(CB->getOperand(ArgNo));
    if (!F)
      return false;
    Result.insert(F);
    foundAny = true;
  }
  return foundAny;
}

} // namespace

llvm::PreservedAnalyses DynamicFunctionIdentifactionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  // First identify dynamic function infrastructure, which may need
  // separate handling on host and device.
  llvm::SmallVector<llvm::Function *> DynamicFunctions;
  llvm::SmallVector<llvm::Function *> DynamicFunctionDefinitions;

  const std::string DynamicFunctionAnnotation = "dynamic_function";
  const std::string DynamicFunctionDefinitionAnnotationArg0 = "dynamic_function_def_arg0";
  const std::string DynamicFunctionDefinitionAnnotationArg1 = "dynamic_function_def_arg1";

  utils::ProcessFunctionAnnotationPass PFA({DynamicFunctionAnnotation,
                                            DynamicFunctionDefinitionAnnotationArg0,
                                            DynamicFunctionDefinitionAnnotationArg1});
  PFA.run(M, AM);
  const auto &FoundAnnotations = PFA.getFoundAnnotations();
  auto DFIt = FoundAnnotations.find(DynamicFunctionAnnotation);
  auto DFD0It = FoundAnnotations.find(DynamicFunctionDefinitionAnnotationArg0);
  auto DFD1It = FoundAnnotations.find(DynamicFunctionDefinitionAnnotationArg1);

  auto RetrieveFunctionNames = [&](auto It, std::vector<std::string>& Output, int ArgNo){
    for (auto *F : It->second) {
      if (F) {
        for (auto *U : F->users()) {
          if (llvm::CallBase *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
            if (CB->getNumOperands() >= 1) {
              llvm::SmallPtrSet<llvm::Function*, 4> Resolved;
              if (resolveFunctionPtrOperand(CB->getOperand(ArgNo), Resolved)) {
                for (auto* DF : Resolved)
                  Output.push_back(DF->getName().str());
              } else {
                M.getContext().emitError(
                    CB, "Detected a dynamic_function or dynamic_function_definition construction "
                        "where the argument is not "
                        "directly a function; dynamic_function function pointer arguments do "
                        "not support indirection.");
              }
            }
          }
        }
      }
    }
  };

  if (DFIt != FoundAnnotations.end()) {
    RetrieveFunctionNames(DFIt, this->DynamicFunctionNames, 1);
  }

  if (DFD0It != FoundAnnotations.end()) {
    RetrieveFunctionNames(DFD0It, this->DynamicFunctionDefinitionNames, 1);
  }

  if (DFD1It != FoundAnnotations.end()) {
    RetrieveFunctionNames(DFD1It, this->DynamicFunctionDefinitionNames, 2);
  }

  return llvm::PreservedAnalyses::none();
}

llvm::PreservedAnalyses HostSideDynamicFunctionHandlerPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  // Provide dummy definitions to avoid linker issues
  for (const auto &FName : DynamicFunctionNames) {
    if (auto *F = M.getFunction(FName)) {
      if (F->isDeclaration()) {
        F->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
        auto BB = llvm::BasicBlock::Create(M.getContext(), "entry", F);
        // It seems that LLVM handles functions that only have unreachable inst differently
        // if there address is taken - we can no longer get unique function pointer
        // addresses for those, which breaks reflection. So try to generate a trivial function
        // instead.
        if(F->getReturnType()->isVoidTy())
          llvm::ReturnInst::Create(M.getContext(), BB);
        else
          llvm::ReturnInst::Create(M.getContext(), llvm::UndefValue::get(F->getReturnType()), BB);
      }
    }
  }

  return llvm::PreservedAnalyses::none();
}


}
