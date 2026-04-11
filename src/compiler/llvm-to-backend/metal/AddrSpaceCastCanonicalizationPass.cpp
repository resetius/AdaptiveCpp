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

#include "hipSYCL/compiler/llvm-to-backend/metal/AddrSpaceCastCanonicalizationPass.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

using namespace llvm;

namespace hipsycl {
namespace compiler {

namespace {

/// Clone a global variable into a new address space.
///
/// Given an existing global variable and a target address space, create a new
/// global variable with the same initializer and linkage in the target address
/// space.  The old global is not modified.  Returns the newly created global.
static GlobalVariable *cloneGlobalWithAddressSpace(Module &M, GlobalVariable *GV,
                                                   unsigned TargetAS) {
  assert(GV && "Expected valid global variable");
  // If the global is already in the desired address space, simply return it.
  if (GV->getAddressSpace() == TargetAS)
    return GV;

  // Construct a unique name for the new global.  Rename the old global to
  // avoid name conflicts and aid debugging.
  std::string OriginalName{GV->getName().str()};
  GV->setName(OriginalName + ".original");

  Constant *Initializer = nullptr;
  if (GV->hasInitializer())
    Initializer = GV->getInitializer();

  // Create a new global variable in the target address space.  Copy all
  // relevant properties from the original.
  auto *NewGV = new GlobalVariable(
      M, GV->getValueType(), GV->isConstant(), GV->getLinkage(), Initializer,
      OriginalName, /*InsertBefore=*/nullptr, GV->getThreadLocalMode(), TargetAS);
  NewGV->setAlignment(GV->getAlign());
  NewGV->copyAttributesFrom(GV);
  // Ensure the new global has the same DLL storage class and visibility.
  NewGV->setUnnamedAddr(GV->getUnnamedAddr());
  return NewGV;
}

/// Replace constant expression addrspacecasts on global variables.
///
/// This function scans all uses of global variables looking for constant
/// expression addrspacecast instructions that convert the global into a
/// different address space.  For each such cast, it clones the global into
/// the target address space and rewrites all uses of the constant expression
/// to reference the new global directly.  Duplicates are avoided by
/// remembering cloned globals.
static bool canonicalizeConstantAddrSpaceCasts(Module &M) {
  bool Changed = false;
  // Map from (GlobalVariable, address space) pairs to the cloned global.
  DenseMap<std::pair<GlobalVariable *, unsigned>, GlobalVariable *> Clones;
  SmallVector<ConstantExpr *, 16> CastsToRemove;

  for (GlobalVariable &GV : M.globals()) {
    // Iterate over a copy of the use list as it may be modified.
    SmallVector<User *, 8> Users(GV.users().begin(), GV.users().end());
    for (User *U : Users) {
      // Only interested in constant expression addrspacecasts.
      auto *CE = dyn_cast<ConstantExpr>(U);
      if (!CE || CE->getOpcode() != Instruction::AddrSpaceCast)
        continue;

      // Determine the destination address space from the cast type.
      auto *DstPtrTy = cast<PointerType>(CE->getType());
      unsigned DestAS = DstPtrTy->getAddressSpace();
      unsigned SrcAS = GV.getAddressSpace();

      if (DestAS == SrcAS)
        continue;

      // Create or retrieve a cloned global in the target address space.
      auto Key = std::make_pair(&GV, DestAS);
      GlobalVariable *NewGV = nullptr;
      auto It = Clones.find(Key);
      if (It != Clones.end()) {
        NewGV = It->second;
      } else {
        NewGV = cloneGlobalWithAddressSpace(M, &GV, DestAS);
        Clones[Key] = NewGV;
        Changed = true;
      }

      // Replace all uses of the constant expression with the new global.
      Constant *Replacement = ConstantExpr::getPointerCast(NewGV, CE->getType());
      CE->replaceAllUsesWith(Replacement);
      CastsToRemove.push_back(CE);
      Changed = true;
    }
  }

  // Destroy the constant expressions after we are done mutating the use lists.
  for (ConstantExpr *CE : CastsToRemove)
    CE->destroyConstant();

  return Changed;
}

/// Recursively rewrite uses of an addrspacecast instruction.
///
/// Given an AddrSpaceCastInst, walk its use-def chain and create new
/// instructions that operate directly on the source value of the cast.  For
/// supported instructions the pass generates equivalent operations that do
/// not require an intermediate cast.  Chains of addrspacecast and bitcast
/// instructions are collapsed.  Unsupported instructions will cause the cast
/// to remain in place.
static bool rewriteAddrSpaceCastUses(AddrSpaceCastInst *Cast) {
  // Only remove downcasts (source AS is more specific than dest AS).
  // Upcasts (generic AS=0 → specific AS=N) were inserted by the address space
  // inference pass to mark pointers as living in a particular space.  Removing
  // them would propagate the generic AS=0 source back through GEPs and PHIs,
  // silently downgrading address space information.
  unsigned SrcAS = Cast->getOperand(0)->getType()->getPointerAddressSpace();
  unsigned DstAS = Cast->getType()->getPointerAddressSpace();
  if (SrcAS == 0 && DstAS != 0)
    return false;

  bool Changed = false;
  Value *Src = Cast->getOperand(0);
  // Worklist of (old value, replacement value) pairs.  Using a vector
  // implements a simple depth-first traversal.
  SmallVector<std::pair<Value *, Value *>, 8> Worklist;
  Worklist.push_back({Cast, Src});
  // Map from original values to their replacements.  This avoids recreating
  // instructions multiple times when they are used by several casts.
  DenseMap<Value *, Value *> Replacements;
  Replacements[Cast] = Src;
  // Collect instructions to remove once rewriting has finished.
  SmallVector<Instruction *, 16> ToErase;

  // Process the worklist.  Each entry consists of an old SSA value and the
  // replacement value that should be used in its stead.
  while (!Worklist.empty()) {
    auto [OldVal, NewVal] = Worklist.pop_back_val();
    // Copy the user list since we may modify it during iteration.
    SmallVector<User *, 8> Users(OldVal->users().begin(), OldVal->users().end());
    for (User *U : Users) {
      // Do not attempt to rewrite constant expressions here.  They should be
      // handled separately by canonicalizeConstantAddrSpaceCasts().
      if (auto *CE = dyn_cast<ConstantExpr>(U))
        continue;

      if (auto *NextCast = dyn_cast<AddrSpaceCastInst>(U)) {
        // Collapse chains of addrspacecasts.  Simply propagate the source value
        // forward and mark the cast for removal.
        Replacements[NextCast] = NewVal;
        Worklist.push_back({NextCast, NewVal});
        ToErase.push_back(NextCast);
        Changed = true;
        continue;
      }

      if (auto *BC = dyn_cast<BitCastInst>(U)) {
        // Replace bitcasts by casting the replacement value to the desired type.
        IRBuilder<> Builder(BC);
        Value *NewBC = Builder.CreateBitCast(NewVal, BC->getType());
        Replacements[BC] = NewBC;
        Worklist.push_back({BC, NewBC});
        BC->replaceAllUsesWith(NewBC);
        ToErase.push_back(BC);
        Changed = true;
        continue;
      }

      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        // Recreate the GEP with the new base pointer.  Use the original element
        // type from the GEP as the pointee type.  Preserve inbounds if set.
        IRBuilder<> Builder(GEP);
        SmallVector<Value *, 8> Indices;
        Indices.reserve(GEP->getNumIndices());
        for (auto Idx = GEP->idx_begin(); Idx != GEP->idx_end(); ++Idx)
          Indices.push_back(*Idx);
        Value *NewGEP = Builder.CreateGEP(
            GEP->getSourceElementType(), NewVal, Indices, "", GEP->isInBounds());
        Replacements[GEP] = NewGEP;
        Worklist.push_back({GEP, NewGEP});
        GEP->replaceAllUsesWith(NewGEP);
        ToErase.push_back(GEP);
        Changed = true;
        continue;
      }

      if (auto *LI = dyn_cast<LoadInst>(U)) {
        // Recreate the load using the new pointer.  Copy alignment, ordering
        // and volatility from the original instruction.
        IRBuilder<> Builder(LI);
        auto *Ty = LI->getType();
        LoadInst *NewLoad = Builder.CreateLoad(Ty, NewVal);
        NewLoad->setAlignment(LI->getAlign());
        NewLoad->setOrdering(LI->getOrdering());
        NewLoad->setSyncScopeID(LI->getSyncScopeID());
        NewLoad->setVolatile(LI->isVolatile());
        // Copy debug location and metadata.
        NewLoad->setDebugLoc(LI->getDebugLoc());
        SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
        LI->getAllMetadata(MDs);
        for (auto &MD : MDs)
          NewLoad->setMetadata(MD.first, MD.second);
        LI->replaceAllUsesWith(NewLoad);
        ToErase.push_back(LI);
        Changed = true;
        continue;
      }

      if (auto *SI = dyn_cast<StoreInst>(U)) {
        // Only rewrite stores where the pointer operand is the cast.  If the
        // stored value is itself the cast we leave it unchanged; the new
        // pointer type will be propagated through other uses.
        if (SI->getPointerOperand() == OldVal) {
          IRBuilder<> Builder(SI);
          Value *Stored = SI->getValueOperand();
          StoreInst *NewStore = Builder.CreateStore(Stored, NewVal);
          NewStore->setAlignment(SI->getAlign());
          NewStore->setOrdering(SI->getOrdering());
          NewStore->setSyncScopeID(SI->getSyncScopeID());
          NewStore->setVolatile(SI->isVolatile());
          NewStore->setDebugLoc(SI->getDebugLoc());
          SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
          SI->getAllMetadata(MDs);
          for (auto &MD : MDs)
            NewStore->setMetadata(MD.first, MD.second);
          ToErase.push_back(SI);
          Changed = true;
        }
        continue;
      }

      if (auto *Phi = dyn_cast<PHINode>(U)) {
        // Before creating a replacement PHI, check that every incoming value
        // that we are NOT replacing is already compatible with the new type
        // (i.e. has the same address space as NewVal).  If any unreplaced
        // incoming has a different address space, promoting this PHI would
        // produce invalid IR — bail out instead.
        Type *NewTy = NewVal->getType();
        bool Compatible = true;
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
          Value *Incoming = Phi->getIncomingValue(i);
          if (Incoming == OldVal)
            continue; // this slot is being replaced
          auto MIt = Replacements.find(Incoming);
          Type *IncomingTy = (MIt != Replacements.end()) ? MIt->second->getType()
                                                         : Incoming->getType();
          if (IncomingTy != NewTy) {
            Compatible = false;
            break;
          }
        }
        if (!Compatible) {
          Worklist.clear();
          return false;
        }

        // If we haven't created a replacement PHI yet, do so now.  Use the
        // replacement pointer type for the new PHI.
        PHINode *NewPhi = nullptr;
        auto It = Replacements.find(Phi);
        if (It != Replacements.end()) {
          NewPhi = cast<PHINode>(It->second);
        } else {
          IRBuilder<> Builder(Phi);
          NewPhi = Builder.CreatePHI(NewTy, Phi->getNumIncomingValues());
          Replacements[Phi] = NewPhi;
          Worklist.push_back({Phi, NewPhi});
        }
        // Populate the incoming values for the new PHI.  Use the rewritten
        // values where available.
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
          Value *Incoming = Phi->getIncomingValue(i);
          BasicBlock *BB = Phi->getIncomingBlock(i);
          Value *Mapped = Incoming;
          auto MIt = Replacements.find(Incoming);
          if (Incoming == OldVal)
            Mapped = NewVal;
          else if (MIt != Replacements.end())
            Mapped = MIt->second;
          NewPhi->addIncoming(Mapped, BB);
        }
        Phi->replaceAllUsesWith(NewPhi);
        ToErase.push_back(Phi);
        Changed = true;
        continue;
      }

      if (auto *Sel = dyn_cast<SelectInst>(U)) {
        // Create a new select instruction with updated operands.
        IRBuilder<> Builder(Sel);
        Value *Cond = Sel->getCondition();
        Value *TrueVal = Sel->getTrueValue();
        Value *FalseVal = Sel->getFalseValue();
        auto MapOp = [&](Value *Op) -> Value * {
          if (Op == OldVal)
            return NewVal;
          auto It = Replacements.find(Op);
          if (It != Replacements.end())
            return It->second;
          return Op;
        };
        Value *NewTrue = MapOp(TrueVal);
        Value *NewFalse = MapOp(FalseVal);
        Value *NewSel = Builder.CreateSelect(Cond, NewTrue, NewFalse);
        Replacements[Sel] = NewSel;
        Worklist.push_back({Sel, NewSel});
        Sel->replaceAllUsesWith(NewSel);
        ToErase.push_back(Sel);
        Changed = true;
        continue;
      }

      // If we encounter an unsupported use, bail out by clearing the worklist
      // and cancel rewriting for this cast.  Leave the original cast in place.
      // This prevents generation of invalid IR in cases we don't handle.
      Worklist.clear();
      return false;
    }
  }

  // Remove all old instructions that were replaced.
  for (Instruction *I : ToErase)
    I->eraseFromParent();

  return Changed;
}

} // namespace

llvm::PreservedAnalyses
AddrSpaceCastCanonicalizationPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool Changed = false;
  // First canonicalize constant expression addrspacecasts on globals.
  Changed |= canonicalizeConstantAddrSpaceCasts(M);

  // Next canonicalize addrspacecast instructions on SSA values.  Iterate over
  // the module to find all such instructions.  We collect them in a vector
  // first since rewriting modifies the instruction list.
  SmallVector<AddrSpaceCastInst *, 16> Casts;
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *Cast = dyn_cast<AddrSpaceCastInst>(&I)) {
          Casts.push_back(Cast);
        }
      }
    }
  }

  for (AddrSpaceCastInst *Cast : Casts) {
    // Skip casts of constant values; these are handled separately.
    if (isa<Constant>(Cast->getOperand(0)))
      continue;
    // Attempt to rewrite the cast and its uses.  The helper returns false
    // if it encountered unsupported constructs and opted not to change the IR.
    Changed |= rewriteAddrSpaceCastUses(Cast);
    // If the cast still has users after rewriting, leave it in place.  It
    // will be removed by later passes when safe to do so.
    if (Cast->use_empty()) {
      Cast->eraseFromParent();
      Changed = true;
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
