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
#include <llvm/IR/ValueHandle.h>

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
///
/// Design notes:
///   - New replacement instructions are created eagerly (inserted into the IR)
///     but replaceAllUsesWith is deferred until the entire walk succeeds.
///     On bail-out the new instructions are unreferenced dead code (safe,
///     cleaned up by later DCE) and the original instructions are untouched.
///   - dropAllReferences is called on every instruction before eraseFromParent
///     so that the erase order does not matter (avoids assert in debug builds
///     when an instruction in ToErase is still referenced by another one).
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

  // Worklist of (old value, replacement value) pairs.
  SmallVector<std::pair<Value *, Value *>, 8> Worklist;
  Worklist.push_back({Cast, Src});

  // Map from original values to their replacements.
  DenseMap<Value *, Value *> Replacements;
  Replacements[Cast] = Src;

  // Instructions to erase after all replacements are wired up.
  SmallVector<Instruction *, 16> ToErase;

  // Deferred replaceAllUsesWith: applied only after the entire walk succeeds.
  // This ensures that a bail-out on an unsupported use leaves the IR valid —
  // new instructions exist but are unreferenced dead code; no original
  // instruction has had its uses redirected to a partial replacement.
  SmallVector<std::pair<Instruction *, Value *>, 16> DeferredReplace;

  while (!Worklist.empty()) {
    Value *OldVal = Worklist.back().first;
    Value *NewVal = Worklist.back().second;
    Worklist.pop_back();
    // Copy the user list since we may modify it during iteration.
    SmallVector<User *, 8> Users(OldVal->users().begin(), OldVal->users().end());
    for (User *U : Users) {
      // Do not attempt to rewrite constant expressions here.  They should be
      // handled separately by canonicalizeConstantAddrSpaceCasts().
      if (isa<ConstantExpr>(U))
        continue;

      // Skip users that are already in Replacements (visited via another path).
      if (Replacements.count(U))
        continue;

      if (auto *NextCast = dyn_cast<AddrSpaceCastInst>(U)) {
        // Collapse chains of addrspacecasts: propagate the source value forward.
        Replacements[NextCast] = NewVal;
        Worklist.push_back({NextCast, NewVal});
        ToErase.push_back(NextCast);
        Changed = true;
        continue;
      }

      if (auto *BC = dyn_cast<BitCastInst>(U)) {
        IRBuilder<> Builder(BC);
        Value *NewBC = Builder.CreateBitCast(NewVal, BC->getType());
        Replacements[BC] = NewBC;
        Worklist.push_back({BC, NewBC});
        DeferredReplace.push_back({BC, NewBC});
        ToErase.push_back(BC);
        Changed = true;
        continue;
      }

      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        IRBuilder<> Builder(GEP);
        SmallVector<Value *, 8> Indices;
        Indices.reserve(GEP->getNumIndices());
        for (auto Idx = GEP->idx_begin(); Idx != GEP->idx_end(); ++Idx)
          Indices.push_back(*Idx);
        Value *NewGEP = Builder.CreateGEP(
            GEP->getSourceElementType(), NewVal, Indices, "", GEP->isInBounds());
        Replacements[GEP] = NewGEP;
        Worklist.push_back({GEP, NewGEP});
        DeferredReplace.push_back({GEP, NewGEP});
        ToErase.push_back(GEP);
        Changed = true;
        continue;
      }

      if (auto *LI = dyn_cast<LoadInst>(U)) {
        IRBuilder<> Builder(LI);
        auto *Ty = LI->getType();
        LoadInst *NewLoad = Builder.CreateLoad(Ty, NewVal);
        NewLoad->setAlignment(LI->getAlign());
        NewLoad->setOrdering(LI->getOrdering());
        NewLoad->setSyncScopeID(LI->getSyncScopeID());
        NewLoad->setVolatile(LI->isVolatile());
        NewLoad->setDebugLoc(LI->getDebugLoc());
        SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
        LI->getAllMetadata(MDs);
        for (auto &MD : MDs)
          NewLoad->setMetadata(MD.first, MD.second);
        DeferredReplace.push_back({LI, NewLoad});
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
        // On bail-out DeferredReplace has not been applied yet, so IR is valid.
        Type *NewTy = NewVal->getType();
        bool Compatible = true;
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
          Value *Incoming = Phi->getIncomingValue(i);
          if (Incoming == OldVal)
            continue;
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

        PHINode *NewPhi = nullptr;
        auto It = Replacements.find(Phi);
        if (It != Replacements.end()) {
          NewPhi = cast<PHINode>(It->second);
        } else {
          IRBuilder<> Builder(Phi);
          NewPhi = Builder.CreatePHI(NewTy, Phi->getNumIncomingValues());
          Replacements[Phi] = NewPhi;
          Worklist.push_back({Phi, NewPhi});
          DeferredReplace.push_back({Phi, NewPhi});
          ToErase.push_back(Phi);
        }
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
        Changed = true;
        continue;
      }

      if (auto *Sel = dyn_cast<SelectInst>(U)) {
        IRBuilder<> Builder(Sel);
        Value *Cond = Sel->getCondition();
        auto MapOp = [&](Value *Op) -> Value * {
          if (Op == OldVal)
            return NewVal;
          auto It = Replacements.find(Op);
          if (It != Replacements.end())
            return It->second;
          return Op;
        };
        Value *NewSel = Builder.CreateSelect(Cond, MapOp(Sel->getTrueValue()),
                                             MapOp(Sel->getFalseValue()));
        Replacements[Sel] = NewSel;
        Worklist.push_back({Sel, NewSel});
        DeferredReplace.push_back({Sel, NewSel});
        ToErase.push_back(Sel);
        Changed = true;
        continue;
      }

      // Unsupported use — bail out without having wired up any replacements
      // (DeferredReplace not applied yet, so the IR remains valid).
      Worklist.clear();
      return false;
    }
  }

  // Wire up replacements now that we know every use can be handled.
  for (auto &[OldInst, NewInst] : DeferredReplace)
    OldInst->replaceAllUsesWith(NewInst);

  // Drop all operand references before erasing so that the erase order does
  // not matter.  Without this, erasing instruction A while instruction B
  // (also in ToErase) still holds A as an operand trips a debug-build assert
  // inside ~Value() that checks use_empty().
  for (Instruction *I : ToErase)
    I->dropAllReferences();
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

  // Next canonicalize addrspacecast instructions on SSA values.  Collect them
  // into WeakTrackingVH handles first, because rewriteAddrSpaceCastUses may
  // erase intermediate casts in a chain (e.g. A→B→GEP: processing A erases B).
  // A raw pointer to B would become dangling; WeakTrackingVH becomes null when
  // the tracked instruction is deleted, allowing the loop to skip it safely.
  SmallVector<WeakTrackingVH, 16> Casts;
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isa<AddrSpaceCastInst>(&I))
          Casts.push_back(&I);
      }
    }
  }

  for (WeakTrackingVH VH : Casts) {
    // VH becomes null when the instruction was erased by a prior iteration
    // (e.g. as part of collapsing an addrspacecast chain).
    if (!VH)
      continue;
    auto *Cast = cast<AddrSpaceCastInst>(VH.operator Value *());
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
