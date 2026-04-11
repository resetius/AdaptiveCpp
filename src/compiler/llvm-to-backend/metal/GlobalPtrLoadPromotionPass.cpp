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

#include "hipSYCL/compiler/llvm-to-backend/metal/GlobalPtrLoadPromotionPass.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

using namespace llvm;

namespace hipsycl {
namespace compiler {

GlobalPtrLoadPromotionPass::GlobalPtrLoadPromotionPass(unsigned GlobalAS)
    : GlobalAS(GlobalAS) {}

namespace {

static bool isPtrOfAS(Type *T, unsigned AS) {
  return T->isPointerTy() && cast<PointerType>(T)->getAddressSpace() == AS;
}

/// Returns true if the address space is a well-known non-global space.
/// Generic (0) is considered unknown, not explicitly non-global.
static bool isExplicitlyNonGlobal(unsigned AS, unsigned GlobalAS) {
  return AS != 0 && AS != GlobalAS;
}

} // namespace

llvm::PreservedAnalyses
GlobalPtrLoadPromotionPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool AnyChanged = false;
  LLVMContext &Ctx = M.getContext();
  Type *GlobalPtrTy = PointerType::get(Ctx, GlobalAS);

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // -----------------------------------------------------------------------
    // Phase 1 – Analysis: collect values that are "logically global"
    //
    // A value is logically global if:
    //   (a) Its type is already ptr addrspace(GlobalAS), OR
    //   (b) It is an addrspacecast from GlobalAS to AS=0 (downcast), OR
    //   (c) It is a GEP whose base is logically global, OR
    //   (d) It is a pointer-typed PHI where at least one incoming is logically
    //       global and no incoming is in an explicitly non-global space, OR
    //   (e) It is a ptr-typed load from a logically-global address.
    // -----------------------------------------------------------------------
    DenseSet<Value *> LogicallyGlobal;
    SmallVector<Value *, 16> Worklist;

    auto Seed = [&](Value *V) {
      if (!LogicallyGlobal.insert(V).second)
        return;
      Worklist.push_back(V);
    };

    // Seed kernel arguments already in GlobalAS.
    for (Argument &A : F.args())
      if (isPtrOfAS(A.getType(), GlobalAS))
        Seed(&A);

    // Seed instructions already typed as GlobalAS pointers.
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (isPtrOfAS(I.getType(), GlobalAS))
          Seed(&I);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();

      for (User *U : V->users()) {
        if (LogicallyGlobal.count(U))
          continue;
        auto *I = dyn_cast<Instruction>(U);
        if (!I || !I->getType()->isPointerTy())
          continue;

        if (auto *ASC = dyn_cast<AddrSpaceCastInst>(I)) {
          // Downcast GlobalAS → generic (0).
          if (ASC->getOperand(0)->getType()->getPointerAddressSpace() == GlobalAS &&
              ASC->getType()->getPointerAddressSpace() == 0)
            Seed(ASC);
          continue;
        }

        if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
          if (GEP->getPointerOperand() == V)
            Seed(GEP);
          continue;
        }

        if (auto *LI = dyn_cast<LoadInst>(I)) {
          // Only pointer-typed loads propagate global-ness.
          if (LI->getPointerOperand() == V && LI->getType()->isPointerTy())
            Seed(LI);
          continue;
        }

        if (auto *Phi = dyn_cast<PHINode>(I)) {
          // Accept only if no incoming is in a known non-global space.
          bool AnyNonGlobal = false;
          for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
            unsigned AS =
                Phi->getIncomingValue(i)->getType()->getPointerAddressSpace();
            if (isExplicitlyNonGlobal(AS, GlobalAS)) {
              AnyNonGlobal = true;
              break;
            }
          }
          if (!AnyNonGlobal)
            Seed(Phi);
          continue;
        }
      }
    }

    // -----------------------------------------------------------------------
    // Phase 2 – Rewrite: create promoted (AS=GlobalAS) versions of all
    // logically-global values that are currently typed as AS=0.
    // -----------------------------------------------------------------------
    // Map: AS=0 logically-global Value* → its AS=GlobalAS replacement.
    DenseMap<Value *, Value *> Promoted;

    auto GetPromoted = [&](Value *V) -> Value * {
      if (isPtrOfAS(V->getType(), GlobalAS))
        return V; // already in GlobalAS
      auto It = Promoted.find(V);
      return (It != Promoted.end()) ? It->second : nullptr;
    };

    // Step 2a – Create empty promoted PHI shells first to handle cycles.
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *Phi = dyn_cast<PHINode>(&I);
        if (!Phi)
          continue;
        if (!LogicallyGlobal.count(Phi))
          continue;
        if (Phi->getType()->getPointerAddressSpace() == GlobalAS)
          continue; // already promoted — no work needed

        IRBuilder<> Builder(Phi);
        PHINode *NewPhi = Builder.CreatePHI(GlobalPtrTy,
                                            Phi->getNumIncomingValues(),
                                            Phi->getName() + ".pg");
        Promoted[Phi] = NewPhi;
      }
    }

    // Step 2b – Promote non-PHI instructions in program order so that
    // dependencies are already resolved when we encounter each instruction.
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isa<PHINode>(&I))
          continue;
        if (!LogicallyGlobal.count(&I))
          continue;
        if (I.getType()->getPointerAddressSpace() == GlobalAS)
          continue; // already in GlobalAS

        if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
          // Downcast from GlobalAS → 0: the promoted version is just the source.
          Promoted[ASC] = ASC->getOperand(0);
          continue;
        }

        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          Value *PromBase = GetPromoted(GEP->getPointerOperand());
          if (!PromBase)
            continue; // dependency not yet promoted (shouldn't happen in SSA order)
          IRBuilder<> Builder(GEP);
          SmallVector<Value *, 8> Indices(GEP->idx_begin(), GEP->idx_end());
          Value *NewGEP =
              Builder.CreateGEP(GEP->getSourceElementType(), PromBase, Indices,
                                GEP->getName() + ".pg", GEP->isInBounds());
          Promoted[GEP] = NewGEP;
          continue;
        }

        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          Value *PromPtr = GetPromoted(LI->getPointerOperand());
          if (!PromPtr)
            continue;
          IRBuilder<> Builder(LI);
          LoadInst *NewLI = Builder.CreateLoad(GlobalPtrTy, PromPtr,
                                               LI->getName() + ".pg");
          NewLI->setAlignment(LI->getAlign());
          NewLI->setOrdering(LI->getOrdering());
          NewLI->setSyncScopeID(LI->getSyncScopeID());
          NewLI->setVolatile(LI->isVolatile());
          NewLI->setDebugLoc(LI->getDebugLoc());
          SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
          LI->getAllMetadata(MDs);
          for (auto &MD : MDs)
            NewLI->setMetadata(MD.first, MD.second);
          Promoted[LI] = NewLI;
          continue;
        }
      }
    }

    if (Promoted.empty())
      continue;

    // Step 2c – Fill promoted PHI incomings now that all other promoted
    // values are available.
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *Phi = dyn_cast<PHINode>(&I);
        if (!Phi || !Promoted.count(Phi))
          continue;
        PHINode *NewPhi = cast<PHINode>(Promoted[Phi]);

        for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
          Value *Inc = Phi->getIncomingValue(i);
          BasicBlock *IncomingBB = Phi->getIncomingBlock(i);

          Value *PromInc = GetPromoted(Inc);
          if (!PromInc) {
            // Inc is not logically global (e.g., a null pointer in AS=0).
            // Insert an addrspacecast at the end of the incoming block.
            IRBuilder<> Builder(IncomingBB->getTerminator());
            PromInc = Builder.CreateAddrSpaceCast(Inc, GlobalPtrTy,
                                                   Inc->getName() + ".to_global");
          }
          NewPhi->addIncoming(PromInc, IncomingBB);
        }
      }
    }

    // Step 2d – Patch uses of old AS=0 values.
    //
    // For uses that belong to another promoted instruction, we skip them —
    // those instructions are being erased.  For all other users we insert a
    // downcast (GlobalAS → 0) so that the surrounding code continues to
    // receive a generic pointer.
    for (auto &[OldVal, NewVal] : Promoted) {
      SmallVector<Use *, 8> Uses;
      for (Use &U : OldVal->uses())
        Uses.push_back(&U);

      for (Use *U : Uses) {
        auto *UserInst = dyn_cast<Instruction>(U->getUser());
        if (!UserInst)
          continue;
        // Skip if the using instruction is itself being replaced.
        if (Promoted.count(UserInst))
          continue;

        // Insert a downcast immediately before the using instruction.
        IRBuilder<> Builder(UserInst);
        Value *Downcast = Builder.CreateAddrSpaceCast(
            NewVal, OldVal->getType(), NewVal->getName() + ".to_generic");
        U->set(Downcast);
      }
    }

    // Step 2e – Erase old promoted instructions.
    //
    // The old instructions may still reference each other (cycles through
    // PHIs).  Drop all references first to break the cycles, then erase.
    SmallVector<Instruction *, 16> ToErase;
    for (auto &[OldVal, NewVal] : Promoted) {
      if (auto *OldInst = dyn_cast<Instruction>(OldVal))
        ToErase.push_back(OldInst);
    }
    for (Instruction *I : ToErase)
      I->dropAllReferences();
    for (Instruction *I : ToErase)
      I->eraseFromParent();

    unsigned PromotedCount = static_cast<unsigned>(Promoted.size());
    HIPSYCL_DEBUG_INFO << "Metal GlobalPtrLoadPromotionPass: promoted "
                       << PromotedCount << " value(s) in " << F.getName()
                       << " to ptr addrspace(" << GlobalAS << ")\n";
    AnyChanged = true;
  }

  return AnyChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
