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
#ifndef HIPSYCL_SSCP_GLOBAL_PTR_LOAD_PROMOTION_PASS_HPP
#define HIPSYCL_SSCP_GLOBAL_PTR_LOAD_PROMOTION_PASS_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

/// \brief Promotes generic (AS=0) pointer values that logically live in global
/// (device) memory to the global address space (AS = GlobalAS).
///
/// After \c AddressSpaceInferencePass, loop-carried pointer loads may remain
/// typed as generic \c ptr (AS=0) even though they are loaded from global
/// memory.  For example, a linked-list traversal kernel produces:
///
///   %4  = addrspacecast ptr addrspace(1) %arg to ptr   ; downcast
///   %phi = phi ptr [ %loaded, loop ], [ %4, entry ]    ; all AS=0
///   %gep = getelementptr i8, ptr %phi, 8
///   %loaded = load ptr, ptr %gep                       ; stays AS=0
///
/// This pass promotes all values "logically in global memory" to
/// \c ptr addrspace(1) so that subsequent passes (canonicalization, annotation)
/// operate on clean, consistent address-space information.
///
/// A value is "logically global" if:
///   - Its LLVM type is already ptr addrspace(GlobalAS), OR
///   - It is an addrspacecast \em from GlobalAS to AS=0, OR
///   - It is a GEP whose base pointer is logically global, OR
///   - It is a PHI where at least one incoming is logically global and none is
///     explicitly from a non-global address space (threadgroup, private, …), OR
///   - It is a load of pointer type from a logically-global address.
///
/// Scalar (non-pointer) loads and stores that use a promoted pointer operand
/// are updated in-place; their result types are unchanged.
///
/// After this pass, \c AddrSpaceCastCanonicalizationPass can cleanly eliminate
/// the remaining downcasts, and \c PointerTranslationAnnotationPass will find
/// all pointer loads in AS=1 and mark them for address translation.
class GlobalPtrLoadPromotionPass
    : public llvm::PassInfoMixin<GlobalPtrLoadPromotionPass> {
public:
  explicit GlobalPtrLoadPromotionPass(unsigned GlobalAS);
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

private:
  unsigned GlobalAS;
};

} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SSCP_GLOBAL_PTR_LOAD_PROMOTION_PASS_HPP
