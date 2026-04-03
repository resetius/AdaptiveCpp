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
#ifndef HIPSYCL_SSCP_ADDRSPACE_CAST_CANONICALIZATION_PASS_HPP
#define HIPSYCL_SSCP_ADDRSPACE_CAST_CANONICALIZATION_PASS_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

/// \brief A pass that canonicalizes address space casts in the LLVM IR.
///
/// The emitter for some backends (such as the Metal backend) currently has
/// special handling to walk through chains of addrspacecast instructions in
/// order to determine the "physical" address space of a pointer.  This pass
/// attempts to remove the need for such logic by rewriting IR so that
/// addrspacecast instructions become unnecessary.  It performs two main tasks:
///
/// 1. For constant globals that are cast into a different address space via a
///    constant expression addrspacecast, the pass clones the global into the
///    target address space and rewrites all users of the cast to reference the
///    new global directly.  This ensures that constant pointers live in the
///    correct address space from the outset.
///
/// 2. For addrspacecast instructions operating on SSA values, the pass
///    recursively rewrites uses of the cast value to operate on the original
///    source value instead.  Supported instructions include loads, stores,
///    bitcasts, getelementptr, phi and select.  For these instructions the
///    pass constructs new instructions using the original pointer and updates
///    their uses accordingly.  Chains of addrspacecast instructions are also
///    collapsed.  If an instruction cannot be rewritten safely, the cast is
///    left in place.
///
/// The result is that most addrspacecast instructions can be eliminated from
/// the IR, simplifying subsequent code generation.
class AddrSpaceCastCanonicalizationPass
    : public llvm::PassInfoMixin<AddrSpaceCastCanonicalizationPass> {
public:
  /// Run the pass on the given module.
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SSCP_ADDRSPACE_CAST_CANONICALIZATION_PASS_HPP
