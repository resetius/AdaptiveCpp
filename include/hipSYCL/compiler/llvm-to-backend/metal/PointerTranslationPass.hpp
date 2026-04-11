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
#ifndef HIPSYCL_SSCP_POINTER_TRANSLATION_PASS_HPP
#define HIPSYCL_SSCP_POINTER_TRANSLATION_PASS_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

/// \brief Inserts host↔GPU pointer address translation around annotated
/// loads and stores in the Metal backend.
///
/// Reads the metadata annotations produced by PointerTranslationAnnotationPass
/// and rewrites each annotated instruction:
///
///   Load  (needs_translation): the loaded host pointer is adjusted to a GPU
///   pointer by byte-offsetting with the per-buffer delta:
///       %delta    = call i64 @__acpp_sscp_metal_symbol_id(ptr @.addr_diff_name)
///       %adjusted = getelementptr i8, ptr %loaded, i64 %delta
///       <replace all uses of %loaded with %adjusted>
///
///   Store (needs_translation): the GPU pointer value is adjusted back to a
///   host pointer before writing to global memory:
///       %neg      = sub i64 0, %delta
///       %adjusted = getelementptr i8, ptr %val, i64 %neg
///       <replace the stored value with %adjusted>
///
/// The delta call is hoisted to the function entry block and shared by all
/// translations within the same function.
///
/// The pass relies on the Emitter's existing __acpp_sscp_metal_symbol_id
/// handling: a call __acpp_sscp_metal_symbol_id("sym") is emitted as
///   name = sym;
/// so the call becomes a direct read of the MSL constant-buffer global
/// __acpp_sscp_metal_gpu_to_host_addr_diff [[buffer(1)]].
class PointerTranslationPass
    : public llvm::PassInfoMixin<PointerTranslationPass> {
public:
  explicit PointerTranslationPass(unsigned GlobalAS);

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

private:
  unsigned GlobalAS;
};

} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SSCP_POINTER_TRANSLATION_PASS_HPP
