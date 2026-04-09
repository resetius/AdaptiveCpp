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
#ifndef HIPSYCL_SSCP_POINTER_TRANSLATION_ANNOTATION_PASS_HPP
#define HIPSYCL_SSCP_POINTER_TRANSLATION_ANNOTATION_PASS_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

/// Metadata key attached to load instructions that load a pointer value from
/// global (device) memory.  Such a value was written by the host CPU and
/// therefore contains a host virtual address.  The emitter (or a later
/// lowering pass) must add the per-allocation GPU/CPU address offset before
/// the pointer can be dereferenced on the GPU.
static constexpr const char *MetalPtrLoadNeedsTranslationMD =
    "acpp.metal.ptr.load.needs_translation";

/// Metadata key attached to store instructions that store a pointer value into
/// global (device) memory.  The pointer being stored is a GPU virtual address;
/// the emitter must subtract the per-allocation offset before writing so that
/// the host CPU can later read a valid host virtual address.
static constexpr const char *MetalPtrStoreNeedsTranslationMD =
    "acpp.metal.ptr.store.needs_translation";

/// \brief Annotates loads and stores that require host↔GPU pointer address
/// translation in the Metal backend.
///
/// On Apple Silicon, the CPU and GPU share physical memory but use different
/// virtual address spaces.  Pointers that arrive as kernel arguments are
/// already GPU virtual addresses and need no adjustment.  However, a pointer
/// value that is *loaded from* device memory was placed there by the CPU and
/// therefore contains a host virtual address — it must be adjusted by a
/// known offset before the GPU can follow it.  Symmetrically, a GPU pointer
/// that is *stored to* device memory must have the offset subtracted so that
/// the CPU can later interpret it correctly.
///
/// This pass runs after AddrSpaceCastCanonicalizationPass (which ensures that
/// every pointer already carries the correct Metal address space number) and
/// attaches one of the two metadata keys above to every qualifying instruction.
/// It does **not** insert any arithmetic instructions; the actual offset
/// adjustments are the responsibility of the emitter or a dedicated lowering
/// pass.
///
/// Annotation rules (see metal-ptr-translation-rules.md for full details):
///   - A load is annotated when its result type is a pointer AND the pointer
///     operand is in global address space (Metal LLVM AS = 1).
///   - A store is annotated when its value operand is a pointer AND the
///     pointer operand is in global address space (Metal LLVM AS = 1).
///   - Thread (AS=5), threadgroup (AS=3) and constant (AS=4) pointers are
///     never annotated.
class PointerTranslationAnnotationPass
    : public llvm::PassInfoMixin<PointerTranslationAnnotationPass> {
public:
  explicit PointerTranslationAnnotationPass(unsigned GlobalAS);

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

private:
  unsigned GlobalAS;
};

} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SSCP_POINTER_TRANSLATION_ANNOTATION_PASS_HPP
