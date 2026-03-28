#!/usr/bin/env bash
# Generates LLVM IR for dynamic_function.cpp on Linux.
# Run this on a Linux machine where acpp is installed in ~/opt/adaptivecpp.
#
# Usage: ./gen-ir-linux.sh [path/to/dynamic_function.cpp]
#
# Output:
#   /tmp/df_before.ll  — raw clang IR, no AdaptiveCpp passes
#   /tmp/df_after.ll   — IR after all AdaptiveCpp passes

set -e

ACPP_DIR="$HOME/opt/adaptivecpp"
SOURCE="${1:-$(dirname "$0")/../tests/compiler/sscp/dynamic_function.cpp}"
SOURCE="$(realpath "$SOURCE")"

if [[ ! -f "$SOURCE" ]]; then
    echo "ERROR: source file not found: $SOURCE" >&2
    exit 1
fi

# Detect clang from acpp's dry-run output
ACPP_CMD=$("$ACPP_DIR/bin/acpp" --acpp-targets=generic --acpp-dryrun "$SOURCE" -o /dev/null 2>/dev/null | head -1)
CLANGXX=$(echo "$ACPP_CMD" | awk '{print $1}')

if [[ ! -x "$CLANGXX" ]]; then
    # Fallback: find clang++ next to acpp
    CLANGXX=$(find "$ACPP_DIR" -name "clang++" | head -1)
fi

if [[ ! -x "$CLANGXX" ]]; then
    echo "ERROR: could not find clang++. Set CLANGXX env var manually." >&2
    exit 1
fi

echo "Using clang++: $CLANGXX"
echo "Source: $SOURCE"

COMMON_FLAGS=(
    -isystem "$ACPP_DIR/include/AdaptiveCpp"
    -D__OPENSYCL__ -D__HIPSYCL__ -D__ADAPTIVECPP__ -D__ACPP__
    -std=c++17
    -D__ACPP_ENABLE_LLVM_SSCP_TARGET__
    -Xclang -disable-O0-optnone
    -S -emit-llvm
)

echo ""
echo "=== Generating /tmp/df_before.ll (no AdaptiveCpp passes) ==="
"$CLANGXX" "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o /tmp/df_before.ll
echo "Done: $(wc -l < /tmp/df_before.ll) lines"

echo ""
echo "=== Generating /tmp/df_after.ll (with AdaptiveCpp passes) ==="
"$CLANGXX" "${COMMON_FLAGS[@]}" \
    -mllvm -acpp-sscp \
    -fplugin="$ACPP_DIR/lib/libacpp-clang.so" \
    -fpass-plugin="$ACPP_DIR/lib/libacpp-clang.so" \
    "$SOURCE" -o /tmp/df_after.ll
echo "Done: $(wc -l < /tmp/df_after.ll) lines"

echo ""
echo "=== C1/C2 constructor definitions ==="
echo "--- df_before.ll ---"
grep -n "define.*dynamic_function_def.*C[12]EPF" /tmp/df_before.ll \
    | sed 's/define [^ ]* [^ ]* //' | sed 's/(.*$//'

echo ""
echo "--- df_after.ll ---"
grep -n "define.*dynamic_function_def.*C[12]EPF" /tmp/df_after.ll \
    | sed 's/define [^ ]* [^ ]* //' | sed 's/(.*$// ' \
    || echo "(not present — may have been aliased or removed)"

echo ""
echo "=== Checking for C1=alias(C2) pattern ==="
grep -n "alias.*dynamic_function_def.*C[12]EPF" /tmp/df_before.ll \
    || echo "No alias found in df_before.ll"

echo ""
echo "=== Annotation calls in df_before.ll ==="
grep -n "__acpp_function_annotation_dynamic_function_def_arg0\b" /tmp/df_before.ll \
    | head -10

echo ""
echo "All done. Files:"
ls -lh /tmp/df_before.ll /tmp/df_after.ll
