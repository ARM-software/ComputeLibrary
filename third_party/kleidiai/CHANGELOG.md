<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Changelog

KleidiAI follows the [Semantic Versioning](https://semver.org/) specification for releases.

## v0.5.0

- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN and 1xN) of QSI8D32 LHS (dynamic 8-bit integer per-block quantized) and QSI4C32 RHS (4-bit integer per-block quantized) to produce F32 output.
    - Optimizations for FEAT_DotProd.
  - Matrix multiplication (MxN and 1xN) of QAI8DX LHS (dynamic 8-bit integer per-row quantized) and QSI4CX RHS (4-bit integer per-channel quantized) to produce F32 output.
    - Optimizations for FEAT_DotProd and FEAT_I8MM.
    - Packing micro-kernels for LHS and non-transposed and transposed RHS.
  - Matrix multiplication (MxN) of BF16 LHS and BF16 RHS to produce F16 output.
    - Packing micro-kernels for LHS and non-transposed RHS.
- New SME micro-kernels:
  - Matrix multiplication (MxN and 1xN) of F16 LHS and F16 RHS to produce F16 output.
    - Packing micro-kernels for LHS and non-transposed and transposed RHS.
  - Matrix multiplication (MxN) of QAI8 LHS and QSI8 RHS to produce QAI8 output.
    - Packing micro-kernels for LHS and non-transposed RHS.
  - Matrix multiplication (MxN and 1xN) of QSI8D32 LHS and QSI4C32 RHS to produce F32 output
- Packing micro-kernels for QSI8D32 LHS and non-transposed QSI4C32 RHS, to work with the SME matrix multiplication (MxN and 1xN) micro-kernels.
- Fixes:
  - Fixes relating to illegal instruction errors on systems with SME but without SVE support:
    - Contain SME assembly inside the SMSTART and SMSTOP boundary.
    - Disable compiler generated SVE instructions by adding the -fno-tree-vectorize compiler option to the build.
  - Fix build warnings in the core library introduced by the -Wpedantic compiler option.
  - Fix typos in the micro-kernel interface files.

## v0.4.0

- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN) of QAI8DX (dynamically quantized 8-bit integer) LHS and QSI4CX (quantized 4-bit integer) RHS with F32 output.
  - Matrix multiplication (MxN and 1xN) of BF16 LHS and RHS with F32 output.
- New SME micro-kernels:
  - SME2 F32 matrix multiplication (1xN) micro-kernels:
    - Compatible with 2VL RHS packing, for sharing one packed RHS with SME2 F32 GEMM micro-kernel.
    - Compatible with 16VL RHS packing.
  - SME F32 packing function for transposed RHS matrix.
- Enhancements to existing micro-kernels:
  - Port several quantized micro-kernels to optimized Advanced SIMD assembly.
- Register SME F32 matrix multiplication micro-kernel in the benchmark suite.
- Enable air gapped CMake builds through local third-party dependencies.

## v0.3.0

- Advanced SIMD FP32 GEMM micro-kernel.
- Micro-kernels to compute the matrix multiplication of dynamically quantized asymmetric signed 8-bit integer with per-row quantization (QAI8DX) LHS and quantized symmetric 4-bit signed integer with per-block quantization (QSI4C32) RHS. The destination matrix data type is single-precision floating-point (F32). The micro-kernels have been optimized using the Arm® CPU feature FEAT_I8MM for the matrix-by-matrix cases and the FEAT_DotProd for the vector-by-matrix cases.
- RHS matrix packing micro-kernels to pack the RHS matrix holding the QSI4C32 values.
- Unit test and example for integer micro-kernels.
- Extend support for signed 4-bit integer inputs in quantized symmetric 4-bit signed integer with per-channel quantization (QSI4CXP) RHS packing micro-kernel.
  - kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0 renamed to kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.
  - kai_rhs_pack_kxn_qsi4cxp_qsu4cxs1s0 renamed to kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.
- Remove FP16 GEMV micro-kernel optimized for Advanced SIMD.
  - Where a dedicated GEMV micro-kernel is not provided, it is recommended to use existing GEMM micro-kernels which have dedicated paths for M=1 (a "GEMV" operation).

## v0.2.0

- Micro-kernels to compute the matrix multiplication of dynamically quantized symmetric signed 8-bit integer with
  per-block quantization (QSI8D32) activations and quantized symmetric 4-bit signed integer with per-block quantization
  (QSI4C32) weights and the accumulation of the result into a single-precision (F32) output,
  optimized for Arm® Neon™ technology.
- Tensor packing micro-kernels to prepare the activations and weights for input to the above matrix multiplication
  micro-kernel.
- Unit test and example for integer micro-kernels.

## v0.1.0

The first release of KleidiAI includes:

- Micro-kernels to compute the matrix multiplication of:
  - Dynamically quantized 8-bit integer (QAI8DX) activations and quantized 4-bit integer (QSI4CX) weights and the
    accumulation of the result into a single-precision (F32) output, optimized for Arm® Neon™ technology.
  - Half precision floating-point (F16) activations and weights and the accumulation of the result into an F16 output,
    optimized for Neon technology.
  - F32 activations and weights and the accumulation of the result into an F32 output, optimized for SME2 technology.
- Tensor packing micro-kernels to prepare the activations and weights for input to the above matrix multiplication
  micro-kernels.
- Examples and documentation demonstrating the usage of the 4-bit integer and 16-bit floating point matrix
  multiplication micro-kernels.
- Testing suite.
- CMake and Bazel build system for micro kernels.
