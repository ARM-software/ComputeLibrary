<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Changelog

KleidiAI follows the [Semantic Versioning](https://semver.org/) specification for releases.

## Upcoming Release

## v1.19.0

- Added new unit test framework
- New SME2 micro-kernels
  - Matrix Multiplication (1xN) Micro-Kernel of QAI8DXP LHS and QSI4C32P RHS with F32 input and output.
  - Matrix Multiplication (MxN) Micro-Kernel of QAI8DXP LHS and QSI4C32P RHS with F32 input and output.
- Add clang-cl compiler macros for kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_asm micro-kernel
- Fixes
  - Fix incorrect API in BF16 kernel interface

## v1.18.0

- Fixes
  - Add Null Bias support for rhs_pack_kxn_x16p32x1b_x16_x16_neon.
  - Updated description of matmul file name from m_step x n_step to m_block x n_block
  - Clamp after scaling in `matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa`.
- New SVE micro-kernels
  - Matrix Multiplication (MxN) Micro-Kernels with F32 input and output.
- Update the example matmul_clamp_f32_qsi8d32p_qsi4c32p to demonstrate how a micro-kernel can be used in a multithreaded environment.
- Documentation
  - Update documentation to use markdown syntax
  - Added FAQ
  - Added a section that describes the directory structure and a section that describes the different example implementations

## v1.17.0

- Fixes
  - Add Null Bias support for rhs_pack_kxn_x32p16x1b_x32_x32_neon.
  - Some micro-kernels report incorrect m_step value.
    - kai_lhs_quant_pack_qai8dxp_bf16_neon
    - kai_lhs_quant_pack_qai8dxp_f16_neon
    - kai_lhs_quant_pack_qai8dxp_f32
    - kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon
    - kai_lhs_quant_pack_qsi8d32p_f32
    - kai_lhs_quant_pack_qsi8d32p_f32_neon
    - kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon
    - kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon

## v1.16.0

- Extended the benchmarking framework to support multiple operators.
  - Initial support for matrix multiplication (matmul) & indirect matrix multiplication (imatmul)
  - Added all imatmul and matmul micro-kernels to the benchmark suite
- Fixes:
  - All SME and SME2 micro-kernels now commit ZA lazy save buffer when building with SME support.
  - Fixed incorrect handling of zero point and scale into two packing kernels which caused incorrect de-quantisation is certain cases:
    - kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon
    - kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s1s0_f32_f32_f32_neon
- NEW SVE micro-kernels (256-bit Vector length specific):
  - Matrix multiplication (MxN) Micro-kernels of QSI8DX LHS and QSI4CX RHS with F32 input and output.
  - Matrix multiplication (1xN) Micro-kernels of QSI8DX LHS and QSI4CX RHS with F32 input and output.

## v1.15.1

- Fixes
  - Added missing checks for bf16 support for quantised matmuls with bf16 input/output.

## v1.15.0

- New SME micro-kernels:
  - Matrix multiplication (MxN) Micro-kernels of QAI8DX LHS and QSI8CX RHS with F32 input and output.
  - Matrix multiplication (1xN) Micro-kernels of QAI8DX LHS and QSI8CX RHS with F32 input and output.
- Wider compiler compatibility for the following kernels:
  - kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa
  - kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot
  - kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla
  - kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla
  - kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa
  - kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa
  - kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa
  - kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot
  - kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot
  - kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa
  - kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot
  - kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa

## v1.14.0

- New SME micro-kernels:
  - Indirect matrix multiplication (MxN) of QAI8 input and output.
  - Indirect matrix multiplication (MxN) of F16 input and output.
  - Indirect matrix multiplication (MxN) of F32 input and output.
  - Matrix multiplication (MxN) of QAI8 LHS and RHS with QAI8 output.
  - Depthwise Convolution RHS F32 Packing micro-kernel.
- New SME2 micro-kernels:
  - Depthwise Convolution (3x3) Planar micro-kernel of F32 LHS and Packed F32 RHS with F32 output using MLA.
- Convert SME2 matmul micro-kernels to pure assembly, and add MSVC support.
  - Affects: kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa
- Optimizations:
  - Packing micro-kernels kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s1s0_f32_f32_f32_neon and kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon have been further optimized.
  - Packing micro-kernel kai_lhs_quant_pack_qai8dxp_f16_neon has been further optimized.
- New Advanced SIMD micro-kernels:
  - Wider 6x32 block size variants of FP16 Matrix Multiplication, including a variant optimized for the Arm® Cortex®-A55 processor.
  - Wider 6x16 block size variants of FP32 Matrix Multiplication, including a variant optimized for the Arm® Cortex®-A55 processor.
- Fixes:
  - Fix out-of-bound read of intermediate values in kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa micro-kernel
  - Fix out-of-bounds write in kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla
  - Fix out-of-bounds read in kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot

## v1.13.0

- Improve performance of lhs_quant_pack_qsi8d32p_f32 using Advanced SIMD reimplemented as lhs_quant_pack_qsi8d32p4x8sb_f32_neon.
- New SME2 micro-kernels:
  - Matrix multiplication (1xN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F16 output, optimized for FEAT_SME2.
  - Matrix multiplication (MxN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F16 output, optimized for FEAT_SME2.
  - Matrix multiplication (MxN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F32 output, optimized for FEAT_SME2.
  - Matrix multiplication (1xN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F32 output, optimized for FEAT_SME2.

## v1.12.0

- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN) Micro-kernels of QAI8DX LHS and QSI4CX RHS with BF16 output, optimized for FEAT_I8MM.
  - Matrix multiplication (1xN) Micro-kernels of QAI8DX LHS and QSI4CX RHS with BF16 output, optimized for FEAT_DotProd.
  - Matrix multiplication (MxN) Micro-kernels of QAI8DX LHS and QSI4C32 RHS with BF16 output, optimized for FEAT_I8MM.
  - Matrix multiplication (1xN) Micro-kernels of QAI8DX LHS and QSI4C32 RHS with BF16 output, optimized for FEAT_DotProd.
- New SME micro-kernels:
  - Matrix multiplication (1xN) of F32 LHS and RHS with F32 output, using instructions compatible with FEAT_SME.
  - Matrix multiplication (1xN) of F16 LHS and RHS with F16 output, using instructions compatible with FEAT_SME.
- Convert SME transposed RHS packing micro-kernels to pure assembly:
  - kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme
  - kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme
- Include more micro-kernels in MSVC build:
  - kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla
  - kai_lhs_quant_pack_qsi8d32p_f32_neon
  - kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon
  - kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon
  - kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon
  - kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon
- Fixes
  - Update kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa to improve accuracy
  - Convert common SME/SME2 code into assembly file kai_common_sme_asm.S
- Documentation
  - Added ONNX Runtime MLAS library integration example.

## v1.11.0

- New Advanced SIMD micro-kernels:
  - Optimized version of kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0 micro-kernel for block depth of 4 bytes (`kai_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon`)
- Improve performance of `kai_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon`

## v1.10.0

- Convert SME and SME2 imatmul micro-kernels to use pure assembly, and add MSVC support. Affects:
  - kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa
  - kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa
  - kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa
  - kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme
  - kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme
  - kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme
  - kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme
  - kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme
  - kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme
- Convert SME and SME2 matmul micro-kernels to pure assembly, and add MSVC support. Affects:
  - kai_lhs_pack_f32p2vlx1_f32_sme
  - kai_lhs_pack_x16p2vlx2_x16_sme
  - kai_lhs_pack_x8p2vlx4_x8_sme
  - kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot
  - kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa
  - kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla
  - kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla
  - kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa
  - kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot
  - kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa
  - kai_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme
  - kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme
  - kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme
  - kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme
- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F32 output, optimized for FEAT_DotProd.
  - Matrix multiplication (MxN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F16 output, optimized for FEAT_DotProd.
  - Matrix multiplication (1xN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F32 output, optimized for FEAT_DotProd.
  - Matrix multiplication (1xN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F16 output, optimized for FEAT_DotProd.
  - Optimized version of kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0 micro-kernel for block depth of 8 bytes (`kai_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon`)
- New SME micro-kernels:
  - Added GEMM F16 and F32 micro-kernels using SME1 MOPA instruction, block size 2VLx2VL.
- Added Convolution example using SME2 Indirect Matmul micro-kernels
- Fixes:
  - Fix issue where kai_get_m_step() returns the incorrect value for micro-kernels
    - matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla
    - matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla
  - Fix issue with negative values handling in kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon

## v1.9.0

- Extend support for signed 4-bit integer inputs in `kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon`.
- Add imatmul documentation
- Better out-of-bounds access detection support in testing framework.
- New SME2 micro-kernels:
  - Matrix multiplication (1xN) of QAI8DX LHS and QSI8CX RHS to produce F32 output.
  - Matrix multiplication (MxN) of QAI8DX LHS and QSI8CX RHS to produce F32 output.
- Fixes:
  - Address segmentation faults in benchmarking tool.
  - Fix clamping issues for FP16 and BF16 in testing framework.

## v1.8.0

- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN) Micro-kernels of QAI8DX LHS and QSI8CX RHS with F16 output, optimized for FEAT_I8MM and FEAT_DotProd.
  - Matrix multiplication (1xN) Micro-kernels of QAI8DX LHS and QSI8CX RHS with F16 output, optimized for FEAT_DotProd.
- New SME micro-kernels:
  - Indirect matrix multiplication (MxN) of F16 input and output.
    - Packing micro-kernels for LHS and RHS
  - Indirect matrix multiplication (MxN) of F32 input and output.
    - Packing micro-kernels for LHS and RHS
- New SME2 micro-kernels:
  - Indirect matrix multiplication (MxN) of F16 input and output.
    - Matrix multiplication of packed indirect LHS and packed RHS
  - Indirect matrix multiplication (MxN) of F32 input and output.
    - Matrix multiplication of packed indirect LHS and packed RHS
- Disable link time optimization for micro-kernel library

## v1.7.0

- New SME micro-kernels:
  - Indirect matrix multiplication (MxN) of QAI8 input and output.
    - Packing micro-kernels for LHS and RHS
- New SME2 micro-kernels:
  - Indirect matrix multiplication (MxN) of QAI8 input and output.
    - Matrix multiplication of packed indirect LHS and packed RHS
- New Advanced SIMD micro-kernels:
  - Matrix multiplication (MxN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F32 output, optimized for FEAT_I8MM.
  - Matrix multiplication (1xN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F32 output, optimized for FEAT_DotProd.
  - Matrix multiplication (MxN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F16 output, optimized for FEAT_I8MM.
  - Matrix multiplication (1xN) Micro-kernels of QSI8D32 LHS and QAI4C32 RHS with F16 output, optimized for FEAT_DotProd.
  - Matrix multiplication (MxN) Micro-kernels of QAI8DX LHS and QSI4CX RHS with F16 output, optimized for FEAT_I8MM and FEAT_DotProd.
  - Matrix multiplication (1xN) Micro-kernels of QAI8DX LHS and QSI4CX RHS with F16 output, optimized for FEAT_DotProd.

## v1.6.0

- Add CMake installation and `find_package()` support.
- Optimize RHS packing qsu4c32s16s0->qsi4c32pscalef16
- Fixes:
  - Fix issue where the following micro-kernels ignored clamping parameters:
    - kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla
    - kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot
    - kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla

## v1.5.0

- Extend benchmark tool to support all matrix multiplication micro-kernels.
- New Advanced SIMD micro-kernels:
  - New 4x8 block size variant of matrix multiplication of QAI8DXP LHS and QSI4C32P RHS with F32 output.
    - Optimizations for FEAT_I8MM.
- Fixes:
  - Remove "-Weffc++" from build flags
  - Fix out-of-bound read from LHS packed matrix in `kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa`.

## v1.4.0

- New Advanced SIMD micro-kernels:
  - New 4x8 block size variant of matrix multiplication of QAI8DXP LHS and QSI4C32P RHS with F32 output.
    - Optimizations for FEAT_DotProd.
  - New 1x8 block size variant of matrix multiplication of QAI8DXP LHS and QSI4C32P RHS with F32 output.
    - Optimizations for FEAT_DotProd.
  - New 1x8 block size variant of matrix multiplication of QAI8DXP 1x8 LHS and QSI4C32P 8x8 RHS with F32 output.
    - Optimizations for FEAT_DotProd.
- New SME2 micro-kernels:
  - Matrix multiplication (1xN) of QAI8 LHS and QSI8 RHS to produce QAI8 output.
- Updated an example to demonstrate integration using CMake
- Build tests for matmul_clamp_f32_qai8dxp_qsi4c32p with MSVC
- Fixes:
  - Fix the RHS packing micro-kernel kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon to handle null bias.
  - Implement matmul portion testing in int8 unit tests
  - Use absolute path as header search path in CMakeLists.txt

## v1.3.0

- Update FP16 example to use NHWC input
- Fixes:
  - Fix build error on MSVC for some kai_matmul_clamp_f32_qai8dxp_qsi4c32p micro-kernels
  - Fix compilation warnings detected by `-Wcast-qual -Wmissing-prototypes -Wstrict-prototypes -Woverlength-strings` compiler options.
    - Support compiling the project with the above compilation options enabled.
  - Remove `-Werror` from default build flags as to not cause integration problems
  - Expose the rhs_packed_stride in the header file
  - Fix validation error when n > nr in kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa

## v1.2.0

- New SME micro-kernels:
  - Matrix multiplication (MxN) for BF16 inputs with F32 output.
- Add MSVC support for test framework
- Fixes:
  - Fix several CPU feature check issues affecting test framework
  - Fix the LHS/RHS packed offset calculation in matmul get_offset methods

## v1.1.0

- New Advanced SIMD micro-kernels:
  - New 16x4 and 1x4 block size variants of matrix multiplication of QAI8DXP LHS and QSI4C32P RHS with F32 output.
    - Optimizations for FEAT_DotProd.
- New SME micro-kernels:
  - Matrix multiplication (MxN and 1xN) of QAI8DXP LHS and QSI4CXP RHS to produce F32 output.
- Packing micro-kernels for QSI4CXP RHS to work with the SME matrix multiplication (MxN and 1xN) micro-kernels.
- Fixes:
  - Fix out-of-bounds read in `kai_lhs_quant_pack_qai8dxp_f32` packing micro-kernel.
  - Unit test improvements.

## v1.0.0

- Breaking changes:
  - Change the F16 matrix multiplication function signature to use single-precision floating-point for the clamp values.
- Optimizations:
  - Optimize QAI8DXP LHS quant and pack micro-kernel using Arm® Neon™
  - Optimize the NxK scalar RHS packing micro-kernel for QSU4C32 with BF16 quantization scales
- Add initial Microsoft® Visual C++™ build support
- API for querying library version
- Fixes:
  - Update QSI8CX tests
  - Asserts will call `abort()` instead of `exit(...)`
  - Changed invalid assertion in F16 micro-kernel
  - Build system improvements
  - Unit test improvements

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
  - SME F32 packing micro-kernel for transposed RHS matrix.
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
- CMake and Bazel build system for micro-kernels.
