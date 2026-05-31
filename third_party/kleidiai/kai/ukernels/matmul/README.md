<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# About

This document contains information related to matrix-multiplication (matmul)
micro-kernels. At the moment there are two main types of micro-kernels, matrix
multiplication and indirect matrix multiplication micro-kernels. The indirect
micro-kernels are denoted _imatmul_.

# Matmul

Matmul micro-kernels operate directly on matrices stored in memory buffers, where the
buffers are normally first packed into a more efficient layout.

# Indirect Matmul

The indirect matmul micro-kernels operate on indirection buffers, matrices of pointers
to actual data.

# Naming convention

The high level view of kernel naming is the following

- LHS micro-kernels are named `kai_lhs_pack_<output>_<input>_<description>`.
- RHS micro-kernels are named `kai_rhs_pack_<orientation>_<output>_<inputs>_<description>`.
- Matmul micro-kernels are named `kai_<operation>_<output>_<LHS input>_<RHS input>_<description>`.

The output of the LHS and RHS matches the inputs of the matmul kernels. Further
details are described below.

## Micro-kernel naming

The naming of micro-kernels must follow the convention below. Unless explicitly
specified, arguments are mandatory.

`kai_<op>_<fused_ops>_<dst_info>_<input_0_info, input_1_info, ...>_<m_block x n_block>_<simd_engine>_<feature>_<instruction>_<uarch>`

| Syntax                          | Description                                                                                                                        | Example                                                                                                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| op                              | The primary operation of the micro-kernel                                                                                          | `matmul`, `imatmul `                                                                                                                                                         |
| fused_ops                       | (Optional) Information on applied fused operations, e.g., activation functions                                                     | `clamp`                                                                                                                                                                      |
| dst_info                        | Description of the destination buffer                                                                                              | See Buffer descriptors section                                                                                                                                               |
| input_0_info, input_1_info, ... | Description of input buffers to the micro-kernel                                                                                   | In `matmul` routines, the LHS precedes the  RHS                                                                                                                              |
| m_block x n_block               | Primary tile size computed by the micro-kernel.                                                                                    | `6x32` where the tile size is 6 rows by 32 columns; `2vlx2vl` where the tile size is equivalent to twice the hardware-defined vector length in the row and column dimensions |
| simd_engine                     | SIMD engine used to drive the computation                                                                                          | `neon`, `sve`, `sve2`, `sme`, `sme2`                                                                                                                                         |
| feature                         | (Optional) Further information about the Arm architecture feature used, often referred to as `FEAT_<feature>` in the specification | `dotprod`, `i8mm `                                                                                                                                                           |
| instruction                     | (Optional) Predominant SIMD instruction used in the micro-kernel                                                                   | `mla`, `mmla`, `mopa`, `dot`                                                                                                                                                 |
| uarch                           | (Optional) Microarchitecture for which the micro-kernel has been optimized for                                                     | `cortexa55` to represent the Arm® Cortex®-A55 processor                                                                                                                      |

## Block

`m_block` and `n_block` refer to the primary tile sizes - the number of rows and number of columns, respectively - computed by the micro kernel. This is encoded only in the file name. Some micro kernels can output only the primary tile size where as some can output more than one tile size.

## Buffer descriptors

Input and output buffers can be described using the following form:

| Syntax   | Description                                                                                       |
|----------|---------------------------------------------------------------------------------------------------|
| f32      | Single-precision floating-point                                                                   |
| f16      | Half-precision floating-point                                                                     |
| bf16     | Brain floating-point                                                                              |
| x        | Data type agnostic. Usually used when describing moving data around like in packing micro-kernels |
| qs       | Quantized symmetric                                                                               |
| qa       | Quantized asymmetric                                                                              |
| i        | Signed integer                                                                                    |
| u        | Unsigned integer                                                                                  |
| 4        | 4-bit quantized                                                                                   |
| 8        | 8-bit quantized                                                                                   |
| dx       | Per dimension quantized                                                                           |
| cx       | Per channel quantized                                                                             |
| c32      | Per block quantization, with block length multiple of 32                                          |
| scalef16 | Scale factors stored as floating-point 16-bit                                                     |
| p        | Indicates data is packed                                                                          |
| s16s0    | Packing order of data is interleaved                                                              |
| s1s0     | Packing order of data is sequential                                                               |
| s        | Scale factors are packed into buffer                                                              |
| b        | Bias values are packed into buffer                                                                |

Example: `qsi4cxp` which means quantized symmetric (`qs`) signed integer 4-bit
data (`i4`) with per channel quantization (`cx`) that has been packed (`p`).

### Packing description

If buffers are packed they must also include information about the packing
layout. For LHS buffer it would be `<type>p<MR>x<BD>...`, and for RHS buffer it
would be `<type>p<NR>x<BD>`. `BD`, block depth, equals `KR / SR`. The `MR`,
`NR`, `KR`, and `SR` values describe the values returned by
`kai_get_(mr|nr|kr)...`, and represent the shape of the packed blocks. `MR`,
`NR` and `KR` can be written as:

| Symbol                | example       | Description                                        |
|-----------------------|---------------|----------------------------------------------------|
| _Integer literal_     | `1`, `2`, `4` | Constant value                                     |
| `mr`/`nr`/`kr`        | `mr`          | Parametric size, given as argument to micro kernel |
| `<Integer literal>vl` | `1vl`, `4vl`  | Accumulator vector length multiple                 |

## Known naming issues

There are several micro-kernels that unfortunately use the incorrect name. For
now we don't change the name as that would break API.

| Micro-kernel                                                     | Correct name                                                     | Comment                                                             |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------|
| `imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa`        | `imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme2_mopa`       | Missing bias `b`                                                    |
| `imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa` | `imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme2_mopa` | Misplaced scaling+bias `sb`                                         |
| `lhs_pack_bf16p2vlx2_f32_sme`                                    | `lhs_pack_bf16p2vlx2_f32_sme2`                                   | Incorrectly indicating SME                                          |
| `lhs_pack_f32p2vlx1_f32_sme`                                     | `lhs_pack_x32p2vlx1_x32_sme`                                     | Legacy naming                                                       |
| `matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa`         | `kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme2_mopa`    | Missing bias `b`                                                    |
| `matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa`       | `matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2b_2vlx2vl_sme2_mopa`      | Also placed in incorrect directory (`fp32_...` should be `f32_...`) |
| `matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa`          | `matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa`        | Legacy naming                                                       |
| `matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa`  | `matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme2_mopa`  | Misplaced scaling+bias `sb`                                         |
| `rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme`                           | `rhs_pack_kxn_bf16p2vlx2b_f32_f32_sme2`                          | Incorrectly indicating SME                                          |
| `rhs_pack_kxn_f32p16vlx1b_f32_f32_sme`                           | `rhs_pack_kxn_x32p16vlx1b_x32_x32_sme`                           | Legacy naming                                                       |
| `rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme`                      | `rhs_pack_kxn_x32p2vlx1b_x32_x32_sme`                            | Legacy naming                                                       |
| `rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme`                      | `rhs_pack_nxk_x32p2vlx1b_x32_x32_sme`                            | Legacy naming                                                       |
