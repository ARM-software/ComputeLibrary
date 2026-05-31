<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# About

This document contains information related to depthwise convolution (dwconv)
micro-kernels.

# Depthwise Conv

Dw conv micro-kernels operate directly on tensors stored in memory buffers. The RHS buffer is
normally pre-packed into a more efficient data layout taking into account vector length (or with interleaved bias).

# Naming

The naming of files has the convention below. Unless explicitly specified, arguments are mandatory.
The naming convention is largely similar to the matmul micro-kernels, with a few new fields.

`kai_<op>_<fused_ops>_<dst_info>_<input_0_info>_<input_1_info>_<filter>_<stride>_<m_step x n_step>_<simd_engine>_<feature>_<instruction>.c`

| Syntax        | Description |Example   |
| --------------| ----------- |----------|
|op             |The main operation| matmul, imatmul, dwconv|
|fused_ops      |(Optional) Information on applied fused operations, e.g., activation functions| `clamp` |
|dst_info       |Description of destination buffer. See buffer descriptors. ||
| input_0_info, input_1_info, ... | Description of input buffers to the micro-kernel. In `matmul` routines, the LHS precedes the RHS. See buffer descriptors. ||
|filter         | Describes convolution filter used by micro-kernel in the format 'h x w' ||
|stride         | Stride used by convolution operation |`s1` means a stride of 1.|
|m_step x n_step  | Output block size when the micro-kernel is ran once. |`4xc` means the micro-kernel produces 4 rows of output, calculating all channel values. Therefore `xc` means the micro-kernel is planar.|
|simd_engine   | SIMD engine used to drive the computation  | `neon`, `sme`, `sme2`|
|feature        | (Optional) Further information about the Arm architecture feature used, often referred to as `FEAT_<feature>` in the specification | `dotprod`, `i8mm`|
|instruction    |Instruction used. This is optional|`mla`, `mopa`|

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

Example: `qsi4cxp` which means quantized symmetric (`qs`) signed integer 4-bit data (`i4`) with per channel quantization (`cx`) that has been packed (`p`).

Input buffer descriptors **must** also include information about how the data has been packed to more easily identify the required packing micro-kernels. In matmul routines this is done by appending `mrxkr` or `nrxkr` to the descriptor where `mr` represents the number of rows of LHS that are packed together, `nr` the number of columns of RHS that are packed together, and `kr` the number of columns of LHS or rows of RHS that are packed together.
