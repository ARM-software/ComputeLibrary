<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# About

This document contains information related to matrix-multiplication (matmul) micro kernels.

# Naming

The naming of files has the convention below. Unless explicitly specified, arguments are mandatory.
`kai_matmul_<fused_ops>_<dst_info>_<lhs_info>_<rhs_info>_<mr x nr x kacc>_<technology>_<feature>_<instruction>.c`

| Syntax        | Description |Example   |
| --------------| ----------- |----------|
|fused_ops      |Optional info on applied fused operation like a clamping activation function. | clamp|
|dst_info       |Destination matrix info. Similar to lhs_info||
|lhs_info       |LHS matrix data type and and packing info. <br> f32 -  Floating-point 32-bit<br>q : Quantized<br>s :  Symmetric<br>a : Asymmetric<br>i : Signed  integer<br> u : Unsigned integer<br>4 : 4-bit Quantized <br> 8 : 8-bit Quantized <br> dx : Per dimension quantization<br>cx : Per channel quantization<br>c32 : Per block quantization, with block length multiple of 32<br> scalef16 : Scale factors are stores as floating-point 16-bit<br> p : Matrix is packed|Example 1<br> qsi4cxp :<br> qs - Quantized symmetric<br> i4 - Signed Integer 4-bit <br> cx - Per channel quantized <br> p - packed<br> Some other examples : <br> s16s0 - Packing order of data is interleaved <br> s1s0 - Packing order of data is sequential<br>fp16 - Floating-point 16-bit data type |
|rhs_info       | Similar to lhs_info||
|mr x nr x kacc |The outer loop calculates mr rows and nr columns. kacc is k-accumulations done per inner loop||
|technology     |Underlying technology. Arm® Neon™|neon|
|feature        |Arm architecture feature used|dotprod, i8mm, sme2|
|instruction    |Instruction used. This is optional|mla|
