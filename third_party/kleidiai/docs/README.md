<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# KleidiAI documentation and guides

Welcome to the KleidiAI documentation hub. Here, you will find a variety of information and step-by-step guides to help you master this library. For instance, you can explore introductory tutorials on running a micro-kernel and discover best practices for optimizing the performance of your AI framework on Arm® CPUs.

## KleidiAI directory structure overview

- `benchmark` — Google Benchmark harness plus `matmul/` registry for exercising micro-kernels and comparing variants.
- `cmake` — Toolchain definitions and helper modules used by the CMake build.
- `docker` — Container files that provision cross-compilation environments for testing and CI.
- `docs` — Authoritative how-to guides, deep dives, and integration notes (summarized below).
- `examples` — Standalone C++ samples that demonstrate end-to-end usage of specific micro-kernels.
- `kai` — Core library sources. `kai_common.h` provides shared utilities; `ukernels/` contains micro-kernel implementations grouped by type (matmul, dwconv etc).
- `test` — Reference implementations, common utilities, and GoogleTest drivers for functional validation.
- `third_party` — Third party dependencies with accompanying license files.
- `tools` — Pre-commit tooling (pre-commit hooks and Python requirements).

## Examples

The `examples` directory contains standalone C++ sample applications that demonstrate end-to-end usage of a selection of the KleidiAI micro-kernels. They demonstrate fundamental aspects of using KleidiAI, like [LHS and RHS packing](../kai/ukernels/matmul/pack/README.md) and setting up [indirection table](imatmul/README.md#packing) for IGEMM micro-kernels. Use CMake to build the examples.

| Example path | Description |
|--------------|-------------|
| `examples/conv2d_imatmul_clamp_f16_f16_f16p_sme2` | Convolution using SME2 indirect GEMM micro-kernel. This example shows how to setup the indirection table. |
| `examples/dwconv_clamp_f32_f32_f32p_planar_sme2` | Depthwise planar convolution with f32 input/output and 3x3 filter using SME2 micro-kernels. |
| `examples/matmul_clamp_f16_f16_f16p` | Matrix multiplication of two f16 matrices using a Advanced SIMD micro-kernel where only the RHS is packed. |
| `examples/matmul_clamp_f32_bf16p_bf16p` | Matrix multiplication of two half-precision brain floating-point (BF16) matrices into a f32 output matrix. Uses both GEMM and GEMV Advanced SIMD micro-kernels.|
| `examples/matmul_clamp_f32_qai8dxp_qsi4c32p` | Matrix multiplication of f32 and per block symmetric quantized int4 matrices using GEMM and GEMV Advanced SIMD micro-kernels. LHS is int8 asymmetric quantized and RHS is packed using packing micro-kernels before the matmul operation is executed. |
| `examples/matmul_clamp_f32_qai8dxp_qsi4cxp` | Matrix multiplication of f32 and per channel symmetric quantized int4 using GEMM and GEMV Advanced SIMD micro-kernels. LHS is int8 asymmetric quantized and RHS is packed before running the matmul operation. |
| `examples/matmul_clamp_f32_qsi8d32p_qsi4c32p` | Matrix multiplication of f32 and per block quantized symmetric int4 using GEMM and GEMV Advanced SIMD micro-kernels. LHS is int8 per block symmetric quantized and RHS is int4 with per block symmetric quantization f16 scale factors. This example demonstrates how to split the workload among multiple worker threads. |

## Documentation and guides

- [How to run the int4 matmul micro-kernels](matmul_qsi4cx/README.md)
- [How to run the indirect matmul micro-kernels](imatmul/README.md)
- [Matmul micro-kernels overview](../kai/ukernels/matmul/README.md)
- [Matmul packing micro-kernels description](../kai/ukernels/matmul/pack/README.md)
- [Depthwise concolution micro-kernels overview](../kai/ukernels/dwconv/README.md)
- [Integrating KleidiAI into MLAS via MlasGemmBatch](framework_integration_examples/kleidiai_mlas_integration.md)
- [Integrating KleidiAI Int4 matrix multiplication micro-kernel into llama.cpp](https://github.com/Arm-Examples/ML-examples/blob/main/kleidiai-examples/llama_cpp/0001-Use-KleidiAI-Int4-Matmul-micro-kernels-in-llama.cpp.patch)
