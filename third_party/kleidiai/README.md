<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

<h1><b>KleidiAI</b></h1>

KleidiAI is an open-source library that provides optimized performance-critical routines, also known as <strong>micro-kernels</strong>, for artificial intelligence (AI) workloads tailored for Arm® CPUs.

These routines are tuned to exploit the capabilities of specific Arm® hardware architectures, aiming to maximize performance.

The KleidiAI library has been designed for ease of adoption into C or C++ machine learning (ML) and AI frameworks. Specifically, developers looking to incorporate specific micro-kernels into their projects can only include the corresponding <strong>.c</strong> and <strong>.h</strong> files associated with those micro-kernels and a common header file.

> ⚠️ The project has not hit a 1.x.y release yet and it is essential to note that API modifications, including function name changes, and feature enhancements may occur without advance notice.

<h1> Who is this library for? </h1>

KleidiAI is a library for AI/ML framework developers interested in accelerating the computation on Arm® CPUs.

<h1> What is a micro-kernel? </h1>

A micro-kernel, or <strong>ukernel</strong>, can be defined as a near-minimum amount of software to accelerate a given ML operator with high performance.

For example, consider the convolution 2d operator performed through the Winograd algorithm. In this case, the computation requires the following four operations:

- Winograd input transform
- Winograd filter transform
- Matrix multiplication
- Winograd output transform

Each of the preceding operations is a micro-kernel.

<em>However, why the preceding operations are not called kernels or functions instead?</em>

<b>Because the micro-kernels are designed to give the flexibility to process also a portion of the output tensor</b>, which is the reason why we call it micro-kernel.

> ℹ️ The API of the micro-kernel is intended to provide the flexibility to dispatch the operation among different working threads or process only a section of the output tensor. Therefore, the caller can control what to process and how.

A micro-kernel exists for different Arm® architectures, technologies, and computational parameters (for example, different output tile sizes). These implementations are called <strong>micro-kernel variants</strong>. All micro-kernel variants of the same micro-kernel type perform the same operation and return the same output result.

<h1> Key features </h1>

Some of the key features of KleidiAI are the following:

- No dependencies on external libraries

- No dynamic memory allocation

- No memory management​

- No scheduling

- Stateless, stable, and consistent API​

- Performance-critical compute-bound and memory-bound micro-kernels

- Specialized micro-kernels utilizing different Arm® CPU architectural features (for example, <strong>FEAT_DotProd</strong> and <strong>FEAT_I8MM</strong>)

- Specialized micro-kernels for different fusion patterns

- Micro-kernel as a standalone library, consisting of only a <strong>.c</strong> and <strong>.h</strong> files

> ℹ️ The micro-kernel API is designed to be as generic as possible for integration into third-party runtimes.

<h1> Current supported Arm® CPUs technologies and features </h1>

<strong>Arm® Neon™</strong>

- <strong>FEAT_DotProd</strong> is optional in Armv8.2-A and mandatory in Armv8.4-A
- <strong>FEAT_I8MM</strong> is optional in Armv8.2-A and mandatory in Armv8.6-A

<h1> Filename convention </h1>

The `kai/ukernels` directory is the home for all micro-kernels. The micro-kernels are grouped in separate directories based on the performed operation. For example, all the matrix-multiplication micro-kernels are held in the `matmul/` operator directory.

Inside the operator directory, you can find:

- *The common micro-kernels*, which are helper micro-kernels necessary for the correct functioning of the main ones. For example, some of these may be required for packing the input tensors and held in the `pack` subdirectory.
- *The micro-kernels* files, which are held in separate sub-directories.

The name of the micro-kernel folder provides the description of the operation performed and the data type of the destination and source tensors. The general syntax for the micro-kernel folder is as follows:

`<op>_<dst-data-type>_<src0-data-type>_<src1-data-type>_...`

All <strong>.c</strong> and <strong>.h</strong> pair files in that folder are micro-kernel variants. The variants are differentiated by specifying the computational paramaters (for example, the block size), the Arm® technology (for example, Arm® Neon™), and Arm® architecture feature exploited (for example, <strong>FEAT_DotProd</strong>). The general syntax for the micro-kernel variant is as follows:

`kai_<micro-kernel-folder>_<compute-params>_<technology>_<arch-feature>.c/.h`

> ℹ️ These files, only depend on the `kai_common.h` file.

All functions defined in the <strong>.h</strong> header file of the micro-kernel variant has the following syntax:

`kai_<op>_<micro-kernel-variant-filename>.c/.h`

<h1> Data types </h1>

Some of the data types currently supported with the KleidiAI library are the following:

| Data type                                                                                                           | Abbreviation | Notes |
|---------------------------------------------------------------------------------------------------------------------| ----------- | ----------- |
| Floating-point 32-bit                                                                                               | <b>f32</b> | |
| Floating-point 16-bit                                                                                               | <b>f16</b> | |
| Brain Floating-point 16-bit                                                                                               | <b>bf16</b> | |
| Quantized (q) Symmetric (s) Signed (i) 4-bit (4) Per-Channel (cx) quantization parameters                           | <b>qsi4cx</b> | An <b>fp32</b> multiplier shared among all values of the same channel. `x` denotes the entirety of the channel |
| Quantized (q) Asymmetric (a) Signed (i) 8-bit (8) Per-Dimension (dx) (for example, Per-Row) quantization parameters | <b>qai8dx</b> | An <b>fp32</b> multiplier and a <b>int32</b> zero offset shared among all values of the same dimension. |

> ℹ️ In some cases, we may append the letter `p` to the data type to specify that the tensor is expected to be <strong>packed</strong>. A packed tensor is a tensor that has been rearranged in our preferred data layout from the original data layout to improve the performance of the micro-kernel. In addition to the letter `p`, we may append other alphanumerical values to specify the attributes of the data packing (for example, the block packing size or the data type of for the additional packed arguments).

<h1> Supported micro-kernels </h1>

<table class="fixed" style="width:100%">
<tr>
    <th style="width:30%">Micro-kernel</th>
    <th style="width:10%">Abbreviation</th>
    <th style="width:20%">Data type</th>
    <th style="width:10%">Reference framework</th>
    <th style="width:30%">Notes</th>
</tr>
<tr>
    <td>Matrix-multiplication with LHS packed and RHS packed matrices</td>
    <td style="width:10%">matmul_clamp_f32_qai8dxp_qsi4cxp</td>
    <td style="width:20%">
        <b>LHS</b>: qai8dxp <br>
        <b>RHS</b>: qsi4cxp <br>
        <b>DST</b>: f32 <br>
    </td>
    <td>
        TensorFlow Lite <br>
    </td>
    <td>
        The packing function for the RHS matrix is available in the `kai_rhs_pack_nxk_qsi4cxp_qsi4cxs1s0.c/.h` files. <br>
        Since the RHS matrix often contains constant values, we recommend packing the RHS matrix only once and freeing the content of the original RHS matrix. <br>
    </td>
</tr>
<tr>
    <td>Matrix-multiplication with RHS packed</td>
    <td style="width:10%">matmul_clamp_f16_f16_f16p</td>
    <td style="width:20%">
        <b>LHS</b>: f16 <br>
        <b>RHS</b>: f16p <br>
        <b>DST</b>: f16 <br>
    </td>
    <td>
        TensorFlow Lite <br>
    </td>
    <td>
        The packing function for the RHS matrix is available in the `kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c/.h` files. <br>
        Since the RHS matrix often contains constant values, we recommend packing the RHS matrix only once and freeing the content of the original RHS matrix. <br>
    </td>
</tr>
<tr>
    <td>Matrix-multiplication with RHS packed</td>
    <td style="width:10%">matmul_clamp_f32_f32_f32p</td>
    <td style="width:20%">
        <b>DST</b>: f32 <br>
        <b>LHS</b>: f32 <br>
        <b>RHS</b>: f32p <br>
    </td>
    <td>
        TensorFlow Lite <br>
    </td>
    <td>
        The packing function for the RHS matrix is listed in the header file of the GEMM micro kernel. <br>
    </td>
</tr>
<tr>
    <td>Dynamic quantization and LHS matrix packing</td>
    <td>kai_lhs_quant_pack_qai8dxp_f32</td>
    <td>
        <b>SRC</b>: f32 <br>
        <b>DST</b>: qai8cx <br>
    </td>
    <td>
        TensorFlow Lite <br>
    </td>
    <td>
        <br>
    </td>
</tr>
<tr>
    <td>Matrix-multiplication with LHS packed and RHS packed matrices</td>
    <td style="width:10%">matmul_clamp_f32_bf16p_bf16p</td>
    <td style="width:20%">
        <b>LHS</b>: bf16p <br>
        <b>RHS</b>: bf16p <br>
        <b>DST</b>: f32 <br>
    </td>
    <td>
    </td>
    <td>
        The packing function for the RHS and Lhs matrices is listed in the header file of the GEMM micro kernel.  <br>
    </td>
</tr>
</table>

<h1> How to build </h1>

<h2> Prerequisites </h2>

KleidiAI requires the following dependencies, obtainable via your preferred package manager, to be installed and available on your system to be able to build the project.

- `build-essential`
- `cmake >= 3.18`

In addition, you may choose to use the following toolchains:

- (Optional) `Arm GNU toolchain` available to download from the [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) website.
- (Optional) `Android NDK` available to download from the [Android Developer](https://developer.android.com/ndk/downloads/index.html) website.

<h2> Compile natively on an Arm®-based system </h2>

You can quickly compile KleidiAI on your system with an Arm® processor by using the following commands:

```shell
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build/
cmake --build ./build
```

<h2> Cross-compile to Android™ </h2>

Cross-compiling for Android systems requires the Android NDK toolset. The downloaded NDK contains the CMake toolchain file necessary for cross-compiling the project and must be provided to CMake with the `-DCMAKE_TOOLCHAIN_FILE` option.

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -S . -B build/
cmake --build ./build
```

<h2> Cross-compile to Linux® </h2>

The Arm GNU toolchain can be used to cross-compile to a Linux system with an Arm® processor like a Raspberry Pi from an x86_64 Linux host machine. Ensure the toolchain is available on your PATH and provide to CMake the Arm GNU Toolchain CMakefile found in `cmake/toolchains` directory with the `-DCMAKE_TOOLCHAIN_FILE` option.

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-none-linux-gnu.toolchain.cmake -S . -B build/
cmake --build ./build
```

<h1> Release Cadence </h1>

The release is a tagged version of the source code. We intend to make weekly releases.

<h1> Support </h1>

Please raise a [GitLab Issue](https://gitlab.arm.com/kleidi/kleidiai/-/issues/new) for technical support.

<h1> Frequently Asked Questions (FAQ) </h1>

<h2> What is the difference between the Compute Library for the Arm® Architecture (ACL) and KleidiAI? </h2>

This question will pop up naturally if you are familiar with the **[ACL](https://github.com/ARM-software/ComputeLibrary)**.

<em>ACL and KleidiAI differ with respect to the integration point into the AI/ML framework</em>.

ACL provides a complete suite of ML operators for Arm® CPUs and Arm Mali™ GPUs. It also provides a runtime with memory management, thread management, fusion capabilities, etc.

Therefore, <strong>ACL is a library suitable for frameworks that need to delegate the model inference computation entirely</strong>.

KleidiAI offers performance-critical operators for ML, like matrix multiplication, pooling, depthwise convolution, and so on. As such, <strong>KleidiAI is designed for frameworks where the runtime, memory manager, thread management, and fusion mechanisms are already available</strong>.

<h2> Can the micro-kernels be multi-threaded? </h2>

<strong>Yes, they can</strong>. The micro-kernel can be dispatched among different threads using the thread management available in the target AI/ML framework.

<em>The micro-kernel does not use any internal threading mechanism</em>. However, the micro-kernel's API is designed to allow the computation to be carried out only on specific areas of the output tensor. Therefore, this mechanism is sufficient to split the workload on parallel threads. More information on dispatching the micro-kernels among different threads will be available soon.

<h1> License </h1>

[Apache-2.0](LICENSES/Apache-2.0.txt).
