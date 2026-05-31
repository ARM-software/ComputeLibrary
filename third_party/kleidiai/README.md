<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# KleidiAI

KleidiAI is an open-source library that provides optimized performance-critical routines, also known as **micro-kernels**, for artificial intelligence (AI) workloads tailored for Arm® CPUs.

These routines are tuned to exploit the capabilities of specific Arm® hardware architectures, aiming to maximize performance.

The KleidiAI library has been designed for ease of adoption into C or C++ machine learning (ML) and AI frameworks. Specifically, developers looking to incorporate specific micro-kernels into their projects can only include the corresponding **.c** and **.h** files associated with those micro-kernels and a common header file.

## Who is this library for?

KleidiAI is a library for AI/ML framework developers interested in accelerating the computation on Arm® CPUs.

## What is a micro-kernel?

A micro-kernel, or **ukernel**, can be defined as a near-minimum amount of software to accelerate a given ML operator with high performance.

Following are examples of a micro-kernel

- Function to perform [packing](kai/ukernels/matmul/pack/README.md)
- Function to perform matrix multiplication

*However, why are the preceding operations not called kernels or functions instead?*

**This is because the micro-kernels are designed to give the flexibility to process also a portion of the output tensor.**

> ℹ️ The API of the micro-kernel is intended to provide the flexibility to dispatch the operation among different working threads or process only a section of the output tensor. Therefore, the caller can control what to process and how.

A micro-kernel exists for different Arm® architectures, technologies, and computational parameters (for example, different output tile sizes). These implementations are called **micro-kernel variants**. All micro-kernel variants of the same micro-kernel type perform the same operation and return the same output result.

## Key features

Some of the key features of KleidiAI are the following:

- No dependencies on external libraries

- No dynamic memory allocation

- No memory management​

- No scheduling

- Stateless, stable, and consistent API​

- Performance-critical compute-bound and memory-bound micro-kernels

- Specialized micro-kernels utilizing different Arm® CPU architectural features (for example, **FEAT_DotProd** and **FEAT_I8MM**)

- Specialized micro-kernels for different fusion patterns

- Micro-kernel as a standalone library, consisting of only a **.c** and **.h** files

> ℹ️ The micro-kernel API is designed to be as generic as possible for integration into third-party runtimes.

## Supported instructions and extensions

- Advanced SIMD instructions
- Scalable Vector Extension (SVE)
- Scalable Matrix Extension (SME)
- Scalable Matrix Extension 2 (SME2)

The SME and SME2 micro-kernels require compiler support to generate SME ABI-compliant code.
You can still use the micro-kernels without compiler support, but not within a call chain that already uses ZA register.
At the moment this is not automatically detected, and you need to build with `KLEIDIAI_INTERNAL_EXTRA_ARCH=+sme` to
enable this support.

## Filename convention

The `kai/ukernels` directory is the home for all micro-kernels. The micro-kernels are grouped in separate directories based on the performed operation. For example, all the matrix-multiplication micro-kernels are held in the `matmul/` operator directory.

Inside the operator directory, you can find:

- *The common micro-kernels*, which are helper micro-kernels necessary for the correct functioning of the main ones. For example, some of these may be required for packing the input tensors and held in the `pack` subdirectory.
- *The micro-kernels* files, which are held in separate sub-directories.

The name of the micro-kernel folder provides the description of the operation performed and the data type of the destination and source tensors. The general syntax for the micro-kernel folder is as follows:

`<op>_<dst-data-type>_<src0-data-type>_<src1-data-type>_...`

All **.c** and **.h** pair files in that folder are micro-kernel variants. The variants are differentiated by specifying the computational paramaters (for example, the block size), the Arm® technology (for example, Arm® Neon™), and Arm® architecture feature exploited (for example, **FEAT_DotProd**). The general syntax for the micro-kernel variant is as follows:

`kai_<micro-kernel-folder>_<compute-params>_<technology>_<arch-feature>.c/.h`

> ℹ️ These files, only depend on the `kai_common.h` file.

All functions defined in the **.h** header file of the micro-kernel variant has the following syntax:

`kai_<op>_<micro-kernel-variant-filename>.c/.h`

## Supported micro-kernels

For a list of supported micro-kernels refer to the [source](/kai/ukernels/) directory. The micro-kernels are grouped in separate directories based on the performed operation.
For example, all the matrix-multiplication micro-kernels are held in the `matmul/` subdirectory. In there, the micro-kernels are grouped into directories whose name syntax describes the micro-kernel from a data type point of view of inputs and outputs.

## How to build

### Prerequisites

KleidiAI requires the following dependencies, obtainable via your preferred package manager, to be installed and available on your system to be able to build the project.

- `build-essential`
- `cmake >= 3.18`

In addition, you may choose to use the following toolchains:

- (Optional) `Arm GNU toolchain` available to download from the [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) website.
- (Optional) `Android NDK` available to download from the [Android Developer](https://developer.android.com/ndk/downloads/index.html) website.

### Compile natively on an Arm®-based system

You can quickly compile KleidiAI on your system with an Arm® processor by using the following commands:

```shell
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build/
cmake --build ./build
```

### Cross-compile to Android™

Cross-compiling for Android systems requires the Android NDK toolset. The downloaded NDK contains the CMake toolchain file necessary for cross-compiling the project and must be provided to CMake with the `-DCMAKE_TOOLCHAIN_FILE` option.

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -S . -B build/
cmake --build ./build
```

### Cross-compile to Linux®

The Arm GNU toolchain can be used to cross-compile to a Linux system with an Arm® processor like a Raspberry Pi from an x86_64 Linux host machine. Ensure the toolchain is available on your PATH and provide to CMake the Arm GNU Toolchain CMakefile found in `cmake/toolchains` directory with the `-DCMAKE_TOOLCHAIN_FILE` option.

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-none-linux-gnu.toolchain.cmake -S . -B build/
cmake --build ./build
```

## Release

### Cadence

Two releases will be done per month. All releases can be found in the [release](https://gitlab.arm.com/kleidi/kleidiai/-/releases) section.

### Version

The release version conforms to Semantic Versioning.

> ⚠️ Please note that API modifications, including function name changes, and feature enhancements may occur without advance notice.

## Support

Please raise a [GitLab Issue](https://gitlab.arm.com/kleidi/kleidiai/-/issues/new) for technical support.

## Frequently Asked Questions (FAQ)

### Is x86 supported?

No, KleidiAI is a library of micro-kernels optimized for the Arm® architecture. The micro-kernels use specific Arm® instruction set architecture features which prevent compilation for other architectures like x86. Note that this applies to all micro-kernels including the packing and quantization micro-kernels.

### What ML operators are supported?

KleidiAI does not provide traditional ML operators; it implements a set of optimized micro-kernels which can be used as building blocks to implement ML operators. Please refer to the sections [What is a micro-kernel](#what-is-a-micro-kernel), [Supported micro-kernels](#supported-micro-kernels) and the [docs](/docs/README.md) for more information.

### What CPUs does KleidiAI support?

Any CPU that implements a set of the Arm® architecture extensions listed in section [Supported instructions and extensions](#supported-instructions-and-extensions) is supported by KleidiAI. However, every micro-kernel does not exist in versions for all architecture extensions. Each micro-kernel advertises the required architecture extensions through the micro-kernel name. See [Micro-kernel naming](kai/ukernels/matmul/README.md#micro-kernel-naming) for further details.

Most operating systems provide a mechanism to list the architecture extensions supported by the runtime environment, e.g. `cat /proc/cpuinfo` on Linux.

### What is the difference between the Compute Library for the Arm® Architecture (ACL) and KleidiAI?

This question will pop up naturally if you are familiar with the **[ACL](https://github.com/ARM-software/ComputeLibrary)**.

> ACL and KleidiAI differ with respect to the integration point into the AI/ML framework.

ACL provides a complete suite of ML operators for Arm® CPUs and Arm Mali™ GPUs. It also provides a runtime with memory management, thread management, fusion capabilities, etc.

Therefore, **ACL is a library suitable for frameworks that need to delegate the model inference computation entirely**.

KleidiAI offers optimized micro-kernels such as matrix multiplication and depthwise convolution which are used as building blocks in the implementation of performance-critical ML operators. As such, **KleidiAI is designed for frameworks where the runtime, memory manager, thread management, and fusion mechanisms are already available**.

### Can the micro-kernels be multi-threaded?

**Yes, they can**. The micro-kernel can be dispatched among multiple threads using the thread management available in the target AI/ML framework.

*The micro-kernel does not use any internal threading mechanism*. However, the micro-kernel's API is designed to allow the computation to be carried out only on specific areas of the output tensor. Therefore, this mechanism is sufficient to split the workload on parallel threads.

Please refer to [examples/matmul_clamp_f32_qsi8d32p_qsi4c32p](/examples/matmul_clamp_f32_qsi8d32p_qsi4c32p) for an example on how to use micro-kernels in a multithreaded environment.

## License

KleidiAI is distributed under the software licenses in LICENSES directory.
