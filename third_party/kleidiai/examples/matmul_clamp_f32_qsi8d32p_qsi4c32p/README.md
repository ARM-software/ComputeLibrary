<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# KleidiAI Examples

## Building

From the examples/matmul_clamp_f32_qsi8d32p_qsi4c32p dir

### Linux®-target

```
$ mkdir -p build && cd build
$ cmake -DCMAKE_C_COMPILER=/path/to/aarch64-none-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/path/to/aarch64-none-linux-gnu-g++ -DCMAKE_BUILD_TYPE=Release ../
```

### Android™-target

```
$ mkdir -p build && cd build
$ cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=30  -DCMAKE_BUILD_TYPE=Release ../
```

## Usage

```
$ ./matmul_clamp_f32_qsi8d32p_qsi4c32p
```
