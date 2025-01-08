<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# KleidiAI benchmark tool

## Building

From the kleidiai-root:

### Linux®-target

```
$ mkdir -p build && cd build
$ cmake -DCMAKE_C_COMPILER=/path/to/aarch64-none-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/path/to/aarch64-none-linux-gnu-g++ -DKLEIDIAI_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release ../
```

### Android™-target

```
$ mkdir -p build && cd build
$ cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=30 -DKLEIDIAI_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release ../
```

## Usage

The dimensions of the LHS- and RHS-matrices needs to be specified with the `-m`, `-n` and `-k` options.
The shape of the LHS-matrix is MxK, and the shape of the RHS-matrix is KxN.

```
$ ./kleidiai_benchmark -m 13 -n 17 -k 18
Run on (8 X 1800 MHz CPU s)
Load Average: 10.01, 10.06, 10.06
-----------------------------------------------------------------------------------------------------
Benchmark                                                           Time             CPU   Iterations
-----------------------------------------------------------------------------------------------------
matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod        123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod        123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm           123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm           123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm           123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm           123 ns          123 ns      1234567
```

### Filtering

Testcases can be filtered using the `--benchmark_filter` accepts a regex. To run only the dotprod-testcases:
(Note: The measurement results are placeholders)

```
$ kleidiai_benchmark --benchmark_filter=dotprod -m 13 -n 17 -k 18
Run on (8 X 1800 MHz CPU s)
Load Average: 10.09, 10.13, 10.09
-----------------------------------------------------------------------------------------------------
Benchmark                                                           Time             CPU   Iterations
-----------------------------------------------------------------------------------------------------
matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod        123 ns          123 ns      1234567
matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod        123 ns          123 ns      1234567
```

This application uses [Google Benchmark](https://github.com/google/benchmark), so all options that Google Benchmark provides can be used.
To list the options provided use the `--help` flag or refer to the [user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md).
