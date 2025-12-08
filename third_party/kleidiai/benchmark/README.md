<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# KleidiAI Benchmark Tool

KleidiAI provides a single benchmarking binary that runs multiple variants via subcommands:

- `kleidiai_benchmark matmul` for standard matrix multiplication (matmul)
- `kleidiai_benchmark imatmul` for indirect matrix multiplication (imatmul, chunked K)

The tool supports flexible argument parsing and Benchmark Framework options.
If no operator is specified, `matmul` will be used by default.

## Building

From the KleidiAI root directory:

### Build instructions

```
mkdir -p build && cd build
cmake -DKLEIDIAI_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release ../
make -j
```

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

### Quick Examples

Run both matmul and imatmul with example dimensions:

```sh
./kleidiai_benchmark matmul  -m 32 -n 32 -k 32
./kleidiai_benchmark imatmul -m 32 -n 32 -c 4 -l 8
```

### Matmul Benchmark

The dimensions of the LHS- and RHS-matrices needs to be specified with the `-m`, `-n` and `-k` options.
The shape of the LHS-matrix is MxK, and the shape of the RHS-matrix is KxN.
Run the matmul benchmark with matrix dimensions:

```
./kleidiai_benchmark matmul -m <M> -n <N> -k <K>
```

Example:

```
$ ./kleidiai_benchmark matmul -m 13 -n 17 -k 18
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

### iMatmul Benchmark (chunked K)

Run the imatmul benchmark with matrix dimensions and chunking:

```
./kleidiai_benchmark imatmul -m <M> -n <N> -c <CHUNK_COUNT> -l <CHUNK_LENGTH>
```

Where:

- `-m`, `-n` are matrix dimensions (LHS: MxK, RHS: KxN)
- `-c` is the number of K chunks
- `-l` is the length of each K chunk

Example:

```
./kleidiai_benchmark imatmul -m 32 -n 32 -c 4 -l 16
Run on (12 X 24 MHz CPU s)
Load Average: 4.59, 3.95, 3.95
---------------------------------------------------------------------------------------------------------
Benchmark                                                               Time             CPU   Iterations
---------------------------------------------------------------------------------------------------------
imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa               123 ns          123 ns      1234567
imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa               123 ns          123 ns      1234567
imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa              123 ns          123 ns      1234567
imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa               123 ns          123 ns      1234567
imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa         123 ns          123 ns      1234567
imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa        123 ns          123 ns      1234567
```

### Filtering

Benchmarks can be filtered using the --benchmark_filter option, which accepts a regex. For example, to only run the sme2 microkernels:
(Note: The measurement results are placeholders)

```
./kleidiai_benchmark matmul  --benchmark_filter=sme2 -m 13 -n 17 -k 18
./kleidiai_benchmark imatmul --benchmark_filter=sme2 -m 13 -n 17 -c 1 -l 18
Run on (8 X 1800 MHz CPU s)
Load Average: 10.09, 10.13, 10.09
-----------------------------------------------------------------------------------------------------
Benchmark                                                           Time             CPU   Iterations
-----------------------------------------------------------------------------------------------------
matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot        123 ns          123 ns      1234567
imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa           123 ns          123 ns      1234567
```

### Listing Available Benchmarks

To list all available benchmarks:

```
./kleidiai_benchmark  --benchmark_list_tests

```

Specify the micro-kernel operator to list all the benchmarks of a certain type.

```
./kleidiai_benchmark matmul  --benchmark_list_tests
./kleidiai_benchmark imatmul --benchmark_list_tests
```

### Notes

This application uses [Google Benchmark](https://github.com/google/benchmark), so all options that Google Benchmark provides can be used.
To list the options provided use the `--help` flag or refer to the [user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md).
