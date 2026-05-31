<!--
    SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Integrating KleidiAI into MLAS via `MlasGemmBatch`

This document provides detailed guidance on how to integrate KleidiAI as an external optimized backend into the ONNX Runtime MLAS (Microsoft Linear Algebra Subprograms) framework. It uses `MlasGemmBatch` as the core example. It is intended to be used as a guide to aid KleidiAI integration into other frameworks.

N.B. Input tensors/matrices may not be structured in the same way as MLAS tensors are at the level of abstraction discussed below, so please make yourself aware of the input requirements to KleidiAI function calls when integrating micro-kernels into your framework.

As of July 4th 2025, the specific examples can be seen as follows:

KleidiAI call from default function (with fallback mechanics):
https://github.com/damdoo01-arm/onnxruntime/blob/kai_sgemm_igemm_quant_gemv/onnxruntime/core/mlas/lib/sgemm.cpp
(Lines 1563-1584)

KleidiAI MlasGemmBatch implementation:
https://github.com/damdoo01-arm/onnxruntime/blob/kai_sgemm_igemm_quant_gemv/onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp
(Lines 140-344)

______________________________________________________________________

## 1. Entry Point: `KleidiAI::MlasGemmBatch` call from default `MlasGemmBatch`

The default `MlasGemmBatch` implementation acts as a gateway to dispatch to external backends (e.g., KleidiAI):

```cpp
void MLASCALL MlasGemmBatch(...) {
    thread_local bool kleidiai_attempted = false;

    if (!kleidiai_attempted &&
        GetMlasPlatform().MlasGemmBatch == &ArmKleidiAI::MlasGemmBatch) {
        kleidiai_attempted = true;
        GetMlasPlatform().MlasGemmBatch(...);
        kleidiai_attempted = false;
        return;
    }
    // Default fallback implementation continues here...
}
```

### Key Notes:

- `kleidiai_attempted` prevents recursive fallback loops.
- The check on `GetMlasPlatform().MlasGemmBatch` enables backend selection without static dispatch.

______________________________________________________________________

## 2. KleidiAI Implementation: `ArmKleidiAI::MlasGemmBatch`

### 2.1 Validation & Fallback Conditions

```cpp
if (M == 0 || N == 0 || K == 0 ||
    TransA != CblasNoTrans ||
    (TransB != CblasNoTrans && !Data[0].BIsPacked) ||
    !MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME()) {
    ::MlasGemmBatch(...); // fallback
    return;
}
```

KleidiAI only supports:

- `TransA == CblasNoTrans`
- `TransB == CblasNoTrans` or `BIsPacked == true`
- SME-capable hardware

Also includes runtime check for tile size suitability:

```cpp
if (M < m_step || N < n_step) {
    if (GetMlasPlatform().MlasGemmBatch != ArmKleidiAI::MlasGemmBatch) {
        ::MlasGemmBatch(...); // fallback
        return;
    }
}
```

______________________________________________________________________

### 2.2 Preprocessing: `beta` Scaling / Zeroing

```cpp
if (Data->beta != 1.0f) { ... }
if (Data->beta == 0.0f) { ... }
```

Handles special cases for scaling or zero-initializing `C` before matmul.

______________________________________________________________________

### 2.3 Packing Strategy

In high-performance GEMM (General Matrix Multiply) kernels, data packing is essential for performance. KleidiAI relies on explicit packing of both LHS (A) and RHS (B) matrices into cache-aligned, kernel-friendly tiles before execution. Packing improves memory access patterns, enables vectorization, and reduces cache pollution.

#### LHS Packing

All `A` matrices are packed using

```cpp
kai_run_lhs_pack_f32p2vlx1_f32_sme().
```

Characteristics:
•	Parallelized across the batch dimension via MlasTrySimpleParallel (equivalent Threading function for other frameworks should be callable at this point).
•	The packed memory layout conforms to KleidiAI’s internal micro-kernel expectations: typically mr × kr tiles (e.g., 32×32).
•	Each batch element A_i is packed into a contiguous buffer at offset batch_idx × LhsPackedStride.

```cpp
size_t LhsPackedStride = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr);
auto LhsPacked = std::make_unique_for_overwrite<std::byte[]>(LhsPackedStride * BatchSize);
```

This allocates a per-batch packing region with sufficient space for tiling.

Threaded Packing Loop:

```cpp
MlasTrySimpleParallel(ThreadPool, BatchSize, [&](ptrdiff_t batch_idx) {
    std::byte* LhsPackedPtr = LhsPackedData + batch_idx * LhsPackedStride;
    kai_run_lhs_pack_f32p2vlx1_f32_sme(..., Data[batch_idx].A, ..., LhsPackedPtr);
    KaiPackedData[batch_idx].A = reinterpret_cast<const float*>(LhsPackedPtr);
});
```

#### RHS Packing (if required)

Conditionally performed if

```cpp
Data[0].BIsPacked == false
```

i.e., the B matrix is not already pre-packed by the calling layer

RHS Packing micro-kernel:
Conditionally performed if Data\[0\].BIsPacked == false, i.e., the B matrix is not already pre-packed by the calling layer

```cpp
ArmKleidiAI::MlasGemmPackB(TransA, TransB, N, K, B, ldb, RhsPackedPtr)
```

This wraps the KleidiAI kai_run_rhs_pack_f32_sme(...) and ensures:

```
•	Alignment to nr × kr tile shape
•	Pointer-based layout suitable for direct loading into the micro-kernel
```

Buffer Allocation:

```cpp
size_t RhsPackedStride = ArmKleidiAI::MlasGemmPackBSize(...);
auto RhsPacked = std::make_unique_for_overwrite<std::byte[]>(RhsPackedStride * BatchSize);
```

Combined LHS/RHS Packing Loop:

```cpp
MlasTrySimpleParallel(ThreadPool, BatchSize * 2, [&](ptrdiff_t batch_idx) {
    if (batch_idx & 1) {
        // LHS
    } else {
        // RHS
    }
});
```

______________________________________________________________________

### 2.4 Tile Dimensioning

To efficiently execute large matrix multiplications on modern CPU architectures—especially those supporting tile-based vector extensions like Arm SME2 the workload must be divided into tiles that can be executed in parallel by multiple threads.

This process involves three core steps:

______________________________________________________________________

#### **Step 1: Establish a 3D Tiling Scheme**

Matrix multiplication over a batch of inputs can be visualized as a 3-dimensional grid of compute tiles:

```
Tiling dimensions = [BatchSize, number of M tiles, number of N tiles]
```

Where:

- `BatchSize` refers to the number of independent matrix multiplications.
- `M tiles` correspond to partitioning the rows of matrix A.
- `N tiles` correspond to partitioning the columns of matrix B.

Initial tile counts are estimated by dividing the matrix sizes by the preferred micro-kernel tile dimensions (`m_step`, `n_step`):

```cpp
tile_count_M = ceil(M / m_step);
tile_count_N = ceil(N / n_step);
```

The total number of work units becomes: `BatchSize × tile_count_M × tile_count_N`.

______________________________________________________________________

#### **Step 2: Balance Tile Count Against Available Threads**

To make full use of the thread pool:

- Estimate how many tiles are ideally needed (limited by thread count).
- Reshape the 3D tile grid to distribute the workload more evenly.

This may involve scaling the number of tiles along the M and N dimensions such that:

```cpp
adjusted_tile_count_M ≈ ceil(ideal_tile_count * tile_count_M / total_tile_count);
adjusted_tile_count_N ≈ ceil(ideal_tile_count * tile_count_N / total_tile_count);
```

This rebalancing avoids creating too many small tiles or leaving threads underutilized.

______________________________________________________________________

#### **Step 3: Derive Updated Step Sizes**

Once the updated tile counts are known, recalculate the actual tile sizes (`m_step`, `n_step`) to match:

```cpp
m_step = ceil(M / adjusted_tile_count_M);
n_step = ceil(N / adjusted_tile_count_N);
```

Finally, the number of tiles is re-derived using the new step sizes:

```cpp
tile_count_M = ceil(M / m_step);
tile_count_N = ceil(N / n_step);
```

### 2.5 Main Tile Execution Loop

This is the core loop that executes `kai_run_matmul_clamp_...()` across all 3D tile indices.

#### 2.5.1 Tile Scheduling

```cpp
MlasTrySimpleParallel(ThreadPool, dim[0] * dim[1] * dim[2], [=](ptrdiff_t tid) {
    size_t BIdx = tid / (dim[1] * dim[2]);
    size_t MIdx = (tid % (dim[1] * dim[2])) / dim[2];
    size_t NIdx = tid % dim[2];
```

Each `tid` maps to a unique tile in `[B, M, N]`.

#### 2.5.2 Input Tile Extraction

The packed matrices are stored contiguously by batch. For each tile:

- Compute offsets:

```cpp
lhs_offset = kai_get_lhs_packed_offset_...(MIdx * m_step, K);
rhs_offset = kai_get_rhs_packed_offset_...(NIdx * n_step, K);
```

- Slice from packed buffer:

```cpp
const float* ATile = reinterpret_cast<...>(KaiPackedData[BIdx].A + lhs_offset);
const void*  BTile = reinterpret_cast<...>(KaiPackedData[BIdx].B + rhs_offset);
```

#### 2.5.3 Micro-kernel Invocation

The SME2-optimized micro-kernel is called as:

```cpp
kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    TileSizeM, TileSizeN, K,
    ATile, BTile,
    temp_tile, // Output buffer
    TileSizeN * sizeof(float), sizeof(float),
    -FLT_MAX, FLT_MAX
);
```

- `temp_tile` is a thread-local scratch buffer.
- Micro-kernel writes a raw `A*B` tile result without alpha/beta.

#### 2.5.4 Writing to Output Matrix `C`

The computed tile is then written to the final `C` matrix:

- Compute the destination pointer:

```cpp
float* dst_tile = Data[BIdx].C + MIdx * m_step * ldc + NIdx * n_step;
```

- Handle 2 cases:
  - **Fast Path** (no accumulation):
    ```cpp
    if (alpha == 1.0f && beta == 0.0f && ldc == TileSizeN && tile is in bounds)
        memcpy(dst_tile, temp_tile, TileSizeM * TileSizeN * sizeof(float));
    ```
  - **General Path** (scaled accumulation):
    ```cpp
    for each (i, j) {
        dst_tile[i * ldc + j] = alpha * temp_tile[i * TileSizeN + j] + beta * dst_tile[i * ldc + j];
    }
    ```

This ensures correct handling of arbitrary GEMM expressions:

```
C = alpha * A * B + beta * C
```

______________________________________________________________________

## 3. Fallback Behavior

If any constraint isn't met (unsupported transpose, no SME, small matrix), the call falls back to the default `MlasGemmBatch` using:

```cpp
::MlasGemmBatch(...);
```

This ensures correctness even if KleidiAI can't process the workload.

______________________________________________________________________

______________________________________________________________________

## 4. Required KleidiAI Functions

- `kai_get_m_step_...`, `n_step_...`, `mr`, `kr`, `sr`
- `kai_run_lhs_pack_...`
- `kai_get_lhs_packed_offset_...`
- `kai_run_matmul_clamp_...`

These functions must be provided by KleidiAI for the SME2 micro-kernel path.

______________________________________________________________________

## 5. Platform Detection & Hooking

The backend is activated through:

```cpp
GetMlasPlatform().MlasGemmBatch = &ArmKleidiAI::MlasGemmBatch;
```

Usually set in MLAS platform initialization during runtime feature detection.

______________________________________________________________________

## 6. Summary of Integration Mechanics

| Stage               | Description                                           |
|--------------------|-------------------------------------------------------|
| Dispatch Check     | Conditional on platform struct function pointer      |
| Pre-conditions     | Matrix sizes, transpose modes, SME support           |
| Fallbacks          | Recursive call into MLAS if unsupported              |
| Data Packing       | Both LHS and RHS packed using KleidiAI routines      |
| Tile Dispatch      | Multi-threaded tile-wise matmul execution            |
| Output Writeback   | `memcpy` or loop with alpha/beta scaling             |

This pattern can be extended for other MLAS APIs (e.g., `MlasGemmPackB`, `MlasConv`) can be seen elsewhere in the onnxruntime code and use a similar override, fallback, and execution structure.
