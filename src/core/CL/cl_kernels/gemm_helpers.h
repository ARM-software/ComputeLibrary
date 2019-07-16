/*
 * Copyright (c) 2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "activation_float_helpers.h"
#include "helpers.h"

#define LOAD_ROW_1(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                      \
    BASENAME##0 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 0 * STRIDE_Y + Z##0));

#define LOAD_ROW_2(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_1(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                      \
    BASENAME##1 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 1 * STRIDE_Y + Z##1));

#define LOAD_ROW_3(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_2(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                      \
    BASENAME##2 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 2 * STRIDE_Y + Z##2));

#define LOAD_ROW_4(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_3(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                      \
    BASENAME##3 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 3 * STRIDE_Y + Z##3));

#define LOAD_ROW_5(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_4(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                      \
    BASENAME##4 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 4 * STRIDE_Y + Z##4));

#define LOAD_ROW_6(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_5(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                      \
    BASENAME##5 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 5 * STRIDE_Y + Z##5));

#define LOAD_ROW_7(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_6(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                      \
    BASENAME##6 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 6 * STRIDE_Y + Z##6));

#define LOAD_ROW_8(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_7(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                      \
    BASENAME##7 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 7 * STRIDE_Y + Z##7));

#define LOAD_ROW_9(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_8(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                      \
    BASENAME##8 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 8 * STRIDE_Y + Z##8));

#define LOAD_ROW_10(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_9(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)      \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##9 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 9 * STRIDE_Y + Z##9));

#define LOAD_ROW_11(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_10(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##A = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 10 * STRIDE_Y + Z##A));

#define LOAD_ROW_12(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_11(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##B = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 11 * STRIDE_Y + Z##B));

#define LOAD_ROW_13(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_12(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##C = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 12 * STRIDE_Y + Z##C));

#define LOAD_ROW_14(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_13(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##D = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 13 * STRIDE_Y + Z##D));

#define LOAD_ROW_15(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_14(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##E = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 14 * STRIDE_Y + Z##E));

#define LOAD_ROW_16(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_15(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##F = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + 15 * STRIDE_Y + Z##F));

// LOAD_ROW_n loads the rows 0..n-1 in variables BASENAME##0 to BASENAME##(n-1)
#define LOAD_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) LOAD_ROW_##M0(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)
/** Load Blocks of M0 consecutive rows and N0 consecutive columns when using Z offset as well
 * Supported cases M0=1,2,3..16. N0=1,2,3,4,8,16, for variables BASENAME[0..M0]
 * The data to load is expected to have consecutive names for each row, For e.g. For M0=3, and basename=c, the expected data is c0, c1 and c2.
 * The Z offset is expected to have consecutive names For e.g. For M0=3, and Z=zin, the expected z offsets are zin0, zin1 and zin2.
 */
#define LOAD_BLOCK(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) LOAD_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)

#define CALCULATE_Z_OFFSET_1(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    Z##0 = (0 + (DATA_TYPE)(Y * (DATA_TYPE)M0)) / (DATA_TYPE)HEIGHT_GEMM3D;                               \
    Z##0 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##0);                                                      \
    Z##0 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_2(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_1(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##1 = (1 + (DATA_TYPE)(Y * (DATA_TYPE)M0)) / (DATA_TYPE)HEIGHT_GEMM3D;                               \
    Z##1 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##1);                                                      \
    Z##1 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_3(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_2(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##2 = (2 + (DATA_TYPE)(Y * (DATA_TYPE)M0)) / (DATA_TYPE)HEIGHT_GEMM3D;                               \
    Z##2 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##2);                                                      \
    Z##2 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_4(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_3(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##3 = (3 + (DATA_TYPE)(Y * (DATA_TYPE)M0)) / (DATA_TYPE)HEIGHT_GEMM3D;                               \
    Z##3 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##3);                                                      \
    Z##3 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_5(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_4(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##4 = (4 + (DATA_TYPE)(Y * (DATA_TYPE)M0)) / (DATA_TYPE)HEIGHT_GEMM3D;                               \
    Z##4 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##4);                                                      \
    Z##4 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_6(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_5(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##5 = (5 + (DATA_TYPE)(Y * (DATA_TYPE)M0)) / (DATA_TYPE)HEIGHT_GEMM3D;                               \
    Z##5 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##5);                                                      \
    Z##5 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_7(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_6(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##6 = (6 + (DATA_TYPE)(Y * (DATA_TYPE)M0)) / (DATA_TYPE)HEIGHT_GEMM3D;                               \
    Z##6 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##6);                                                      \
    Z##6 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_8(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_7(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##7 = (7 + (DATA_TYPE)(Y * (DATA_TYPE)M0)) / (DATA_TYPE)HEIGHT_GEMM3D;                               \
    Z##7 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##7);                                                      \
    Z##7 *= (CROSS_PLANE_PAD * STRIDE_Y);

// CALCULATE_Z_OFFSET_n calculates Z for Z##0 to Z##(n-1)
#define CALCULATE_Z_OFFSET_STR(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) CALCULATE_Z_OFFSET_##M0(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)
/** The Z offsets are expected to have consecutive names, For e.g. For M0=3, and Z=zin, the expected Z offsets are zin1, zin2, zin3.
 * Note for the REINTERPRET_INPUT_AS_3D case
 * Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
 * in order to take into account the presence of possible cross plane paddings
 *
 *  |                  |
 *  |      plane0      |
 *  |                  |
 *  |__________________|
 *  |******************|
 *  |  cross_plane_pad |
 *  |******************|
 *  |                  |
 *  |      plane1      |
 *  |                  |
 *  |__________________|
 */
#define CALCULATE_Z_OFFSET(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) CALCULATE_Z_OFFSET_STR(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)

// STORE_ROW_n macros
#define STORE_ROW_1(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    VSTORE(N0)                                                 \
    (BASENAME##0, 0, (__global DATA_TYPE *)(PTR + 0 * STRIDE_Y + Z##0));

#define STORE_ROW_2(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_1(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                 \
    (BASENAME##1, 0, (__global DATA_TYPE *)(PTR + 1 * STRIDE_Y + Z##1));

#define STORE_ROW_3(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_2(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                 \
    (BASENAME##2, 0, (__global DATA_TYPE *)(PTR + 2 * STRIDE_Y + Z##2));

#define STORE_ROW_4(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_3(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                 \
    (BASENAME##3, 0, (__global DATA_TYPE *)(PTR + 3 * STRIDE_Y + Z##3));

#define STORE_ROW_5(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_4(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                 \
    (BASENAME##4, 0, (__global DATA_TYPE *)(PTR + 4 * STRIDE_Y + Z##4));

#define STORE_ROW_6(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_5(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                 \
    (BASENAME##5, 0, (__global DATA_TYPE *)(PTR + 5 * STRIDE_Y + Z##5));

#define STORE_ROW_7(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_6(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                 \
    (BASENAME##6, 0, (__global DATA_TYPE *)(PTR + 6 * STRIDE_Y + Z##6));

#define STORE_ROW_8(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_7(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                 \
    (BASENAME##7, 0, (__global DATA_TYPE *)(PTR + 7 * STRIDE_Y + Z##7));

#define STORE_ROW_9(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_8(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                 \
    (BASENAME##8, 0, (__global DATA_TYPE *)(PTR + 8 * STRIDE_Y + Z##8));

#define STORE_ROW_10(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_9(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)      \
    VSTORE(N0)                                                  \
    (BASENAME##9, 0, (__global DATA_TYPE *)(PTR + 9 * STRIDE_Y + Z##9));

#define STORE_ROW_11(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_10(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                  \
    (BASENAME##A, 0, (__global DATA_TYPE *)(PTR + 10 * STRIDE_Y + Z##A));

#define STORE_ROW_12(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_11(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                  \
    (BASENAME##B, 0, (__global DATA_TYPE *)(PTR + 11 * STRIDE_Y + Z##B));

#define STORE_ROW_13(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_12(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                  \
    (BASENAME##C, 0, (__global DATA_TYPE *)(PTR + 12 * STRIDE_Y + Z##C));

#define STORE_ROW_14(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_13(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                  \
    (BASENAME##D, 0, (__global DATA_TYPE *)(PTR + 13 * STRIDE_Y + Z##D));

#define STORE_ROW_15(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_14(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                  \
    (BASENAME##E, 0, (__global DATA_TYPE *)(PTR + 14 * STRIDE_Y + Z##E));

#define STORE_ROW_16(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_15(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                  \
    (BASENAME##F, 0, (__global DATA_TYPE *)(PTR + 15 * STRIDE_Y + Z##F));

// CONVERT_STORE_ROW_n macros
#define CONVERT_STORE_ROW_1(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    VSTORE(N0)                                                         \
    (CONVERT_SAT((BASENAME##0), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 0 * STRIDE_Y + Z##0));

#define CONVERT_STORE_ROW_2(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_1(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                         \
    (CONVERT_SAT((BASENAME##1), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 1 * STRIDE_Y + Z##1));

#define CONVERT_STORE_ROW_3(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_2(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                         \
    (CONVERT_SAT((BASENAME##2), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 2 * STRIDE_Y + Z##2));

#define CONVERT_STORE_ROW_4(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_3(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                         \
    (CONVERT_SAT((BASENAME##3), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 3 * STRIDE_Y + Z##3));

#define CONVERT_STORE_ROW_5(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_4(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                         \
    (CONVERT_SAT((BASENAME##4), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 4 * STRIDE_Y + Z##4));

#define CONVERT_STORE_ROW_6(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_5(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                         \
    (CONVERT_SAT((BASENAME##5), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 5 * STRIDE_Y + Z##5));

#define CONVERT_STORE_ROW_7(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_6(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                         \
    (CONVERT_SAT((BASENAME##6), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 6 * STRIDE_Y + Z##6));

#define CONVERT_STORE_ROW_8(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_7(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                         \
    (CONVERT_SAT((BASENAME##7), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 7 * STRIDE_Y + Z##7));

#define CONVERT_STORE_ROW_9(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_8(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                         \
    (CONVERT_SAT((BASENAME##8), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 8 * STRIDE_Y + Z##8));

#define CONVERT_STORE_ROW_10(N0, DATA, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_9(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    VSTORE(N0)                                                     \
    (CONVERT_SAT((BASENAME##9), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 9 * STRIDE_Y + Z##9));

#define CONVERT_STORE_ROW_11(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_10(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                          \
    (CONVERT_SAT((BASENAME##A), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 10 * STRIDE_Y + Z##A));

#define CONVERT_STORE_ROW_12(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_11(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                          \
    (CONVERT_SAT((BASENAME##B), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 11 * STRIDE_Y + Z##B));

#define CONVERT_STORE_ROW_13(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_12(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                          \
    (CONVERT_SAT((BASENAME##C), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 12 * STRIDE_Y + Z##C));

#define CONVERT_STORE_ROW_14(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_13(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                          \
    (CONVERT_SAT((BASENAME##D), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 13 * STRIDE_Y + Z##D));

#define CONVERT_STORE_ROW_15(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_14(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                          \
    (CONVERT_SAT((BASENAME##E), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 14 * STRIDE_Y + Z##E));

#define CONVERT_STORE_ROW_16(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    CONVERT_STORE_ROW_15(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE(N0)                                                          \
    (CONVERT_SAT((BASENAME##F), VEC_DATA_TYPE(DATA_TYPE, N0)), 0, (__global DATA_TYPE *)(PTR + 15 * STRIDE_Y + Z##F));

// STORE_ROW_n stores the rows 0..n-1 from variables BASENAME##0 to BASENAME##(n-1)
#define STORE_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) STORE_ROW_##M0(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)

// CONVERT_STORE_ROW_n converts and stores the rows 0..n-1 from variables BASENAME##0 to BASENAME##(n-1)
#define CONVERT_STORE_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) CONVERT_STORE_ROW_##M0(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)

/** Store a block of size M0 (rows) x NO (columns).
 *  Supported cases M0=1,2,3..16. N0=2,3,4,8,16, for variables BASENAME[0..M]
 *  The data to store is expected to have consecutive names for each row, For e.g. For M0=3, and basename=c, the expected data is c0, c1 and c2.
 *  The Z offset is expected to have consecutive names For e.g. For M0=3, and Z=zin, the expected z offsets are zin0, zin1 and zin2.
 */
#define STORE_BLOCK(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) STORE_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)

/** Convert and store a block of size M0 (rows) x NO (columns).
 *  Supported cases M0=1,2,3..16. N0=2,3,4,8,16, for variables BASENAME[0..M]
 *  The data to store is expected to have consecutive names for each row, For e.g. For M0=3, and basename=c, the expected data is c0, c1 and c2.
 *  The Z offset is expected to have consecutive names For e.g. For M0=3, and Z=zin, the expected z offsets are zin0, zin1 and zin2.
 */
#define CONVERT_STORE_BLOCK(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) CONVERT_STORE_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)

#define SCALE_ROW_1(DATA_TYPE, BASENAME, SCALE) \
    BASENAME##0 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_2(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_1(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##1 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_3(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_2(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##2 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_4(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_3(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##3 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_5(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_4(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##4 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_6(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_5(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##5 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_7(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_6(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##6 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_8(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_7(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##7 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_9(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_8(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##8 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_10(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_9(DATA_TYPE, BASENAME, SCALE)      \
    BASENAME##9 *= (DATA_TYPE)SCALE;

#define SCALE_ROW_11(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_10(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##A *= (DATA_TYPE)SCALE;

#define SCALE_ROW_12(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_11(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##B *= (DATA_TYPE)SCALE;

#define SCALE_ROW_13(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_12(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##C *= (DATA_TYPE)SCALE;

#define SCALE_ROW_14(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_13(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##D *= (DATA_TYPE)SCALE;

#define SCALE_ROW_15(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_14(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##E *= (DATA_TYPE)SCALE;

#define SCALE_ROW_16(DATA_TYPE, BASENAME, SCALE) \
    SCALE_ROW_15(DATA_TYPE, BASENAME, SCALE)     \
    BASENAME##F *= (DATA_TYPE)SCALE;

// SCALE_BLOCK_n scales the variables BASENAME##0 to BASENAME##(n-1) by SCALE
#define SCALE_BLOCK_STR(N, DATA_TYPE, BASENAME, SCALE) SCALE_ROW_##N(DATA_TYPE, BASENAME, SCALE)
/** Scale elements stored in variables BASENAME##0 to BASENAME##(N-1) by SCALE
 * Supported cases N=1,2,3..16, for variables BASENAME[0..N]
 */
#define SCALE_BLOCK(N, DATA_TYPE, BASENAME, SCALE) SCALE_BLOCK_STR(N, DATA_TYPE, BASENAME, SCALE)

/** Given a set of vectors of size K0, these macros create a new vector to contain the values at index IDX_COL (with IDX_COL < N0) for all input vectors */
#define COLUMN_VECTOR1(IDX_COL, BASENAME, X) \
    uchar BASENAME##IDX_COL = (uchar)((X##0).s##IDX_COL);
#define COLUMN_VECTOR2(IDX_COL, BASENAME, X) \
    uchar2 BASENAME##IDX_COL = (uchar2)((X##0).s##IDX_COL, (X##1).s##IDX_COL);
#define COLUMN_VECTOR3(IDX_COL, BASENAME, X) \
    uchar3 BASENAME##IDX_COL = (uchar3)((X##0).s##IDX_COL, (X##1).s##IDX_COL, (X##2).s##IDX_COL);
#define COLUMN_VECTOR4(IDX_COL, BASENAME, X) \
    uchar4 BASENAME##IDX_COL = (uchar4)((X##0).s##IDX_COL, (X##1).s##IDX_COL, (X##2).s##IDX_COL, (X##3).s##IDX_COL);
#define COLUMN_VECTOR8(IDX_COL, BASENAME, X) \
    uchar8 BASENAME##IDX_COL = (uchar8)((X##0).s##IDX_COL, (X##1).s##IDX_COL, (X##2).s##IDX_COL, (X##3).s##IDX_COL, (X##4).s##IDX_COL, (X##5).s##IDX_COL, (X##6).s##IDX_COL, (X##7).s##IDX_COL);
#define COLUMN_VECTOR16(IDX_COL, BASENAME, X) \
    uchar16 BASENAME##IDX_COL = (uchar16)((X##0).s##IDX_COL, (X##1).s##IDX_COL, (X##2).s##IDX_COL, (X##3).s##IDX_COL, (X##4).s##IDX_COL, (X##5).s##IDX_COL, (X##6).s##IDX_COL, (X##7).s##IDX_COL, (X##8).s##IDX_COL, (X##9).s##IDX_COL, (X##A).s##IDX_COL, (X##B).s##IDX_COL, (X##C).s##IDX_COL, (X##D).s##IDX_COL, (X##E).s##IDX_COL, (X##F).s##IDX_COL);

/** Given N0 vectors of size K0, these macros create K0 vectors of size N0 which are the result of a transposition */
#define TRANSPOSE_K0X1(K0, BASENAME, B) \
    COLUMN_VECTOR(K0, 0, BASENAME, B);
#define TRANSPOSE_K0X2(K0, BASENAME, B) \
    TRANSPOSE_K0X1(K0, BASENAME, B);    \
    COLUMN_VECTOR(K0, 1, BASENAME, B);
#define TRANSPOSE_K0X3(K0, BASENAME, B) \
    TRANSPOSE_K0X2(K0, BASENAME, B);    \
    COLUMN_VECTOR(K0, 2, BASENAME, B);
#define TRANSPOSE_K0X4(K0, BASENAME, B) \
    TRANSPOSE_K0X3(K0, BASENAME, B);    \
    COLUMN_VECTOR(K0, 3, BASENAME, B);
#define TRANSPOSE_K0X8(K0, BASENAME, B) \
    TRANSPOSE_K0X4(K0, BASENAME, B);    \
    COLUMN_VECTOR(K0, 4, BASENAME, B);  \
    COLUMN_VECTOR(K0, 5, BASENAME, B);  \
    COLUMN_VECTOR(K0, 6, BASENAME, B);  \
    COLUMN_VECTOR(K0, 7, BASENAME, B);
#define TRANSPOSE_K0X16(K0, BASENAME, B) \
    TRANSPOSE_K0X8(K0, BASENAME, B);     \
    COLUMN_VECTOR(K0, 8, BASENAME, B);   \
    COLUMN_VECTOR(K0, 9, BASENAME, B);   \
    COLUMN_VECTOR(K0, A, BASENAME, B);   \
    COLUMN_VECTOR(K0, B, BASENAME, B);   \
    COLUMN_VECTOR(K0, C, BASENAME, B);   \
    COLUMN_VECTOR(K0, D, BASENAME, B);   \
    COLUMN_VECTOR(K0, E, BASENAME, B);   \
    COLUMN_VECTOR(K0, F, BASENAME, B);

#define COLUMN_VECTOR(K0, IDX_COL, BASENAME, B) \
    CONCAT(COLUMN_VECTOR, K0)                   \
    (IDX_COL, BASENAME, B);

#define TRANSPOSE_K0XN0(K0, N0, BASENAME, B) \
    CONCAT(TRANSPOSE_K0X, N0)                \
    (K0, BASENAME, B);

#define ADD_ROW_1(BASENAME, BIAS) \
    BASENAME##0 += BIAS##0;

#define ADD_ROW_2(BASENAME, BIAS) \
    ADD_ROW_1(BASENAME, BIAS)     \
    BASENAME##1 += BIAS##1;

#define ADD_ROW_3(BASENAME, BIAS) \
    ADD_ROW_2(BASENAME, BIAS)     \
    BASENAME##2 += BIAS##2;

#define ADD_ROW_4(BASENAME, BIAS) \
    ADD_ROW_3(BASENAME, BIAS)     \
    BASENAME##3 += BIAS##3;

#define ADD_ROW_5(BASENAME, BIAS) \
    ADD_ROW_4(BASENAME, BIAS)     \
    BASENAME##4 += BIAS##4;

#define ADD_ROW_6(BASENAME, BIAS) \
    ADD_ROW_5(BASENAME, BIAS)     \
    BASENAME##5 += BIAS##5;

#define ADD_ROW_7(BASENAME, BIAS) \
    ADD_ROW_6(BASENAME, BIAS)     \
    BASENAME##6 += BIAS##6;

#define ADD_ROW_8(BASENAME, BIAS) \
    ADD_ROW_7(BASENAME, BIAS)     \
    BASENAME##7 += BIAS##7;

#define ADD_ROW_9(BASENAME, BIAS) \
    ADD_ROW_8(BASENAME, BIAS)     \
    BASENAME##8 += BIAS##8;

#define ADD_ROW_10(BASENAME, BIAS) \
    ADD_ROW_9(BASENAME, BIAS)      \
    BASENAME##9 += BIAS##9;

#define ADD_ROW_11(BASENAME, BIAS) \
    ADD_ROW_10(BASENAME, BIAS)     \
    BASENAME##A += BIAS##A;

#define ADD_ROW_12(BASENAME, BIAS) \
    ADD_ROW_11(BASENAME, BIAS)     \
    BASENAME##B += BIAS##B;

#define ADD_ROW_13(BASENAME, BIAS) \
    ADD_ROW_12(BASENAME, BIAS)     \
    BASENAME##C += BIAS##C;

#define ADD_ROW_14(BASENAME, BIAS) \
    ADD_ROW_13(BASENAME, BIAS)     \
    BASENAME##D += BIAS##D;

#define ADD_ROW_15(BASENAME, BIAS) \
    ADD_ROW_14(BASENAME, BIAS)     \
    BASENAME##E += BIAS##E;

#define ADD_ROW_16(BASENAME, BIAS) \
    ADD_ROW_15(BASENAME, BIAS)     \
    BASENAME##F += BIAS##F;

// ADD_ROW_n add the variables BIAS##0... BIAS##(n-1) to BASENAME##0 to BASENAME##(n-1)
#define ADD_BLOCK_STR(N, BASENAME, BIAS) ADD_ROW_##N(BASENAME, BIAS)
/** Add BIAS to  BASENAME##0 ... BASENAME##(N-1)
 * Supported cases N=1,2,3..16, for variables BASENAME[0..N]
 */
#define ADD_BLOCK(N, BASENAME, BIAS) ADD_BLOCK_STR(N, BASENAME, BIAS)

#define ADD_ROW_BROADCAST_1(BASENAME, BIAS) \
    BASENAME##0 += BIAS;

#define ADD_ROW_BROADCAST_2(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_1(BASENAME, BIAS)     \
    BASENAME##1 += BIAS;

#define ADD_ROW_BROADCAST_3(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_2(BASENAME, BIAS)     \
    BASENAME##2 += BIAS;

#define ADD_ROW_BROADCAST_4(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_3(BASENAME, BIAS)     \
    BASENAME##3 += BIAS;

#define ADD_ROW_BROADCAST_5(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_4(BASENAME, BIAS)     \
    BASENAME##4 += BIAS;

#define ADD_ROW_BROADCAST_6(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_5(BASENAME, BIAS)     \
    BASENAME##5 += BIAS;

#define ADD_ROW_BROADCAST_7(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_6(BASENAME, BIAS)     \
    BASENAME##6 += BIAS;

#define ADD_ROW_BROADCAST_8(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_7(BASENAME, BIAS)     \
    BASENAME##7 += BIAS;

#define ADD_ROW_BROADCAST_9(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_8(BASENAME, BIAS)     \
    BASENAME##8 += BIAS;

#define ADD_ROW_BROADCAST_10(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_9(BASENAME, BIAS)      \
    BASENAME##9 += BIAS;

#define ADD_ROW_BROADCAST_11(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_10(BASENAME, BIAS)     \
    BASENAME##A += BIAS;

#define ADD_ROW_BROADCAST_12(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_11(BASENAME, BIAS)     \
    BASENAME##B += BIAS;

#define ADD_ROW_BROADCAST_13(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_12(BASENAME, BIAS)     \
    BASENAME##C += BIAS;

#define ADD_ROW_BROADCAST_14(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_13(BASENAME, BIAS)     \
    BASENAME##D += BIAS;

#define ADD_ROW_BROADCAST_15(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_14(BASENAME, BIAS)     \
    BASENAME##E += BIAS;

#define ADD_ROW_BROADCAST_16(BASENAME, BIAS) \
    ADD_ROW_BROADCAST_15(BASENAME, BIAS)     \
    BASENAME##F += BIAS;

// ADD_ROW_n add the variables BIAS to BASENAME##0 to BASENAME##(n-1)
#define ADD_BLOCK_BROADCAST_STR(N, BASENAME, BIAS) ADD_ROW_BROADCAST_##N(BASENAME, BIAS)
/** Add elements stored in variables BIAS##0 ... BIAS##(N-1) to  BASENAME##0 ... BASENAME##(N-1)
 * Supported cases N=1,2,3..16, for variables BASENAME[0..N]
 */
#define ADD_BLOCK_BROADCAST(N, BASENAME, BIAS) ADD_BLOCK_BROADCAST_STR(N, BASENAME, BIAS)

#define ACTIVATION_ROW_1(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    BASENAME##0 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##0, A_VAL, B_VAL);

#define ACTIVATION_ROW_2(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_1(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##1 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##1, A_VAL, B_VAL);

#define ACTIVATION_ROW_3(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_2(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##2 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##2, A_VAL, B_VAL);

#define ACTIVATION_ROW_4(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_3(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##3 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##3, A_VAL, B_VAL);

#define ACTIVATION_ROW_5(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_4(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##4 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##4, A_VAL, B_VAL);

#define ACTIVATION_ROW_6(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_5(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##5 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##5, A_VAL, B_VAL);

#define ACTIVATION_ROW_7(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_6(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##6 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##6, A_VAL, B_VAL);

#define ACTIVATION_ROW_8(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_7(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##7 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##7, A_VAL, B_VAL);

#define ACTIVATION_ROW_9(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_8(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##8 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##8, A_VAL, B_VAL);

#define ACTIVATION_ROW_10(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_9(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)      \
    BASENAME##9 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##9, A_VAL, B_VAL);

#define ACTIVATION_ROW_11(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_10(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##A = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##A, A_VAL, B_VAL);

#define ACTIVATION_ROW_12(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_11(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##B = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##B, A_VAL, B_VAL);

#define ACTIVATION_ROW_13(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_12(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##C = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##C, A_VAL, B_VAL);

#define ACTIVATION_ROW_14(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_13(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##D = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##D, A_VAL, B_VAL);

#define ACTIVATION_ROW_15(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_14(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##E = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##E, A_VAL, B_VAL);

#define ACTIVATION_ROW_16(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_15(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##F = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, BASENAME##F, A_VAL, B_VAL);

// ACTIVATION_ROW_n apply activation to the variables BASENAME##0... BASENAME##(n-1)
#define ACTIVATION_BLOCK_STR(N, ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) ACTIVATION_ROW_##N(ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)
/** Apply activation to the variables BASENAME##0... BASENAME##(n-1)
 * Supported cases N=1,2,3..16, for variables BASENAME[0..N]
 */
#define ACTIVATION_BLOCK(N, ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL) ACTIVATION_BLOCK_STR(N, ACTIVATION_TYPE, DATA_TYPE, BASENAME, A_VAL, B_VAL)