/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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

/** Utility macro to access a vector with the scalar positions
 *
 * Supported cases are: Offset can only be of the same size of the OpenCL vector (2,3,4,8,16)
 *
 * @param[in] offset The offset within the vector. Offset can only be of the same size of the OpenCL vector (2,3,4,8,16)
 * @param[in] n0     The number of consecutive columns to access. n0 + offset must be <= 16
 * @param[in] x      Vector to access
 *
 */
#define SCALAR_ACCESS_STR(offset, n0, x) scalar_access_##offset##_##n0(x)
#define SCALAR_ACCESS(offset, n0, x) SCALAR_ACCESS_STR(offset, n0, x)

// offset == 0
#define scalar_access_0_1(x) ((x).s0)
#define scalar_access_0_2(x) ((x).s01)
#define scalar_access_0_3(x) ((x).s012)
#define scalar_access_0_4(x) ((x).s0123)
#define scalar_access_0_8(x) ((x).s01234567)
#define scalar_access_0_16(x) ((x).s0123456789ABCDEF)

// offset == 1
#define scalar_access_1_1(x) ((x).s1)
#define scalar_access_1_2(x) ((x).s12)
#define scalar_access_1_3(x) ((x).s123)
#define scalar_access_1_4(x) ((x).s1234)
#define scalar_access_1_8(x) ((x).s12345678)

// offset == 2
#define scalar_access_2_1(x) ((x).s2)
#define scalar_access_2_2(x) ((x).s23)
#define scalar_access_2_3(x) ((x).s234)
#define scalar_access_2_4(x) ((x).s2345)
#define scalar_access_2_8(x) ((x).s23456789)

// offset == 3
#define scalar_access_3_1(x) ((x).s3)
#define scalar_access_3_2(x) ((x).s34)
#define scalar_access_3_3(x) ((x).s345)
#define scalar_access_3_4(x) ((x).s3456)
#define scalar_access_3_8(x) ((x).s3456789A)

// offset == 4
#define scalar_access_4_1(x) ((x).s4)
#define scalar_access_4_2(x) ((x).s45)
#define scalar_access_4_3(x) ((x).s456)
#define scalar_access_4_4(x) ((x).s4567)
#define scalar_access_4_8(x) ((x).s456789AB)

// offset == 8
#define scalar_access_8_1(x) ((x).s8)
#define scalar_access_8_2(x) ((x).s89)
#define scalar_access_8_3(x) ((x).s89A)
#define scalar_access_8_4(x) ((x).s89AB)
#define scalar_access_8_8(x) ((x).s89ABCDEF)

// offset == 12
#define scalar_access_12_1(x) ((x).sC)
#define scalar_access_12_2(x) ((x).sCD)
#define scalar_access_12_3(x) ((x).sCDE)
#define scalar_access_12_4(x) ((x).sCDEF)

// offset == 16
#define scalar_access_16_1(x) ((x).sF)

/** Loads the rows from 0 to n-1 in the given variables (BASENAME0 to BASENAMEn-1) without allocating variables.
 * @name LOAD_TENSOR_ROW_n
 *
 * @param[in] N0         The number of columns to load
 * @param[in] DATA_TYPE  The data type of variables
 * @param[in] BASENAME   The basename of the destination variables for the loaded rows
 * @param[in] PTR        The base pointer
 * @param[in] COL_OFFSET The column vector offset. COL_OFFSET + N0 must be <= 16
 * @param[in] STRIDE_Y   The stride value in y-axis direction
 * @param[in] Z          The z-axis offset vector
 * @{
 */
#define LOAD_TENSOR_ROW_0(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    ({})

#define LOAD_TENSOR_ROW_1(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##0) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 0 * STRIDE_Y + Z##0));

#define LOAD_TENSOR_ROW_2(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_1(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##1) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 1 * STRIDE_Y + Z##1));

#define LOAD_TENSOR_ROW_3(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_2(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##2) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 2 * STRIDE_Y + Z##2));

#define LOAD_TENSOR_ROW_4(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_3(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##3) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 3 * STRIDE_Y + Z##3));

#define LOAD_TENSOR_ROW_5(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_4(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##4) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 4 * STRIDE_Y + Z##4));

#define LOAD_TENSOR_ROW_6(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_5(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##5) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 5 * STRIDE_Y + Z##5));

#define LOAD_TENSOR_ROW_7(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_6(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##6) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 6 * STRIDE_Y + Z##6));

#define LOAD_TENSOR_ROW_8(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_7(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##7) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 7 * STRIDE_Y + Z##7));

#define LOAD_TENSOR_ROW_9(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_8(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##8) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 8 * STRIDE_Y + Z##8));

#define LOAD_TENSOR_ROW_10(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_9(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)      \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##9) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 9 * STRIDE_Y + Z##9));

#define LOAD_TENSOR_ROW_11(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_10(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##A) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 10 * STRIDE_Y + Z##A));

#define LOAD_TENSOR_ROW_12(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_11(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##B) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 11 * STRIDE_Y + Z##B));

#define LOAD_TENSOR_ROW_13(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_12(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##C) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 12 * STRIDE_Y + Z##C));

#define LOAD_TENSOR_ROW_14(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_13(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##D) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 13 * STRIDE_Y + Z##D));

#define LOAD_TENSOR_ROW_15(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_14(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##E) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 14 * STRIDE_Y + Z##E));

#define LOAD_TENSOR_ROW_16(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) \
    LOAD_TENSOR_ROW_15(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)     \
    SCALAR_ACCESS(COL_OFFSET, N0, BASENAME##F) = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + 15 * STRIDE_Y + Z##F));
/** @}*/ // end of group LOAD_TENSOR_ROW_n

/** Load tensor (consecutive rows and columns) with Z offset.
 * @name LOAD_TENSOR
 *
 * Supported cases are M0=1,2,3,...,16 and N0=1,2,3,4,8,16
 * The data to load is expected to have consecutive names for each row.
 * E.g., for M0=3, and BASENAME=c, the expected data is c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for M0=3, and Z=zin, the expected Z offsets are zin0, zin1 and zin2.
 *
 * @param[in] M0         The number of consecutive rows
 * @param[in] N0         The number of consecutive columns
 * @param[in] DATA_TYPE  The data type of the target
 * @param[in] BASENAME   The basename of the result variables
 * @param[in] PTR        The base pointer for the data
 * @param[in] COL_OFFSET The column vector offset. COL_OFFSET + N0 must be <= 16
 * @param[in] STRIDE_Y   The stride in y-axis direction
 * @param[in] Z          The z-axis offset vector
 * @{
 */
#define LOAD_TENSOR_STR(M0, N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) LOAD_TENSOR_ROW_##M0(N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)
#define LOAD_TENSOR(M0, N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z) LOAD_TENSOR_STR(M0, N0, DATA_TYPE, BASENAME, PTR, COL_OFFSET, STRIDE_Y, Z)
/** @} */ // end of group LOAD_TENSOR

/** Load 2D tensor (consecutive rows and columns) with Z offset.
 * @name LOAD_TENSOR_M0Xn
 *
 * @param[in] M0        The number of rows to load [0-16]
 * @param[in] N0        The number of columns to load [0-16]
 * @param[in] DATA_TYPE The data type of variables
 * @param[in] BASENAME  The basename of the destination variables for the loaded rows
 * @param[in] PTR       The base pointer
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The z-axis offset vector
 * @{
 */
#define LOAD_TENSOR_M0X0(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    ({})

#define LOAD_TENSOR_M0X1(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, N0, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);

#define LOAD_TENSOR_M0X2(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, N0, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);

#define LOAD_TENSOR_M0X3(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, N0, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);

#define LOAD_TENSOR_M0X4(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, N0, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);

#define LOAD_TENSOR_M0X5(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, 4, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);       \
    LOAD_TENSOR(M0, 1, DATA_TYPE, a, input_ptr + 4 * sizeof(DATA_TYPE), 4, src_stride_y, zin);

#define LOAD_TENSOR_M0X6(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, 4, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);       \
    LOAD_TENSOR(M0, 2, DATA_TYPE, a, input_ptr + 4 * sizeof(DATA_TYPE), 4, src_stride_y, zin);

#define LOAD_TENSOR_M0X7(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, 4, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);       \
    LOAD_TENSOR(M0, 3, DATA_TYPE, a, input_ptr + 4 * sizeof(DATA_TYPE), 4, src_stride_y, zin);

#define LOAD_TENSOR_M0X8(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, N0, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);

#define LOAD_TENSOR_M0X9(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, 8, DATA_TYPE, a, input_ptr 0, src_stride_y, zin);        \
    LOAD_TENSOR(M0, 1, DATA_TYPE, a, input_ptr + 8 * sizeof(DATA_TYPE), 8, src_stride_y, zin);

#define LOAD_TENSOR_M0X10(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, 8, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);        \
    LOAD_TENSOR(M0, 2, DATA_TYPE, a, input_ptr + 8 * sizeof(DATA_TYPE), 8, src_stride_y, zin);

#define LOAD_TENSOR_M0X11(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, 8, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);        \
    LOAD_TENSOR(M0, 3, DATA_TYPE, a, input_ptr + 8 * sizeof(DATA_TYPE), 8, src_stride_y, zin);

#define LOAD_TENSOR_M0X12(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, 8, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);        \
    LOAD_TENSOR(M0, 4, DATA_TYPE, a, input_ptr + 8 * sizeof(DATA_TYPE), 8, src_stride_y, zin);

#define LOAD_TENSOR_M0X13(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin)                  \
    LOAD_TENSOR(M0, 8, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);                         \
    LOAD_TENSOR(M0, 4, DATA_TYPE, a, input_ptr + 8 * sizeof(DATA_TYPE), 8, src_stride_y, zin); \
    LOAD_TENSOR(M0, 1, DATA_TYPE, a, input_ptr + 12 * sizeof(DATA_TYPE), 12, src_stride_y, zin);

#define LOAD_TENSOR_M0X14(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin)                  \
    LOAD_TENSOR(M0, 8, DATA_TYPE, a, input_ptr 0, src_stride_y, zin);                          \
    LOAD_TENSOR(M0, 4, DATA_TYPE, a, input_ptr + 8 * sizeof(DATA_TYPE), 8, src_stride_y, zin); \
    LOAD_TENSOR(M0, 2, DATA_TYPE, a, input_ptr + 12 * sizeof(DATA_TYPE), 12, src_stride_y, zin);

#define LOAD_TENSOR_M0X15(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin)                  \
    LOAD_TENSOR(M0, 8, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);                         \
    LOAD_TENSOR(M0, 4, DATA_TYPE, a, input_ptr + 8 * sizeof(DATA_TYPE), 8, src_stride_y, zin); \
    LOAD_TENSOR(M0, 3, DATA_TYPE, a, input_ptr + 12 * sizeof(DATA_TYPE), 12, src_stride_y, zin);

#define LOAD_TENSOR_M0X16(M0, N0, DATA_TYPE, a, input_ptr, src_stride_y, zin) \
    LOAD_TENSOR(M0, N0, DATA_TYPE, a, input_ptr, 0, src_stride_y, zin);
/** @}*/ // end of group LOAD_TENSOR_M0Xn

/** Load 2D tensor (consecutive rows and columns) with Z offset.
 * @name LOAD_TENSOR_M0XN0
 *
 * @param[in] M0        The number of consecutive rows [0-16]
 * @param[in] N0        The number of consecutive columns [0-16]
 * @param[in] DATA_TYPE The data type of the target
 * @param[in] BASENAME  The basename of the result variables
 * @param[in] PTR       The base pointer for the data
 * @param[in] STRIDE_Y  The stride in y-axis direction
 * @param[in] Z         The z-axis offset vector
 * @{
 */
#define LOAD_TENSOR_M0XN0_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) LOAD_TENSOR_M0X##N0(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)
#define LOAD_TENSOR_M0XN0(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) LOAD_TENSOR_M0XN0_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)
/** @}*/ // end of group LOAD_TENSOR_M0XN0

/** Loads the rows from 0 to n-1 in the given variables (BASENAME0 to BASENAMEn-1).
 * @name LOAD_ROW_n
 *
 * @param[in] N0        The number of columns to load
 * @param[in] DATA_TYPE The data type of variables
 * @param[in] BASENAME  The basename of the destination variables for the loaded rows
 * @param[in] PTR       The base pointer
 * @param[in] OFFSET    The offset within a row
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The z-axis offset vector
 * @{
 */
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

/** @}*/ // end of group LOAD_ROW_n

/** Load Blocks (consecutive rows and columns) with Z offset.
 * @name LOAD_BLOCK
 *
 * Supported cases are M0=1,2,3,...,16 and N0=1,2,3,4,8,16
 * The data to load is expected to have consecutive names for each row.
 * E.g., for M0=3, and BASENAME=c, the expected data is c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for M0=3, and Z=zin, the expected Z offsets are zin0, zin1 and zin2.
 *
 * @param[in] M0        The number of consecutive rows
 * @param[in] N0        The number of consecutive columns
 * @param[in] DATA_TYPE The data type of the target
 * @param[in] BASENAME  The basename of the result variables
 * @param[in] PTR       The base pointer for the data
 * @param[in] OFFSET    The offset within a row
 * @param[in] STRIDE_Y  The stride in y-axis direction
 * @param[in] Z         The z-axis offset vector
 * @{
 */
#define LOAD_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) LOAD_ROW_##M0(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)
#define LOAD_BLOCK(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) LOAD_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)
/** @} */ // end of group LOAD_BLOCK

/** Partially load the 0 to (n-1)th rows of the given variables
 * @name LOAD_ROW_PARTIAL_n
 * Within each row, load the lower @p LOAD_N0 elements of vectors of width @p N0
 *
 * @note in case @p LOAD_N0 != 1, 2, 3, 4, 8, 16, extra vload(s) will be invoked, thus incurring small performance penalty.
 *
 * @param[in] N0        The width of the passed in vector. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] LOAD_N0   The **lower** size of the vectors to load. Supported: [1-16 and <= @p N0
 * @param[in] DATA_TYPE The data type of the vectors
 * @param[in] BASENAME  The basename of the variables
 * @param[in] PTR       The base pointer
 * @param[in] OFFSET    The offset within a row
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The offset in z-axis direction
 * @{
 */
#define LOAD_ROW_PARTIAL_1(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                         \
    (BASENAME##0, 0, (__global DATA_TYPE *)(PTR + OFFSET + 0 * STRIDE_Y + Z##0));

#define LOAD_ROW_PARTIAL_2(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_1(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                         \
    (BASENAME##1, 0, (__global DATA_TYPE *)(PTR + OFFSET + 1 * STRIDE_Y + Z##1));

#define LOAD_ROW_PARTIAL_3(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_2(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                         \
    (BASENAME##2, 0, (__global DATA_TYPE *)(PTR + OFFSET + 2 * STRIDE_Y + Z##2));

#define LOAD_ROW_PARTIAL_4(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_3(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                         \
    (BASENAME##3, 0, (__global DATA_TYPE *)(PTR + OFFSET + 3 * STRIDE_Y + Z##3));

#define LOAD_ROW_PARTIAL_5(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_4(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                         \
    (BASENAME##4, 0, (__global DATA_TYPE *)(PTR + OFFSET + 4 * STRIDE_Y + Z##4));

#define LOAD_ROW_PARTIAL_6(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_5(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                         \
    (BASENAME##5, 0, (__global DATA_TYPE *)(PTR + OFFSET + 5 * STRIDE_Y + Z##5));

#define LOAD_ROW_PARTIAL_7(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_6(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                         \
    (BASENAME##6, 0, (__global DATA_TYPE *)(PTR + OFFSET + 6 * STRIDE_Y + Z##6));

#define LOAD_ROW_PARTIAL_8(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_7(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                         \
    (BASENAME##7, 0, (__global DATA_TYPE *)(PTR + OFFSET + 7 * STRIDE_Y + Z##7));

#define LOAD_ROW_PARTIAL_9(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_8(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                         \
    (BASENAME##8, 0, (__global DATA_TYPE *)(PTR + OFFSET + 8 * STRIDE_Y + Z##8));

#define LOAD_ROW_PARTIAL_10(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_9(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)      \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                          \
    (BASENAME##9, 0, (__global DATA_TYPE *)(PTR + OFFSET + 9 * STRIDE_Y + Z##9));

#define LOAD_ROW_PARTIAL_11(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_10(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                          \
    (BASENAME##A, 0, (__global DATA_TYPE *)(PTR + OFFSET + 10 * STRIDE_Y + Z##A));

#define LOAD_ROW_PARTIAL_12(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_11(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                          \
    (BASENAME##B, 0, (__global DATA_TYPE *)(PTR + OFFSET + 11 * STRIDE_Y + Z##B));

#define LOAD_ROW_PARTIAL_13(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_12(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                          \
    (BASENAME##C, 0, (__global DATA_TYPE *)(PTR + OFFSET + 12 * STRIDE_Y + Z##C));

#define LOAD_ROW_PARTIAL_14(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_13(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                          \
    (BASENAME##D, 0, (__global DATA_TYPE *)(PTR + OFFSET + 13 * STRIDE_Y + Z##D));

#define LOAD_ROW_PARTIAL_15(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_14(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                          \
    (BASENAME##E, 0, (__global DATA_TYPE *)(PTR + OFFSET + 14 * STRIDE_Y + Z##E));

#define LOAD_ROW_PARTIAL_16(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) \
    LOAD_ROW_PARTIAL_15(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)     \
    VLOAD_PARTIAL(N0, LOAD_N0)                                                          \
    (BASENAME##F, 0, (__global DATA_TYPE *)(PTR + OFFSET + 15 * STRIDE_Y + Z##F));
/** @} */ // end of group LOAD_ROW_PARTIAL_n

/** Partially load a block of the given size LOAD_M0xLOAD_N0
 * @name LOAD_BLOCK_PARTIAL
 *
 * @note The vector width @p N0 is also required for correct partial storing behaviour.
 * @note in case @p LOAD_N0 != 1, 2, 3, 4, 8, 16, extra vload(s) will be invoked, thus incurring small performance penalty.
 *
 * The data to load is expected to have consecutive names for each row.
 * E.g., for LOAD_M0=3 and basename=c, the expected names are c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for LOAD_M0=3 and Z=zin, the expected z offset names are zin0, zin1 and zin2.
 *
 * @param[in] LOAD_M0   The number of rows to load. Supported: 1-16
 * @param[in] LOAD_N0   The lower number of elements of vectors to load. Supported: 1-16 and <= @p N0
 * @param[in] N0        The size of each vector. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] DATA_TYPE The data type of the vectors
 * @param[in] BASENAME  The basename of the variables
 * @param[in] PTR       The base pointer
 * @param[in] OFFSET    The offset within a row
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The offset in z-axis direction
 * @{
 */
#define LOAD_BLOCK_PARTIAL_STR(LOAD_M0, LOAD_N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) LOAD_ROW_PARTIAL_##LOAD_M0(N0, LOAD_N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)
#define LOAD_BLOCK_PARTIAL(LOAD_M0, LOAD_N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z) LOAD_BLOCK_PARTIAL_STR(LOAD_M0, LOAD_N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)
/** Load a block that can be partial in both x and y dimensions
 *
 * @note in cases @p PARTIAL_STORE_N0 != 1, 2, 3, 4, 8, 16, extra vload(s) will be invoked, thus incurring small performance penalty.
 *
 * The data to load is expected to have consecutive names for each row.
 * E.g., for M0=3 and basename=c, the expected names are c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for M0=3 and Z=zin, the expected z offset names are zin0, zin1 and zin2.
 *
 * @param[in] M0               The number of rows to load, for non-partial blocks. Supported: 1-16
 * @param[in] N0               The size of each vector, for non-partial blocks. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] DATA_TYPE        The data type of the vectors
 * @param[in] BASENAME         The basename of the variables
 * @param[in] PTR              The base pointer
 * @param[in] OFFSET           The offset within a row
 * @param[in] STRIDE_Y         The stride value in y-axis direction
 * @param[in] Z                The offset in z-axis direction
 * @param[in] PARTIAL_STORE_M0 The partial size in y, for partial blocks. Supported range: [1, @p M0)
 * @param[in] PARTIAL_STORE_N0 The partial size in x, for partial blocks. Supported range: [1, @p N0)
 * @param[in] PARTIAL_COND_Y   Condition on the y axis to perform the partial load Y. True to use PARTIAL_STORE_M0 rather than M0.
 * @param[in] PARTIAL_COND_X   Condition on the x axis to perform the partial load X. True to use PARTIAL_STORE_N0 rather than N0.
 */
#define LOAD_BLOCK_PARTIAL_IN_X_AND_Y(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    if(!(PARTIAL_COND_X) && !(PARTIAL_COND_Y))                                                                                                                   \
    {                                                                                                                                                            \
        LOAD_BLOCK_PARTIAL(M0, N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z);                                                                           \
    }                                                                                                                                                            \
    else if((PARTIAL_COND_Y) && !(PARTIAL_COND_X))                                                                                                               \
    {                                                                                                                                                            \
        LOAD_BLOCK_PARTIAL(PARTIAL_STORE_M0, N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z);                                                             \
    }                                                                                                                                                            \
    else if(!(PARTIAL_COND_Y) && (PARTIAL_COND_X))                                                                                                               \
    {                                                                                                                                                            \
        LOAD_BLOCK_PARTIAL(M0, PARTIAL_STORE_N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z);                                                             \
    }                                                                                                                                                            \
    else                                                                                                                                                         \
    {                                                                                                                                                            \
        LOAD_BLOCK_PARTIAL(PARTIAL_STORE_M0, PARTIAL_STORE_N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z);                                               \
    }
/** Load a block that can only be partial in x but not y.
 *
 * @note in case @p N0 or @p PARTIAL_STORE_N0 != 1, 2, 3, 4, 8, 16, extra vload(s) will be invoked, thus incurring small performance penalty.
 *
 * The data to load is expected to have consecutive names for each row.
 * E.g., for M0=3 and basename=c, the expected names are c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for M0=3 and Z=zin, the expected z offset names are zin0, zin1 and zin2.
 *
 * @param[in] M0               The number of rows to load, for non-partial blocks. Supported: 1-16
 * @param[in] N0               The size of each vector, for non-partial blocks. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] DATA_TYPE        The data type of the vectors
 * @param[in] BASENAME         The basename of the variables
 * @param[in] PTR              The base pointer
 * @param[in] OFFSET           The offset within a row
 * @param[in] STRIDE_Y         The stride value in y-axis direction
 * @param[in] Z                The offset in z-axis direction
 * @param[in] PARTIAL_STORE_N0 The partial size in x, for partial blocks. Supported range: [1, @p N0)
 * @param[in] PARTIAL_COND_X   Condition on the x axis to perform the partial load X. True to use PARTIAL_STORE_N0 rather than N0.
 */
#define LOAD_BLOCK_PARTIAL_IN_X(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_N0, PARTIAL_COND_X) \
    if(!(PARTIAL_COND_X))                                                                                                \
    {                                                                                                                    \
        LOAD_BLOCK_PARTIAL(M0, N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z);                                   \
    }                                                                                                                    \
    else                                                                                                                 \
    {                                                                                                                    \
        LOAD_BLOCK_PARTIAL(M0, PARTIAL_STORE_N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z);                     \
    }
/** Load a block that can only be partial in y but not x.
 *
 * @note in case @p N0 or @p PARTIAL_STORE_N0 != 1, 2, 3, 4, 8, 16, extra vload(s) will be invoked, thus incurring small performance penalty.
 *
 * The data to store is expected to have consecutive names for each row.
 * E.g., for M0=3 and basename=c, the expected names are c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for M0=3 and Z=zin, the expected z offset names are zin0, zin1 and zin2.
 *
 * @param[in] M0               The number of rows to store, for non-partial blocks. Supported: 1-16
 * @param[in] N0               The size of each vector, for non-partial blocks. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] DATA_TYPE        The data type of the vectors
 * @param[in] BASENAME         The basename of the variables
 * @param[in] PTR              The base pointer
 * @param[in] OFFSET           The offset within a row
 * @param[in] STRIDE_Y         The stride value in y-axis direction
 * @param[in] Z                The offset in z-axis direction
 * @param[in] PARTIAL_STORE_M0 The partial size in y, for partial blocks. Supported range: [1, @p M0)
 * @param[in] PARTIAL_COND_Y   Condition on the y axis to perform the partial store Y. True to use PARTIAL_STORE_M0 rather than M0.
 */
#define LOAD_BLOCK_PARTIAL_IN_Y(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_COND_Y) \
    if(!(PARTIAL_COND_Y))                                                                                                \
    {                                                                                                                    \
        LOAD_BLOCK_PARTIAL(M0, N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z);                                   \
    }                                                                                                                    \
    else                                                                                                                 \
    {                                                                                                                    \
        LOAD_BLOCK_PARTIAL(PARTIAL_STORE_M0, N0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z);                     \
    }
/** @} */ // end of group LOAD_BLOCK_PARTIAL
/** Boundary-aware GeMM block load
 * @name LOAD_BLOCK_BOUNDARY_AWARE
 * This macro assumes the following schemes to achieve boundary-awareness:
 *  - Overlapping load in Y axis from lhs tensor. This implies lhs has no padding along y dim.
 *  - Non-Overlapping(normal) load from rhs tensor. This imples rhs can have paddings.
 *  - Overlapping load in Y axis from bias tensor. This implies rhs has no padding along y dim.
 * The macro then ensures that the src tensor can be loaded without any paddings in both x and y dim.
 *
 * In the y dimension, we place the partial blocks **at the beginning** while in the x dimension, we place the partial
 * blocks **at the end**.
 * Say, the src tensor is of shape MxN and we have M0 and N0 as the block size, this is how we define "partial blocks"/
 * "boundary block" (we use the 2 terms "partial blocks" and "boundary blocks" interchangeably) and its various parameters:
 *
 *  *--x-->                         x == 0                        x == 1
 *  |                  |<------------------------------N-------------------------->|
 *  y                  |<--------------N0------------->|<----PARTIAL_STORE_N0----->|
 *  |     -------------#############################################################
 *  *     |          | |...............................|...........................|
 * y == 0 | PAR_..._M0 |......Boundary block in y......|.Boundary block in x and y.|
 *        |          | |...............................|...........................|
 *        M          --#############################################################
 *        |          | |                               |...........................|
 * y == 1 |         M0 |      Non-boundary block       |....Boundary block in x....|
 *        |          | |                               |...........................|
 *        |------------#############################################################
 *
 * Then @p PARTIAL_STORE_M0 = M % M0      and @p PARTIAL_STORE_N0 = N % N0
 *
 * @note in cases @p PARTIAL_STORE_N0 != 1, 2, 3, 4, 8, 16, extra vload(s) will be invoked, thus incurring small performance penalty.
 *
 * It automatically detects if a giving M,N,M0,N0 combination can yield partial blocks in either X and Y dimension,
 * and select corresponding load methods such that the boundary detection logic is only added when needed.
 *
 * The data to load is expected to have consecutive names for each row.
 * E.g., for M0=3 and basename=c, the expected names are c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for M0=3 and Z=zin, the expected z offset names are zin0, zin1 and zin2.
 *
 * The macro will result in a declaration of @p M0 vectors of size @p N0 with data
 * type @p DATA_TYPE containing values partially loaded from the specified
 * address in memory. The remaining (N0 - PARTIAL_STORE_N0) elements will be
 * filled with zeros.
 *
 * @param[in] M0               The number of rows to load, for non-partial blocks. Supported: 1-16
 * @param[in] N0               The size of each vector, for non-partial blocks. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] DATA_TYPE        The data type of the vectors
 * @param[in] BASENAME         The basename of the variables
 * @param[in] PTR              The base pointer
 * @param[in] OFFSET           The offset within a row
 * @param[in] STRIDE_Y         The stride value in y-axis direction
 * @param[in] Z                The offset in z-axis direction
 * @param[in] PARTIAL_STORE_M0 The partial size in y, for partial blocks. Supported: [0, @p M0)
 * @param[in] PARTIAL_STORE_N0 The partial size in x, for partial blocks. Supported: [0, @p N0)
 * @param[in] PARTIAL_COND_Y   Condition on the y axis to perform the partial load Y. True to use PARTIAL_STORE_M0 rather than M0.
 * @param[in] PARTIAL_COND_X   Condition on the x axis to perform the partial load X. True to use PARTIAL_STORE_N0 rather than N0.
 * @{
 */
#if PARTIAL_STORE_M0 == 0 && PARTIAL_STORE_N0 == 0
// Case1: No partial blocks in either x or y
#define LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    LOAD_BLOCK(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z)

#elif PARTIAL_STORE_M0 > 0 && PARTIAL_STORE_N0 == 0
// Case2: Partial blocks in y
#define LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), BASENAME, 0);                                                                                 \
    LOAD_BLOCK_PARTIAL_IN_Y(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_COND_Y)

#elif PARTIAL_STORE_M0 == 0 && PARTIAL_STORE_N0 > 0
// Case3: Partial blocks in x
#define LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), BASENAME, 0);                                                                                 \
    LOAD_BLOCK_PARTIAL_IN_X(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_N0, PARTIAL_COND_X)

#else // PARTIAL_STORE_M0 == 0 && PARTIAL_STORE_N0 == 0
// Case4: Partial blocks in both x and y
#define LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), BASENAME, 0);                                                                                 \
    LOAD_BLOCK_PARTIAL_IN_X_AND_Y(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X)

#endif // PARTIAL_STORE_M0 == 0 && PARTIAL_STORE_N0 == 0
/** @} */ // end of group LOAD_BLOCK_BOUNDARY_AWARE

/** Loads the rows from 0 to n-1 in the given variables (BASENAME0 to BASENAMEn-1).
 * @name LOAD_TEXTURE2D_ROW_n
 *
 * @param[in] N0         The number of pixels to read
 * @param[in] DATA_TYPE  The data type of variables
 * @param[in] BASENAME   The basename of the destination variables for the loaded rows
 * @param[in] IMG        The 2D OpenCL image object
 * @param[in] X_COORD    The x coordinate for the top-left pixel
 * @param[in] Y_COORD    The y coordinate for the top-left pixel
 * @param[in] X_STEP_ROW The incremental step row for the x coordinate (in pixels)
 * @param[in] Y_STEP_ROW The incremental step row for the y coordinate (in pixels)
 * @{
 */
#define LOAD_TEXTURE2D_ROW_1(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    BASENAME##0 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 0 * X_STEP_ROW), (Y_COORD + 0 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_2(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_1(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##1 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 1 * X_STEP_ROW), (Y_COORD + 1 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_3(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_2(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##2 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 2 * X_STEP_ROW), (Y_COORD + 2 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_4(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_3(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##3 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 3 * X_STEP_ROW), (Y_COORD + 3 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_5(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_4(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##4 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 4 * X_STEP_ROW), (Y_COORD + 4 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_6(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_5(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##5 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 5 * X_STEP_ROW), (Y_COORD + 5 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_7(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_6(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##6 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 6 * X_STEP_ROW), (Y_COORD + 6 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_8(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_7(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##7 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 7 * X_STEP_ROW), (Y_COORD + 7 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_9(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_8(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##8 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 8 * X_STEP_ROW), (Y_COORD + 8 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_10(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_9(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)      \
    BASENAME##9 = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 9 * X_STEP_ROW), (Y_COORD + 9 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_11(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_10(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##A = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 10 * X_STEP_ROW), (Y_COORD + 10 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_12(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_11(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##B = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 11 * X_STEP_ROW), (Y_COORD + 11 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_13(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_12(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##C = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 12 * X_STEP_ROW), (Y_COORD + 12 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_14(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_13(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##D = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 13 * X_STEP_ROW), (Y_COORD + 13 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_15(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_14(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##E = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 14 * X_STEP_ROW), (Y_COORD + 14 * Y_STEP_ROW))

#define LOAD_TEXTURE2D_ROW_16(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) \
    LOAD_TEXTURE2D_ROW_15(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)     \
    BASENAME##F = READ_IMAGE2D(DATA_TYPE, N0, IMG, (X_COORD + 15 * X_STEP_ROW), (Y_COORD + 15 * Y_STEP_ROW))
/** @} */ // end of group LOAD_TEXTURE2D_ROW_n

/** Load a 2D texture in unit of pixel. A pixel is made of 4 floating point values
 * @name LOAD_TEXTURE2D
 *
 * Supported cases are M0=1,2,3,...,16 and N0=1
 * The data to load is expected to have consecutive names for each row.
 * E.g., for M0=3, and BASENAME=c, the expected data is c0, c1 and c2.
 *
 * @param[in] M0         The number of consecutive rows
 * @param[in] N0         The number of consecutive pixels. Only 1, 2 and 4 are supported
 * @param[in] DATA_TYPE  The data type of the target
 * @param[in] BASENAME   The basename of the result variables
 * @param[in] IMG        The 2D OpenCL image object
 * @param[in] X_COORD    The x coordinate for the top-left pixel
 * @param[in] Y_COORD    The y coordinate for the top-left pixel
 * @param[in] X_STEP_ROW The incremental step row for the x coordinate (in pixels)
 * @param[in] Y_STEP_ROW The incremental step row for the y coordinate (in pixels)
 * @{
 */
#define LOAD_TEXTURE2D_STR(M0, N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) LOAD_TEXTURE2D_ROW_##M0(N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)
#define LOAD_TEXTURE2D(M0, N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW) LOAD_TEXTURE2D_STR(M0, N0, DATA_TYPE, BASENAME, IMG, X_COORD, Y_COORD, X_STEP_ROW, Y_STEP_ROW)
/** @} */ // end of group LOAD_TEXTURE2D

/** Loads the rows from 0 to n-1 in the given variables (BASENAME0 to BASENAMEn-1) passing the Y index for each row to be loaded.
 * @name LOAD_ROW_INDIRECT_n
 *
 * @param[in] N0        The number of columns to load
 * @param[in] DATA_TYPE The data type of variables
 * @param[in] BASENAME  The basename of the destination variables for the loaded rows
 * @param[in] PTR       The base pointer
 * @param[in] OFFSET    The offset within a row
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Y         The y-axis offset vector
 * @param[in] Y_MASK    The y-axis mask vector. If 0, forces BASENAMEn to 0
 * @{
 */
#define LOAD_ROW_INDIRECT_1(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)      \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##0;                                                                            \
    if(Y_MASK##0 != 0)                                                                      \
        BASENAME##0 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##0 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##0 = 0;

#define LOAD_ROW_INDIRECT_2(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)      \
    LOAD_ROW_INDIRECT_1(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)          \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##1;                                                                            \
    if(Y_MASK##1 != 0)                                                                      \
        BASENAME##1 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##1 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##1 = 0;

#define LOAD_ROW_INDIRECT_3(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)      \
    LOAD_ROW_INDIRECT_2(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)          \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##2;                                                                            \
    if(Y_MASK##2 != 0)                                                                      \
        BASENAME##2 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##2 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##2 = 0;

#define LOAD_ROW_INDIRECT_4(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)      \
    LOAD_ROW_INDIRECT_3(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)          \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##3;                                                                            \
    if(Y_MASK##3 != 0)                                                                      \
        BASENAME##3 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##3 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##3 = 0;

#define LOAD_ROW_INDIRECT_5(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)      \
    LOAD_ROW_INDIRECT_4(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)          \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##4;                                                                            \
    if(Y_MASK##4 != 0)                                                                      \
        BASENAME##4 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##4 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##4 = 0;

#define LOAD_ROW_INDIRECT_6(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)      \
    LOAD_ROW_INDIRECT_5(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)          \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##5;                                                                            \
    if(Y_MASK##5 != 0)                                                                      \
        BASENAME##5 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##5 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##5 = 0;

#define LOAD_ROW_INDIRECT_7(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)      \
    LOAD_ROW_INDIRECT_6(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)          \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##6;                                                                            \
    if(Y_MASK##6 != 0)                                                                      \
        BASENAME##6 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##6 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##6 = 0;

#define LOAD_ROW_INDIRECT_8(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)      \
    LOAD_ROW_INDIRECT_7(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)          \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##7;                                                                            \
    if(Y_MASK##7 != 0)                                                                      \
        BASENAME##7 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##7 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##7 = 0;

#define LOAD_ROW_INDIRECT_9(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)      \
    LOAD_ROW_INDIRECT_8(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)          \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##8;                                                                            \
    if(Y_MASK##8 != 0)                                                                      \
        BASENAME##8 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##8 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##8 = 0;

#define LOAD_ROW_INDIRECT_10(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)     \
    LOAD_ROW_INDIRECT_9(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)          \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##9;                                                                            \
    if(Y_MASK##9 != 0)                                                                      \
        BASENAME##9 = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##9 * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##9 = 0;

#define LOAD_ROW_INDIRECT_11(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)     \
    LOAD_ROW_INDIRECT_10(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)         \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##A;                                                                            \
    if(Y_MASK##A != 0)                                                                      \
        BASENAME##A = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##A * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##A = 0;

#define LOAD_ROW_INDIRECT_12(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)     \
    LOAD_ROW_INDIRECT_11(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)         \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##B;                                                                            \
    if(Y_MASK##B != 0)                                                                      \
        BASENAME##B = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##B * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##B = 0;

#define LOAD_ROW_INDIRECT_13(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)     \
    LOAD_ROW_INDIRECT_12(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)         \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##C;                                                                            \
    if(Y_MASK##C != 0)                                                                      \
        BASENAME##C = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##C * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##C = 0;

#define LOAD_ROW_INDIRECT_14(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)     \
    LOAD_ROW_INDIRECT_13(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)         \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##D;                                                                            \
    if(Y_MASK##D != 0)                                                                      \
        BASENAME##D = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##D * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##D = 0;

#define LOAD_ROW_INDIRECT_15(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)     \
    LOAD_ROW_INDIRECT_14(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)         \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##E;                                                                            \
    if(Y_MASK##E != 0)                                                                      \
        BASENAME##E = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##E * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##E = 0;

#define LOAD_ROW_INDIRECT_16(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)     \
    LOAD_ROW_INDIRECT_15(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)         \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                                            \
    BASENAME##F;                                                                            \
    if(Y_MASK##F != 0)                                                                      \
        BASENAME##F = VLOAD(N0)(0, (__global DATA_TYPE *)(PTR + OFFSET + Y##F * STRIDE_Y)); \
    else                                                                                    \
        BASENAME##F = 0;
/** @} */ // end of group LOAD_ROW_INDIRECT_n

/** Load blocks (consecutive rows and columns) with Y offset.
 * @name LOAD_BLOCK_INDIRECT
 *
 * Supported cases are M0=1,2,3,...,16 and N0=1,2,3,4,8,16
 * The data to load is expected to have consecutive names for each row.
 * E.g., for M0=3, and BASENAME=c, the expected data is c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for M0=3, and Z=zin, the expected Z offsets are zin0, zin1 and zin2.
 *
 * @param[in] M0        The number of consecutive rows
 * @param[in] N0        The number of consecutive columns
 * @param[in] DATA_TYPE The data type of the target
 * @param[in] BASENAME  The basename of the result variables
 * @param[in] PTR       The base pointer for the data
 * @param[in] OFFSET    The offset within a row
 * @param[in] STRIDE_Y  The stride in y-axis direction
 * @param[in] Y         The y-axis offset vector
 * @param[in] Y_MASK    The y-axis mask vector. If 0, forces BASENAMEn to 0
 * @{
 */
#define LOAD_BLOCK_INDIRECT_STR(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK) LOAD_ROW_INDIRECT_##M0(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)
#define LOAD_BLOCK_INDIRECT(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK) LOAD_BLOCK_INDIRECT_STR(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y, Y, Y_MASK)
/** @} */ // end of group LOAD_BLOCK_INDIRECT

/** Loads the elements from 0 to n-1 in the given variables (BASENAME0 to BASENAMEn-1).
 * @name LOAD_ELEMENT_n
 *
 * @param[in] N0        The number of rows to load
 * @param[in] DATA_TYPE The data type of variables
 * @param[in] BASENAME  The basename of the destination variables for the loaded rows
 * @param[in] PTR       The base pointer
 * @param[in] OFFSET    The offset within a row
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @{
 */
#define LOAD_ELEMENT_1(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##0 = *((__global DATA_TYPE *)(PTR + OFFSET + 0 * STRIDE_Y));

#define LOAD_ELEMENT_2(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_1(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##1 = *((__global DATA_TYPE *)(PTR + OFFSET + 1 * STRIDE_Y));

#define LOAD_ELEMENT_3(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_2(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##2 = *((__global DATA_TYPE *)(PTR + OFFSET + 2 * STRIDE_Y));

#define LOAD_ELEMENT_4(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_3(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##3 = *((__global DATA_TYPE *)(PTR + OFFSET + 3 * STRIDE_Y));

#define LOAD_ELEMENT_5(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_4(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##4 = *((__global DATA_TYPE *)(PTR + OFFSET + 4 * STRIDE_Y));

#define LOAD_ELEMENT_6(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_5(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##5 = *((__global DATA_TYPE *)(PTR + OFFSET + 5 * STRIDE_Y));

#define LOAD_ELEMENT_7(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_6(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##6 = *((__global DATA_TYPE *)(PTR + OFFSET + 6 * STRIDE_Y));

#define LOAD_ELEMENT_8(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_7(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##7 = *((__global DATA_TYPE *)(PTR + OFFSET + 7 * STRIDE_Y));

#define LOAD_ELEMENT_9(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_8(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                       \
    BASENAME##8 = *((__global DATA_TYPE *)(PTR + OFFSET + 8 * STRIDE_Y));

#define LOAD_ELEMENT_10(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_9(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)      \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                        \
    BASENAME##9 = *((__global DATA_TYPE *)(PTR + OFFSET + 9 * STRIDE_Y));

#define LOAD_ELEMENT_11(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_10(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                        \
    BASENAME##A = *((__global DATA_TYPE *)(PTR + OFFSET + 10 * STRIDE_Y));

#define LOAD_ELEMENT_12(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_11(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                        \
    BASENAME##B = *((__global DATA_TYPE *)(PTR + OFFSET + 11 * STRIDE_Y));

#define LOAD_ELEMENT_13(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_12(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                        \
    BASENAME##C = *((__global DATA_TYPE *)(PTR + OFFSET + 12 * STRIDE_Y));

#define LOAD_ELEMENT_14(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_13(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                        \
    BASENAME##D = *((__global DATA_TYPE *)(PTR + OFFSET + 13 * STRIDE_Y));

#define LOAD_ELEMENT_15(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_14(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                        \
    BASENAME##E = *((__global DATA_TYPE *)(PTR + OFFSET + 14 * STRIDE_Y));

#define LOAD_ELEMENT_16(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) \
    LOAD_ELEMENT_15(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)     \
    VEC_DATA_TYPE(DATA_TYPE, N0)                                        \
    BASENAME##F = *((__global DATA_TYPE *)(PTR + OFFSET + 15 * STRIDE_Y));

/** @}*/ // end of group LOAD_ELEMENT_n

/** Load Scalar as Vector (consecutive elements).
 * @name LOAD_SCALAR_AS_VECTOR
 *
 * Supported cases are M0=1,2,3,...,16 and N0=1,2,3,4,8,16
 * The data to load is expected to have consecutive names for each row.
 * E.g., for M0=3, and BASENAME=c, the expected data is c0, c1 and c2.
 *
 * @param[in] M0        The number of consecutive rows
 * @param[in] N0        The number of consecutive columns
 * @param[in] DATA_TYPE The data type of the target
 * @param[in] BASENAME  The basename of the result variables
 * @param[in] PTR       The base pointer for the data
 * @param[in] OFFSET    The offset within a row
 * @param[in] STRIDE_Y  The stride in y-axis direction
 * @{
 */
#define LOAD_SCALAR_AS_VECTOR_STR(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) LOAD_ELEMENT_##M0(N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)
#define LOAD_SCALAR_AS_VECTOR(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y) LOAD_SCALAR_AS_VECTOR_STR(M0, N0, DATA_TYPE, BASENAME, PTR, OFFSET, STRIDE_Y)
/** @} */ // end of group LOAD_SCALAR_AS_VECTOR

/** Basic macros to calculate Z offset values from Z0 to Zn-1
 * @name CALCULATE_Z_OFFSET_n
 *
 * @param[in] M0              The number of offset values to calculate
 * @param[in] DATA_TYPE       The data type of the results
 * @param[in] Z               The basename of the result variables
 * @param[in] Y               The work-itme ID of y-axis
 * @param[in] HEIGHT_GEMM3D   The height of GEMM3D
 * @param[in] DEPTH_GEMM3D    The depth of GEMM3D
 * @param[in] CROSS_PLANE_PAD The padding required for plane changes accross the z-dimension
 * @param[in] STRIDE_Y        The stride value in y-axis direction
 *
 * @{
 */
#define CALCULATE_Z_OFFSET_1(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    Z##0 = (0 + (DATA_TYPE)(Y)) / (DATA_TYPE)HEIGHT_GEMM3D;                                               \
    Z##0 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##0);                                                      \
    Z##0 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_2(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_1(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##1 = (1 + (DATA_TYPE)(Y)) / (DATA_TYPE)HEIGHT_GEMM3D;                                               \
    Z##1 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##1);                                                      \
    Z##1 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_3(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_2(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##2 = (2 + (DATA_TYPE)(Y)) / (DATA_TYPE)HEIGHT_GEMM3D;                                               \
    Z##2 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##2);                                                      \
    Z##2 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_4(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_3(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##3 = (3 + (DATA_TYPE)(Y)) / (DATA_TYPE)HEIGHT_GEMM3D;                                               \
    Z##3 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##3);                                                      \
    Z##3 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_5(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_4(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##4 = (4 + (DATA_TYPE)(Y)) / (DATA_TYPE)HEIGHT_GEMM3D;                                               \
    Z##4 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##4);                                                      \
    Z##4 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_6(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_5(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##5 = (5 + (DATA_TYPE)(Y)) / (DATA_TYPE)HEIGHT_GEMM3D;                                               \
    Z##5 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##5);                                                      \
    Z##5 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_7(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_6(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##6 = (6 + (DATA_TYPE)(Y)) / (DATA_TYPE)HEIGHT_GEMM3D;                                               \
    Z##6 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##6);                                                      \
    Z##6 *= (CROSS_PLANE_PAD * STRIDE_Y);

#define CALCULATE_Z_OFFSET_8(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) \
    CALCULATE_Z_OFFSET_7(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)     \
    Z##7 = (7 + (DATA_TYPE)(Y)) / (DATA_TYPE)HEIGHT_GEMM3D;                                               \
    Z##7 = min((DATA_TYPE)(DEPTH_GEMM3D - 1), Z##7);                                                      \
    Z##7 *= (CROSS_PLANE_PAD * STRIDE_Y);

/** @} */ // end of group CALCULATE_Z_OFFSET_n

/** Calculate Z offset values from Z0 to Zn-1
 * @name CALCULATE_Z_OFFSET
 *
 * The Z offsets are expected to have consecutive names.
 * E.g., for M0=3 and Z=zin, the expected names of Z offsets are zin1, zin2, zin3.
 * Note that, CROSS_PLANE_PAD (cross plain padding) is required to take into account
 * the possible cross plane paddings in case of the plance changes across the z-dimension.
 *
 * <!--
 * |                  |
 * |      plane0      |
 * |                  |
 * |__________________|
 * |******************|
 * |  cross_plane_pad |
 * |******************|
 * |                  |
 * |      plane1      |
 * |                  |
 * |__________________|
 * -->
 *
 * @param[in] M0              The number of offset values to calculate
 * @param[in] DATA_TYPE       The data type of the results
 * @param[in] Z               The basename of the result variables
 * @param[in] Y               The work-itme ID of y-axis
 * @param[in] HEIGHT_GEMM3D   The height of GEMM3D
 * @param[in] DEPTH_GEMM3D    The depth of GEMM3D
 * @param[in] CROSS_PLANE_PAD The padding required for plane changes accross the z-dimension
 * @param[in] STRIDE_Y        The stride value in y-axis direction
 * @{
 */
#define CALCULATE_Z_OFFSET_STR(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) CALCULATE_Z_OFFSET_##M0(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)
#define CALCULATE_Z_OFFSET(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y) CALCULATE_Z_OFFSET_STR(M0, DATA_TYPE, Z, Y, HEIGHT_GEMM3D, DEPTH_GEMM3D, CROSS_PLANE_PAD, STRIDE_Y)
/** @} */ // end of group CALCULATE_Z_OFFSET

/** Scale the rows in the given variables (BASENAME0 to BASENAMEn-1)
 * @name SCALE_ROW_n
 *
 * @param[in] DATA_TYPE The data type of the variables
 * @param[in] BASENAME  The basename of the variables
 * @param[in] SCALE     The scale factor
 * @{
 */
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
/** @} */ // end of group SCALE_ROW_n

/** Scale elements stored in a block (BASENAME)
 * @name SCALE_BLOCK
 *
 * Supported cases are N=1,2,3,...,16
 *
 * @param[in] N         The number of rows in the block
 * @param[in] DATA_TYPE The data type of the block
 * @param[in] BASENAME  The basename of the block
 * @param[in] SCALE     The scale factor
 * @{
 */
#define SCALE_BLOCK_STR(N, DATA_TYPE, BASENAME, SCALE) SCALE_ROW_##N(DATA_TYPE, BASENAME, SCALE)
#define SCALE_BLOCK(N, DATA_TYPE, BASENAME, SCALE) SCALE_BLOCK_STR(N, DATA_TYPE, BASENAME, SCALE)
/** @} */ // end of group SCALE_BLOCK

/** Create a new vector containing the values at the given index for a set of given vectors
 * @name COLUMN_VECTORn
 *
 * @param[in] IDX_COL  The index value
 * @param[in] BASENAME The basename of the destination vectors
 * @param[in] X        The basename of the source vectors
 * @param[in] TYPE     The data type of the destination vectors
 * @{
 */
#define COLUMN_VECTOR1(IDX_COL, BASENAME, X, TYPE) \
    TYPE BASENAME##IDX_COL = (TYPE)((X##0).s##IDX_COL);
#define COLUMN_VECTOR2(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 2)                         \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 2))((X##0).s##IDX_COL, (X##1).s##IDX_COL);
#define COLUMN_VECTOR3(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 3)                         \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 3))((X##0).s##IDX_COL, (X##1).s##IDX_COL, (X##2).s##IDX_COL);
#define COLUMN_VECTOR4(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 4)                         \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 4))((X##0).s##IDX_COL, (X##1).s##IDX_COL, (X##2).s##IDX_COL, (X##3).s##IDX_COL);
#define COLUMN_VECTOR8(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 8)                         \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 8))((X##0).s##IDX_COL, (X##1).s##IDX_COL, (X##2).s##IDX_COL, (X##3).s##IDX_COL, (X##4).s##IDX_COL, (X##5).s##IDX_COL, (X##6).s##IDX_COL, (X##7).s##IDX_COL);
#define COLUMN_VECTOR16(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 16)                         \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 16))((X##0).s##IDX_COL, (X##1).s##IDX_COL, (X##2).s##IDX_COL, (X##3).s##IDX_COL, (X##4).s##IDX_COL, (X##5).s##IDX_COL, (X##6).s##IDX_COL, (X##7).s##IDX_COL, (X##8).s##IDX_COL, (X##9).s##IDX_COL, (X##A).s##IDX_COL, (X##B).s##IDX_COL, (X##C).s##IDX_COL, (X##D).s##IDX_COL, (X##E).s##IDX_COL, (X##F).s##IDX_COL);
/** @} */ // end of group COLUMN_VECTORn

/** Create a new vector containing the values at the given index. Utility macros for transposing a colum-vector
 * @name COLUMN_VECTOR_SCALARn
 *
 * @param[in] IDX_COL  The index value
 * @param[in] BASENAME The basename of the destination vectors
 * @param[in] X        The basename of the source vectors
 * @param[in] TYPE     The data type of the destination vectors
 * @{
 */
#define COLUMN_VECTOR_SCALAR1(IDX_COL, BASENAME, X, TYPE) \
    TYPE BASENAME##IDX_COL = (TYPE)((X##0));
#define COLUMN_VECTOR_SCALAR2(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 2)                                \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 2))((X##0), (X##1));
#define COLUMN_VECTOR_SCALAR3(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 3)                                \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 3))((X##0), (X##1), (X##2));
#define COLUMN_VECTOR_SCALAR4(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 4)                                \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 4))((X##0), (X##1), (X##2), (X##3));
#define COLUMN_VECTOR_SCALAR8(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 8)                                \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 8))((X##0), (X##1), (X##2), (X##3), (X##4), (X##5), (X##6), (X##7));
#define COLUMN_VECTOR_SCALAR16(IDX_COL, BASENAME, X, TYPE) \
    VEC_DATA_TYPE(TYPE, 16)                                \
    BASENAME##IDX_COL = (VEC_DATA_TYPE(TYPE, 16))((X##0), (X##1), (X##2), (X##3), (X##4), (X##5), (X##6), (X##7), (X##8), (X##9), (X##A), (X##B), (X##C), (X##D), (X##E), (X##F));
/** @} */ // end of group COLUMN_VECTOR_SCALARn

/** Create transposed vectors of the given vectors
 * @name TRANSPOSE_K0Xn
 *
 * @param[in] K0       The size of the source vectors
 * @param[in] BASENAME The basename of transposed vectors
 * @param[in] BS       The basename of source vectors for transposition
 * @param[in] TYPE     The data type of the transposed vectors
 * @{
 */
#define TRANSPOSE_K0X1(K0, BASENAME, BS, TYPE) \
    COLUMN_VECTOR_SCALAR(K0, 0, BASENAME, BS, TYPE);
#define TRANSPOSE_K0X2(K0, BASENAME, BS, TYPE) \
    COLUMN_VECTOR(K0, 0, BASENAME, BS, TYPE);  \
    COLUMN_VECTOR(K0, 1, BASENAME, BS, TYPE);
#define TRANSPOSE_K0X3(K0, BASENAME, BS, TYPE) \
    TRANSPOSE_K0X2(K0, BASENAME, BS, TYPE);    \
    COLUMN_VECTOR(K0, 2, BASENAME, BS, TYPE);
#define TRANSPOSE_K0X4(K0, BASENAME, BS, TYPE) \
    TRANSPOSE_K0X3(K0, BASENAME, BS, TYPE);    \
    COLUMN_VECTOR(K0, 3, BASENAME, BS, TYPE);
#define TRANSPOSE_K0X8(K0, BASENAME, BS, TYPE) \
    TRANSPOSE_K0X4(K0, BASENAME, BS, TYPE);    \
    COLUMN_VECTOR(K0, 4, BASENAME, BS, TYPE);  \
    COLUMN_VECTOR(K0, 5, BASENAME, BS, TYPE);  \
    COLUMN_VECTOR(K0, 6, BASENAME, BS, TYPE);  \
    COLUMN_VECTOR(K0, 7, BASENAME, BS, TYPE);
#define TRANSPOSE_K0X16(K0, BASENAME, BS, TYPE) \
    TRANSPOSE_K0X8(K0, BASENAME, BS, TYPE);     \
    COLUMN_VECTOR(K0, 8, BASENAME, BS, TYPE);   \
    COLUMN_VECTOR(K0, 9, BASENAME, BS, TYPE);   \
    COLUMN_VECTOR(K0, A, BASENAME, BS, TYPE);   \
    COLUMN_VECTOR(K0, B, BASENAME, BS, TYPE);   \
    COLUMN_VECTOR(K0, C, BASENAME, BS, TYPE);   \
    COLUMN_VECTOR(K0, D, BASENAME, BS, TYPE);   \
    COLUMN_VECTOR(K0, E, BASENAME, BS, TYPE);   \
    COLUMN_VECTOR(K0, F, BASENAME, BS, TYPE);

/** @} */ // end of group TRANSPOSE_K0Xn

/** Create column vectors to contain the values at the given index for a set of given vectors
 *
 * @param[in] K0       The number of source vectors
 * @param[in] IDX_COL  The index value
 * @param[in] BASENAME The basename of the destination vectors
 * @param[in] BS       The basename of the source vectors
 * @param[in] TYPE     The data type of the destination vectors
 */
#define COLUMN_VECTOR(K0, IDX_COL, BASENAME, BS, TYPE) \
    CONCAT(COLUMN_VECTOR, K0)                          \
    (IDX_COL, BASENAME, BS, TYPE);

/** Create column vectors to contain the values at the given index. Utility macro for transposing a column-vector
 *
 * @param[in] K0       The number of source vectors
 * @param[in] IDX_COL  The index value
 * @param[in] BASENAME The basename of the destination vectors
 * @param[in] BS       The basename of the source vectors
 * @param[in] TYPE     The data type of the destination vectors
 */
#define COLUMN_VECTOR_SCALAR(K0, IDX_COL, BASENAME, BS, TYPE) \
    CONCAT(COLUMN_VECTOR_SCALAR, K0)                          \
    (IDX_COL, BASENAME, BS, TYPE);

/** Create transposed vectors form the given source vectors
 *
 * @param[in] K0       The size of source vectors
 * @param[in] N0       The number of source vectors
 * @param[in] BASENAME The basename of transposed vectors
 * @param[in] BS       The basename of source vectors for transposition
 * @param[in] TYPE     The data type of the transposed vectors
 *
 */
#define TRANSPOSE_K0XN0(K0, N0, BASENAME, BS, TYPE) \
    CONCAT(TRANSPOSE_K0X, N0)                       \
    (K0, BASENAME, BS, TYPE);

/** Add the variables (BIAS0 to BIASn-1) to the others (BASENAME0 to BASENAMEn-1)
 * @name ADD_ROW_n
 *
 * @param[in] BASENAME The basename of the destination variables
 * @param[in] BIAS     The basename of the added variables
 * @{
 */
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

/** @} */ // end of group ADD_ROW_n

/** Add the block (BIAS) to another block (BASENAME)
 * @name ADD_BLOCK
 *
 * Supported cases are N=1,2,3,...,16
 *
 * @param[in] N        The number of vectors in the block
 * @param[in] BASENAME The basename of the destination variables
 * @param[in] BIAS     The basename of the added variables
 * @{
 */
#define ADD_BLOCK_STR(N, BASENAME, BIAS) ADD_ROW_##N(BASENAME, BIAS)
#define ADD_BLOCK(N, BASENAME, BIAS) ADD_BLOCK_STR(N, BASENAME, BIAS)
/** @} */ // end of group ADD_BLOCK

/** Broadcast (add single value) to the each element of the destination variables
 * @name ADD_ROW_BROADCAST_n
 *
 * @param[in] BASENAME The basename of the destination variables
 * @param[in] BIAS     The variable containing the value to add
 * @{
 */
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
/** @} */ // end of group ADD_ROW_BROADCAST_n

/** Broadcast (add a value) to the each element of the destination block (BASENAME)
 * @name ADD_BLOCK_BROADCAST
 *
 * Supported cases are N=1,2,3,...,16.
 *
 * @param[in] N        The number of vectors in the block
 * @param[in] BASENAME The basename of the destination variables
 * @param[in] BIAS     The variable containing the value to add
 * @{
 */
#define ADD_BLOCK_BROADCAST_STR(N, BASENAME, BIAS) ADD_ROW_BROADCAST_##N(BASENAME, BIAS)
#define ADD_BLOCK_BROADCAST(N, BASENAME, BIAS) ADD_BLOCK_BROADCAST_STR(N, BASENAME, BIAS)
/** @} */ // end of group ADD_BLOCK_BROADCAST

/** Apply activation to the given variables
 * @name ACTIVATION_ROW_n
 *
 * @param[in] ACTIVATION_TYPE The type of the activation
 * @param[in] DATA_TYPE       The data type of the vectors
 * @param[in] BASENAME        The basename of the variables
 * @param[in] A_VAL           Additional value required by the activation
 * @param[in] B_VAL           Additional value required by the activation
 * @{
 */
#define ACTIVATION_ROW_1(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    BASENAME##0 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##0, A_VAL, B_VAL);

#define ACTIVATION_ROW_2(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_1(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##1 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##1, A_VAL, B_VAL);

#define ACTIVATION_ROW_3(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_2(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##2 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##2, A_VAL, B_VAL);

#define ACTIVATION_ROW_4(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_3(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##3 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##3, A_VAL, B_VAL);

#define ACTIVATION_ROW_5(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_4(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##4 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##4, A_VAL, B_VAL);

#define ACTIVATION_ROW_6(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_5(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##5 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##5, A_VAL, B_VAL);

#define ACTIVATION_ROW_7(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_6(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##6 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##6, A_VAL, B_VAL);

#define ACTIVATION_ROW_8(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_7(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##7 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##7, A_VAL, B_VAL);

#define ACTIVATION_ROW_9(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_8(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##8 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##8, A_VAL, B_VAL);

#define ACTIVATION_ROW_10(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_9(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)      \
    BASENAME##9 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##9, A_VAL, B_VAL);

#define ACTIVATION_ROW_11(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_10(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##A = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##A, A_VAL, B_VAL);

#define ACTIVATION_ROW_12(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_11(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##B = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##B, A_VAL, B_VAL);

#define ACTIVATION_ROW_13(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_12(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##C = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##C, A_VAL, B_VAL);

#define ACTIVATION_ROW_14(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_13(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##D = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##D, A_VAL, B_VAL);

#define ACTIVATION_ROW_15(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_14(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##E = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##E, A_VAL, B_VAL);

#define ACTIVATION_ROW_16(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) \
    ACTIVATION_ROW_15(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)     \
    BASENAME##F = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME##F, A_VAL, B_VAL);
/** @} */ // end of group ACTIVATION_ROW_n

/** Apply activation to a block (BASENAME)
 * @name ACTIVATION_BLOCK
 *
 * Supported cases are N=1,2,3,...,16.
 *
 * @param[in] N               The number of vectors in the block
 * @param[in] ACTIVATION_TYPE The type of the activation
 * @param[in] DATA_TYPE       The data type of the vectors
 * @param[in] BASENAME        The basename of the variables
 * @param[in] A_VAL           Additional value required by the activation
 * @param[in] B_VAL           Additional value required by the activation
 * @{
 */
#define ACTIVATION_BLOCK_STR(N, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) ACTIVATION_ROW_##N(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)
#define ACTIVATION_BLOCK(N, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL) ACTIVATION_BLOCK_STR(N, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, BASENAME, A_VAL, B_VAL)
/** @} */ // end of group ACTIVATION_BLOCK

/** Apply convert_<data_type> to the given variables
 * @name CONVERT_ROW_n
 *
 * @param[in] N            The size of the vectors
 * @param[in] DATA_TYPE    The data type of the vectors
 * @param[in] BASENAME_SRC The basename of the source variables
 * @param[in] BASENAME_DST The basename of the destination variables
 * @{
 */
#define CONVERT_ROW_1(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    VEC_DATA_TYPE(DATA_TYPE, N)                                 \
    BASENAME_DST##0 = CONVERT(BASENAME_SRC##0, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_2(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_1(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                 \
    BASENAME_DST##1 = CONVERT(BASENAME_SRC##1, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_3(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_2(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                 \
    BASENAME_DST##2 = CONVERT(BASENAME_SRC##2, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_4(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_3(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                 \
    BASENAME_DST##3 = CONVERT(BASENAME_SRC##3, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_5(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_4(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                 \
    BASENAME_DST##4 = CONVERT(BASENAME_SRC##4, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_6(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_5(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                 \
    BASENAME_DST##5 = CONVERT(BASENAME_SRC##5, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_7(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_6(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                 \
    BASENAME_DST##6 = CONVERT(BASENAME_SRC##6, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_8(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_7(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                 \
    BASENAME_DST##7 = CONVERT(BASENAME_SRC##7, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_9(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_8(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                 \
    BASENAME_DST##8 = CONVERT(BASENAME_SRC##8, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_10(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_9(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)      \
    VEC_DATA_TYPE(DATA_TYPE, N)                                  \
    BASENAME_DST##9 = CONVERT(BASENAME_SRC##9, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_11(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_10(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                  \
    BASENAME_DST##A = CONVERT(BASENAME_SRC##A, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_12(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_11(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                  \
    BASENAME_DST##B = CONVERT(BASENAME_SRC##B, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_13(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_12(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                  \
    BASENAME_DST##C = CONVERT(BASENAME_SRC##C, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_14(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_13(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                  \
    BASENAME_DST##D = CONVERT(BASENAME_SRC##D, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_15(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_14(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                  \
    BASENAME_DST##E = CONVERT(BASENAME_SRC##E, VEC_DATA_TYPE(DATA_TYPE, N));

#define CONVERT_ROW_16(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) \
    CONVERT_ROW_15(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)     \
    VEC_DATA_TYPE(DATA_TYPE, N)                                  \
    BASENAME_DST##F = CONVERT(BASENAME_SRC##F, VEC_DATA_TYPE(DATA_TYPE, N));
/** @} */ // end of group CONVERT_ROW_n

/** Apply convert_<data_type> to a block (BASENAME_SRC) and save to another block (BASENAME_DST)
 * @name CONVERT_BLOCK
 *
 * Supported cases N=1,2,3,...,16.
 *
 * @param[in] M            The number of vectors to convert
 * @param[in] N            The size of the vectors
 * @param[in] DATA_TYPE    The data type of the vectors
 * @param[in] BASENAME_SRC The basename of the source variables
 * @param[in] BASENAME_DST The basename of the destination variables
 * @{
 */
#define CONVERT_BLOCK_STR(M, N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) CONVERT_ROW_##M(N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)
#define CONVERT_BLOCK(M, N, DATA_TYPE, BASENAME_SRC, BASENAME_DST) CONVERT_BLOCK_STR(M, N, DATA_TYPE, BASENAME_SRC, BASENAME_DST)
/** @} */ // end of group CONVERT_BLOCK
