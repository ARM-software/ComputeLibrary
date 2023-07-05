/*
 * Copyright (c) 2020, 2023 Arm Limited.
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

/** Store the 0 to (n-1)th rows of the given variables
 * @name STORE_ROW_n
 *
 * @param[in] N0        The width of the passed in vector. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] DATA_TYPE The data type of the vectors
 * @param[in] BASENAME  The basename of the variables
 * @param[in] PTR       The base pointer
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The offset in z-axis direction
 * @{
 */
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
/** @} */ // end of groupd STORE_ROW_n

/** Convert and store the 0th to (n-1)th rows of the given variables
 * @name CONVERT_STORE_ROW_n
 *
 * @param[in] N0        The size of the vectors
 * @param[in] DATA_TYPE The data type of the vectors
 * @param[in] BASENAME  The basename of the variables
 * @param[in] PTR       The base pointer
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The offset in z-axis direction
 * @{
 */
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

/** @} */ // end of groupd CONVERT_STORE_ROW_n

/** Store a block of the given size M0xN0
 * @name STORE_BLOCK
 *
 * Supported cases are M0=1,2,3,...,16 and N0=2,3,4,8,16.
 * The data to store is expected to have consecutive names for each row.
 * E.g., for M0=3 and basename=c, the expected names are c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for M0=3 and Z=zin, the expected z offset names are zin0, zin1 and zin2.
 *
 * @param[in] M0        The number of rows to store
 * @param[in] N0        The size of each vector
 * @param[in] DATA_TYPE The data type of the vectors
 * @param[in] BASENAME  The basename of the variables
 * @param[in] PTR       The base pointer
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The offset in z-axis direction
 * @{
 */
#define STORE_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) STORE_ROW_##M0(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)
#define STORE_BLOCK(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) STORE_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)
/** @} */ // end of group STORE_BLOCK

/** Convert and store a block of the given size M0xN0
 * @name CONVERT_STORE_BLOCK
 *
 * Supported cases are M0=1,2,3,...,16 and N0=2,3,4,8,16.
 * The data to store is expected to have consecutive names for each row.
 * E.g., for M0=3 and basename=c, the expected names are c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for M0=3 and Z=zin, the expected z offset names are zin0, zin1 and zin2.
 *
 * @param[in] M0        The number of rows to store
 * @param[in] N0        The size of each vector
 * @param[in] DATA_TYPE The data type of the vectors
 * @param[in] BASENAME  The basename of the variables
 * @param[in] PTR       The base pointer
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The offset in z-axis direction
 * @{
 */
#define CONVERT_STORE_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) CONVERT_STORE_ROW_##M0(N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)
#define CONVERT_STORE_BLOCK(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) CONVERT_STORE_BLOCK_STR(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)
/** @} */ // end of group CONVERT_STORE_BLOCK

/** Partially store the 0 to (n-1)th rows of the given variables
 * @name STORE_ROW_PARTIAL_n
 * Within each row, store the lower @p STORE_N0 elements of vectors of width @p N0
 *
 * @note in case @p STORE_N0 != 1, 2, 3, 4, 8, 16, extra vstore(s) will be invoked, thus incurring small performance penalty.
 *
 * @param[in] N0        The width of the passed in vector. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] STORE_N0  The **lower** size of the vectors to store. Supported: [1-16 and <= @p N0
 * @param[in] DATA_TYPE The data type of the vectors
 * @param[in] BASENAME  The basename of the variables
 * @param[in] PTR       The base pointer
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The offset in z-axis direction
 * @{
 */
#define STORE_ROW_PARTIAL_1(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    VSTORE_PARTIAL(N0, STORE_N0)                                                 \
    (BASENAME##0, 0, (__global DATA_TYPE *)(PTR + 0 * STRIDE_Y + Z##0));

#define STORE_ROW_PARTIAL_2(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_1(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                 \
    (BASENAME##1, 0, (__global DATA_TYPE *)(PTR + 1 * STRIDE_Y + Z##1));

#define STORE_ROW_PARTIAL_3(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_2(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                 \
    (BASENAME##2, 0, (__global DATA_TYPE *)(PTR + 2 * STRIDE_Y + Z##2));

#define STORE_ROW_PARTIAL_4(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_3(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                 \
    (BASENAME##3, 0, (__global DATA_TYPE *)(PTR + 3 * STRIDE_Y + Z##3));

#define STORE_ROW_PARTIAL_5(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_4(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                 \
    (BASENAME##4, 0, (__global DATA_TYPE *)(PTR + 4 * STRIDE_Y + Z##4));

#define STORE_ROW_PARTIAL_6(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_5(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                 \
    (BASENAME##5, 0, (__global DATA_TYPE *)(PTR + 5 * STRIDE_Y + Z##5));

#define STORE_ROW_PARTIAL_7(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_6(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                 \
    (BASENAME##6, 0, (__global DATA_TYPE *)(PTR + 6 * STRIDE_Y + Z##6));

#define STORE_ROW_PARTIAL_8(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_7(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                 \
    (BASENAME##7, 0, (__global DATA_TYPE *)(PTR + 7 * STRIDE_Y + Z##7));

#define STORE_ROW_PARTIAL_9(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_8(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                 \
    (BASENAME##8, 0, (__global DATA_TYPE *)(PTR + 8 * STRIDE_Y + Z##8));

#define STORE_ROW_PARTIAL_10(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_9(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)      \
    VSTORE_PARTIAL(N0, STORE_N0)                                                  \
    (BASENAME##9, 0, (__global DATA_TYPE *)(PTR + 9 * STRIDE_Y + Z##9));

#define STORE_ROW_PARTIAL_11(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_10(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                  \
    (BASENAME##A, 0, (__global DATA_TYPE *)(PTR + 10 * STRIDE_Y + Z##A));

#define STORE_ROW_PARTIAL_12(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_11(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                  \
    (BASENAME##B, 0, (__global DATA_TYPE *)(PTR + 11 * STRIDE_Y + Z##B));

#define STORE_ROW_PARTIAL_13(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_12(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                  \
    (BASENAME##C, 0, (__global DATA_TYPE *)(PTR + 12 * STRIDE_Y + Z##C));

#define STORE_ROW_PARTIAL_14(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_13(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                  \
    (BASENAME##D, 0, (__global DATA_TYPE *)(PTR + 13 * STRIDE_Y + Z##D));

#define STORE_ROW_PARTIAL_15(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_14(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                  \
    (BASENAME##E, 0, (__global DATA_TYPE *)(PTR + 14 * STRIDE_Y + Z##E));

#define STORE_ROW_PARTIAL_16(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) \
    STORE_ROW_PARTIAL_15(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)     \
    VSTORE_PARTIAL(N0, STORE_N0)                                                  \
    (BASENAME##F, 0, (__global DATA_TYPE *)(PTR + 15 * STRIDE_Y + Z##F));
/** @} */ // end of groupd STORE_ROW_PARTIAL_n

/** Partially store a block of the given size STORE_M0xSTORE_N0
 * @name STORE_BLOCK_PARTIAL
 *
 * @note The vector width @p N0 is also required for correct partial storing behaviour.
 * @note in case @p STORE_N0 != 1, 2, 3, 4, 8, 16, extra vstore(s) will be invoked, thus incurring small performance penalty.
 *
 * The data to store is expected to have consecutive names for each row.
 * E.g., for STORE_M0=3 and basename=c, the expected names are c0, c1 and c2.
 * The Z offset is expected to have consecutive names.
 * E.g., for STORE_M0=3 and Z=zin, the expected z offset names are zin0, zin1 and zin2.
 *
 * @param[in] STORE_M0  The number of rows to store. Supported: 1-16
 * @param[in] STORE_N0  The lower number of elements of vectors to store. Supported: 1-16 and <= @p N0
 * @param[in] N0        The size of each vector. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] DATA_TYPE The data type of the vectors
 * @param[in] BASENAME  The basename of the variables
 * @param[in] PTR       The base pointer
 * @param[in] STRIDE_Y  The stride value in y-axis direction
 * @param[in] Z         The offset in z-axis direction
 * @{
 */
#define STORE_BLOCK_PARTIAL_STR(STORE_M0, STORE_N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) STORE_ROW_PARTIAL_##STORE_M0(N0, STORE_N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)
#define STORE_BLOCK_PARTIAL(STORE_M0, STORE_N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z) STORE_BLOCK_PARTIAL_STR(STORE_M0, STORE_N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)
/** Store a block that can be partial in both x and y dimensions
 *
 * @note in cases @p PARTIAL_STORE_N0 != 1, 2, 3, 4, 8, 16, extra vstore(s) will be invoked, thus incurring small performance penalty.
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
 * @param[in] STRIDE_Y         The stride value in y-axis direction
 * @param[in] Z                The offset in z-axis direction
 * @param[in] PARTIAL_STORE_M0 The partial size in y, for partial blocks. Supported range: [1, @p M0)
 * @param[in] PARTIAL_STORE_N0 The partial size in x, for partial blocks. Supported range: [1, @p N0)
 * @param[in] PARTIAL_COND_Y   Condition on the y axis to perform the partial store Y. True to use PARTIAL_STORE_M0 rather than M0.
 * @param[in] PARTIAL_COND_X   Condition on the x axis to perform the partial store X. True to use PARTIAL_STORE_N0 rather than N0.
 */
#define STORE_BLOCK_PARTIAL_IN_X_AND_Y(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    if(!(PARTIAL_COND_X) && !(PARTIAL_COND_Y))                                                                                                            \
    {                                                                                                                                                     \
        STORE_BLOCK_PARTIAL(M0, N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z);                                                                           \
    }                                                                                                                                                     \
    else if((PARTIAL_COND_Y) && !(PARTIAL_COND_X))                                                                                                        \
    {                                                                                                                                                     \
        STORE_BLOCK_PARTIAL(PARTIAL_STORE_M0, N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z);                                                             \
    }                                                                                                                                                     \
    else if(!(PARTIAL_COND_Y) && (PARTIAL_COND_X))                                                                                                        \
    {                                                                                                                                                     \
        STORE_BLOCK_PARTIAL(M0, PARTIAL_STORE_N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z);                                                             \
    }                                                                                                                                                     \
    else                                                                                                                                                  \
    {                                                                                                                                                     \
        STORE_BLOCK_PARTIAL(PARTIAL_STORE_M0, PARTIAL_STORE_N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z);                                               \
    }
/** Store a block that can only be partial in x but not y.
 *
 * @note in case @p N0 or @p PARTIAL_STORE_N0 != 1, 2, 3, 4, 8, 16, extra vstore(s) will be invoked, thus incurring small performance penalty.
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
 * @param[in] STRIDE_Y         The stride value in y-axis direction
 * @param[in] Z                The offset in z-axis direction
 * @param[in] PARTIAL_STORE_N0 The partial size in x, for partial blocks. Supported range: [1, @p N0)
 * @param[in] PARTIAL_COND_X   Condition on the x axis to perform the partial store X. True to use PARTIAL_STORE_N0 rather than N0.
 */
#define STORE_BLOCK_PARTIAL_IN_X(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_N0, PARTIAL_COND_X) \
    if(!(PARTIAL_COND_X))                                                                                         \
    {                                                                                                             \
        STORE_BLOCK_PARTIAL(M0, N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z);                                   \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
        STORE_BLOCK_PARTIAL(M0, PARTIAL_STORE_N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z);                     \
    }
/** Store a block that can only be partial in y but not x.
 *
 * @note in case @p N0 or @p PARTIAL_STORE_N0 != 1, 2, 3, 4, 8, 16, extra vstore(s) will be invoked, thus incurring small performance penalty.
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
 * @param[in] STRIDE_Y         The stride value in y-axis direction
 * @param[in] Z                The offset in z-axis direction
 * @param[in] PARTIAL_STORE_M0 The partial size in y, for partial blocks. Supported range: [1, @p M0)
 * @param[in] PARTIAL_COND_Y   Condition on the y axis to perform the partial store Y. True to use PARTIAL_STORE_M0 rather than M0.
 */
#define STORE_BLOCK_PARTIAL_IN_Y(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_COND_Y) \
    if(!(PARTIAL_COND_Y))                                                                                         \
    {                                                                                                             \
        STORE_BLOCK_PARTIAL(M0, N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z);                                   \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
        STORE_BLOCK_PARTIAL(PARTIAL_STORE_M0, N0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z);                     \
    }
/** @} */ // end of group STORE_BLOCK_PARTIAL

/** Boundary-aware GEMM block store
 * @name STORE_BLOCK_BOUNDARY_AWARE
 * This macro assumes the following schemes to achieve boundary-awareness:
 *  - Overlapping load in Y axis from lhs tensor. This implies lhs has no padding along y dim.
 *  - Non-Overlapping(normal) load from rhs tensor. This imples rhs can have paddings.
 *  - Overlapping load in Y axis from bias tensor. This implies rhs has no padding along y dim.
 * The macro then ensures that the dst tensor can be stored without any paddings in both x and y dim.
 *
 * In the y dimension, we place the partial blocks **at the beginning** while in the x dimension, we place the partial
 * blocks **at the end**.
 * Say, the dst tensor is of shape MxN and we have M0 and N0 as the block size, this is how we define "partial blocks"/
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
 * @note in cases @p PARTIAL_STORE_N0 != 1, 2, 3, 4, 8, 16, extra vstore(s) will be invoked, thus incurring small performance penalty.
 *
 * It automatically detects if a giving M,N,M0,N0 combination can yield partial blocks in either X and Y dimension,
 * and select corresponding store methods such that the boundary detection logic is only added when needed.
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
 * @param[in] STRIDE_Y         The stride value in y-axis direction
 * @param[in] Z                The offset in z-axis direction
 * @param[in] PARTIAL_STORE_M0 The partial size in y, for partial blocks. Supported: [0, @p M0)
 * @param[in] PARTIAL_STORE_N0 The partial size in x, for partial blocks. Supported: [0, @p N0)
 * @param[in] PARTIAL_COND_Y   Condition on the y axis to perform the partial store Y. True to use PARTIAL_STORE_M0 rather than M0.
 * @param[in] PARTIAL_COND_X   Condition on the x axis to perform the partial store X. True to use PARTIAL_STORE_N0 rather than N0.
 * @{
 */
#if defined(PARTIAL_STORE_M0) && defined(PARTIAL_STORE_N0)
#if PARTIAL_STORE_M0 == 0 && PARTIAL_STORE_N0 == 0
// Case1: No partial blocks in either x or y
#define STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    STORE_BLOCK(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z)

#elif PARTIAL_STORE_M0 > 0 && PARTIAL_STORE_N0 == 0
// Case2: Partial blocks in y
#define STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    STORE_BLOCK_PARTIAL_IN_Y(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_COND_Y)

#elif PARTIAL_STORE_M0 == 0 && PARTIAL_STORE_N0 > 0
// Case3: Partial blocks in x
#define STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    STORE_BLOCK_PARTIAL_IN_X(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_N0, PARTIAL_COND_X)

#else // PARTIAL_STORE_M0 == 0 && PARTIAL_STORE_N0 == 0
// Case4: Partial blocks in both x and y
#define STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X) \
    STORE_BLOCK_PARTIAL_IN_X_AND_Y(M0, N0, DATA_TYPE, BASENAME, PTR, STRIDE_Y, Z, PARTIAL_STORE_M0, PARTIAL_STORE_N0, PARTIAL_COND_Y, PARTIAL_COND_X)

#endif // PARTIAL_STORE_M0 == 0 && PARTIAL_STORE_N0 == 0

#endif    // defined(PARTIAL_STORE_M0) && defined(PARTIAL_STORE_N0)
/** @} */ // end of group STORE_BLOCK_BOUNDARY_AWARE

/** Compute the start m0 row (LHS, BIAS and DST) in a boundary-aware way so as to avoid padding
 * @name COMPUTE_M0_START_ROW
 * If there're any partial blocks in y dimension, they are placed at the beginning of the rows.
 * This shift amount is added to all rows such that the partial block (at the beginning) overlaps with the subsequent
 * blocks in the y dimension to avoid any padding.
 * EG: M0=4, PARTIAL_STORE_M0=1:
 *                  | Non-overlapping | +M0_ROW_SHIFT (Overlapping)
 * block 0 (partial)| start row = 0   | start row = 0
 * block 1 (full)   | start row = 4   | start row = 1
 * block 2 (full)   | start row = 8   | start row = 5
 *
 * @param[in] y                Global id of current block in y.
 * @param[in] M0               The number of rows to store, for non-partial blocks. Supported: 1-16
 * @param[in] PARTIAL_STORE_M0 The partial size in y, for partial blocks. Supported: [0, @p M0)
 * @{
 */
#if defined(PARTIAL_STORE_M0)
#define COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) \
    ((uint)(max(0, (int)(y * M0) - (int)((M0 - PARTIAL_STORE_M0) % M0))))
#else // defined(PARTIAL_STORE_M0)
#define COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) \
    ((uint)(y * M0))
#endif    // defined(PARTIAL_STORE_M0)
/** @} */ // end of group COMPUTE_M0_START_ROW

/** Store a vector that can only be partial in x.
 * @name STORE_VECTOR_SELECT
 * @note in case @p vec_size or @p leftover != 1, 2, 3, 4, 8, 16, extra vstore(s) will be invoked, thus incurring small performance penalty.
 *
 * The data to store is expected to end in a 0.
 * E.g., for basename=c, the expected name is c0.
 *
 * @param[in] basename  The name of the variable without trailing 0
 * @param[in] data_type The data type of the vector
 * @param[in] ptr       The base pointer
 * @param[in] vec_size  The vector size if cond = false. Supported: 1, 2, 3, 4, 8, 16
 * @param[in] leftover  The vector size if cond = true. Supported range: [1, @p vec_size0)
 * @param[in] cond      Condition to select either vec_size0 or vec_size1
 * @{
 */
#define STORE_VECTOR_SELECT(basename, data_type, ptr, vec_size, leftover, cond) \
    STORE_BLOCK_PARTIAL_IN_X(1, vec_size, data_type, basename, ptr, 0, 0, leftover, cond)
/** @} */ // end of group STORE_VECTOR_SELECT
