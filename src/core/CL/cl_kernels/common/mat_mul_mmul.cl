/*
 * Copyright (c) 2023 Arm Limited.
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
#include "helpers.h"
#include "tile_helpers.h"

#if defined(MAT_MUL_NATIVE_MMUL_NT_NT)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul) using MMUL: LHS non-transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The tile's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=1).
 * @note The number of leftover outputs rows/columns must be passed using -DN0_LEFTOVER and -DM0_LEFTOVER (e.g. -DN0_LEFTOVER=2, -DM0_LEFTOVER=3)
 * @note The MMUL block dimension (MMUL_M0, MMUL_N0, MMUL_K0) must be passed at compile time using -DMMUL_M0, -DMMUL_N0 and -DMMUL_K0 (e.g. -DMMUL_M0=4, -DMMUL_N0=4, -DMMUL_K0=4).
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_MMUL_NT_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_ptr                           Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                      Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                             The width of the rhs tensor
 * @param[in]  rhs_h                             The height of the rhs tensor
 * @param[in]  rhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the rhs matrix
 * @param[out] dst_ptr                           Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                             The width of the dst tensor
 * @param[in]  dst_h                             The height of the dst tensor
 * @param[in]  dst_n                             Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the dst matrix
 * @param[in]  M                                 Number of rows in LHS matrix
 * @param[in]  N                                 Number of columns in RHS matrix
 * @param[in]  K                                 Number of columns in LHS matrix and rows in RHS matrix, which is multiple of MMUL_K0.
 */
__kernel void mat_mul_native_mmul_nt_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K)
{
#define MMUL_BLOCK_SIZE (MMUL_M0 * MMUL_N0) // MMUL block size for the output matrix

    // The output/destination matrix is divided into "sections". Each section is filled by a group of
    // threads of size MMUL_BLOCK_SIZE, bundled together according to GWS_x.
    // Each thread writes to a tile of M0 x N0 (the usual output block size for a thread) in the output matrix.
    // Therefore, the section dimensions are (MMUL_M0 x M0) x (MMUL_N0 x N0).

    // The GWS is constructed in such a way that the y global id is the y section coordinate,
    // and the x global id is a transformed thread id: MMUL_BLOCK_SIZE number of consecutive threads
    // in the x dimension corresponding to a section.
    // This can be visualized as first obtaining the coordinates of all the sections:
    // x = [0, (N / N0) / MMUL_N0) --> (N / N0) / MMUL_N0 is the number of sections in x dimension
    // y = [0, (M / M0) / MMUL_M0) --> (M / M0) / MMUL_M0 is the number of sections in y dimension
    // Then multiply the x coordinates with MMUL_SECTION_NUM_THREADS to get the consecutive thread ids in the x dimension
    // x = [0, ((N / N0) / MMUL_N0) * MMUL_N0 * MMUL_M0)
    // x = [0, (N / N0) * MMUL_MO)
    const uint x0 = get_global_id(0); // [0, (N / N0) * MMUL_M0)
                                      // The upper limit is a simplified version of (N / N0) / MMUL_N0) * MMUL_BLOCK_SIZE)
    const uint y0 = get_global_id(1); // [0, (M / M0) / MMUL_M0)
    const uint z  = get_global_id(2); // Batch

    // Get section coordinates
    const uint section_x = (x0 / MMUL_BLOCK_SIZE);
    const uint section_y = y0;

    // Within these sections, each thread writes onto a small output block of size M0 x N0
    // in row major order. A section divided into tiles can be visualized as below.
    //
    //                   (Figure 1)
    //          A Section in the Output Matrix
    //
    //    _____N0__________N0____________________N0____
    //    |           |          |         |           |
    //    |           |          |         |           |
    // M0 |  Thread 1 | Thread 2 |   ...   |  Thread   |
    //    |           |          |         |  MMUL_N0  |
    //    |___________|__________|_________|___________|
    //    |           |                    |           |
    //    |           |                    |           |
    // M0 |  Thread   |     .              |           |
    //    | MMUL_N0+1 |       .            |           | (M0 x MMUL_M0)
    //    |___________|         .          |           |
    //    |     .                          |           |
    //    |     .                          |           |
    //    |     .                          |           |
    //    |                                |___________|
    //    |                                |           |
    //    |                                |  Thread   |
    // M0 |                                | MMUL_N0 x |
    //    |                                | MMUL_M0   |
    //    |________________________________|___________|
    //                  N0 x MMUL_N0
    //
    // The output matrix has several of these sections. As shown above, each section
    // will be filled by a separate thread group of size MMUL_BLOCK_SIZE. The overall
    // section layout of the output matrix is as below. For instance, S(1,1) will be filled
    // by MMUL_BLOCK_SIZE (possibly equal to 16) threads, so as S(0,1) and others.
    //
    //                          (Figure 2)
    //                          DST Matrix
    //              ____________________________________
    //             |        |        |       |         |
    //             | S(0,0) | S(0,1) | ...   | S(0, X) |
    //             |________|________|_______|_________|
    //             |        |        |       |         |
    //             | S(1,0) | S(1,1) | ...   | S(1, X) |
    //             |________|________|_______|_________|
    //             |   .    |        |                 |
    //             |   .    |        |                 |        Y = (M / M0) / MMUL_M0 - 1 (Max possible section y coordinate)
    //             |   .    |        |                 |        X = (N / N0) / MMUL_N0 - 1 (Max possible section x coordinate)
    //             |________|________|_________________|
    //             |        |        |       |         |        S(y, x) denotes the section, and y and x are computed in
    //             | S(Y,0) | S(Y,1) |       | S(Y, X) |        section_y, section_x respectively.
    //             |________|________|_______|_________|
    //
    //
    //
    //
    // A complete view involving the three matrices is given below. It examplifies how the section S(0,0) is computed.
    //
    //                                                    (Figure 3)
    //                                                  Complete View
    //
    //                       LHS Matrix                             RHS Matrix                                          DST Matrix
    //
    //                   ___MMUL_K0___________               __MMUL_N0 x N0____________                     ___MMUL_N0 x N0____________________
    //                  /|xxxxxxxxxx|         |             /|xxxxxxxxxxxxxxx|         |                   /|xxxxxxxxxxxxxxxxxxx|             |
    //                 / |xxxxxxxxxx|         |    MMUK_K0  ||xxxxxxxxxxxxxxx|         |                  / |xxxxxxxxxxxxxxxxxxx|             |
    //      MMUL_M0    | |xxxxxxxxxx|  --->   |             ||xxxxxxxxxxxxxxx| . . .   |        MMUL_M0  |  |xxxxxxxxxxxxxxxxxxx|             |
    //        x M0     | |xxxxxxxxxx|         |             \|_______________|_________|          x M0   |  |xxxxxxxxxxxxxxxxxxx|     ...     |
    //                 | |xxxxxxxxxx|         |              |                         |                 |  |xxxxxxxxxxxxxxxxxxx|             |
    //                 | |xxxxxxxxxx|         |     x        |       |                 |   =              \ |xxxxxxxxxxxxxxxxxxx|             |
    //                  \|__________|_________|              |       |                 |                   \|___________________|             |
    //                   |                    |              |       \/                |                    |                                 |
    //                   |   ,                |              |_________________________|                    |         .                       |
    //                   |   ,                |                                                             |         .                       |
    //                   |   ,                |                                                             |         .                       |
    //                   |____________________|                                                             |_________________________________|
    //
    // Horizontal and vertical arrows show the direction of K loop (main loop in the kernel).
    // Each output section shown above is a zooomed out version of Figure 1.
    //
    // In each iteration of the main loop, LHS matrix traverses towards rightward, and RHS matrix traverses towards downward,
    // the LHS section of (MMUL_M0 x M0) x MMUL_K0 and RHS section of MMUL_K0 x (MMUL_N0 x N0) is multiplied
    // "cooperatively" using arm_matrix_multiply calls, and the result is accummulated over the output (DST) section
    // of size (MMUL_M0 x M0) x (MMUL_N0 x N0) shown with 'x' signs.
    //
    // If it was a single thread, this multiplication would have been straightforward with a T_MMUL call.
    // However, since it involves multiple threads working together using the aforementioned extension, it
    // works slightly differently.
    //
    // Here is how threads access the LHS and RHS matrices:
    // (Assume MMUL_K0 = MMUL_N0 = MMUL_M0 = 4 because the following diagram is heavily dependent on this)
    //
    //                                              (Figure 4)
    //                               Thread Access Layouts in LHS & RHS matrices
    //
    //                   LHS matrix                                                             RHS Matrix
    //           ___________________________                     __________N0 times______N0 times____________________N0 times_______
    //          |__T0__|__T1__|__T2__|__T3__|                   |__T0__| ... |__T0__|__T1__| ...  |__T1__| ... |__T3__| ... |__T3__|
    //          |__T0__| ...                |                   |__T4__| ... |__T4__|__T5__| ...  |__T5__| ... |__T7__| ... |__T7__|
    //    M0    |   .    .                  |                   |__T8__| ... |__T8__|__T9__| ...  |__T9__| ... |__T11_| ... |__T11_|
    //   Times  |   .       .               |                   |__T12_|_____|__T12_|__T13_|______|__T13_|_____|__T15_|_____|__T15_|
    //          |   .           .           |           X
    //          |__T0__|__T1__|__T2__|__T3__|
    //          |__T4__|__T5__|__T6__|__T7__|
    //          |__T4__|__T5__|__T6__|__T7__|
    //    M0    |   .    .                  |
    //   Times  |   .       .               |
    //          |   .           .           |
    //          |__T4__|__T5__|__T6__|__T7__|
    //          |__T8__|__T9__|__T10_|__T11_|
    //    M0    |   .                       |
    //   Times  |   .                       |
    //          |   .                       |
    //          |__T12_|__T13_|__T14_|__T15_|
    //    M0    |   .                       |
    //   Times  |   .                       |
    //          |   .                       |
    //          |__T12_|__T13_|__T14_|__T15_|
    //
    //
    // This access layout is designed such that the threads access continuous elements of each matrix (in terms of row/column).
    // To multiply these large sections, the arm_matrix_multiply call is made for each of the M0xN0 elements. So, for each
    // combination of m0 and n0 (iterators of M0 and N0 from 0 to M0-1 and N0-1 respectively), one arm_matrix_multiply call is
    // made, and MMUL_BLOCK_SIZE number of threads compute the result.
    //
    // The matrix multiplication taking place in this extension
    // is an "interleaved" one, because, for example, if m0=0 and n0=0, i.e. the first iteration, we would use the first,
    // M0-th, 2M0-th and 3M0-th rows of the LHS matrix. Similarly, we jump N0 steps in the RHS matrix. This is how we access
    // one element for each thread in a single (m0, n0) loop.
    //
    //   For example, if we have
    //          - a 8 x 4 LHS section
    //          - 4 x 8 RHS section
    //          - Each vector variable ai, bj represent a 4x1 vector
    //          - ^T (superscript T) denotes transpose
    //          - M0 = N0 = 2
    //          - MMUL_N0 = MMUL_M0 = MMUL_K0 = 4
    //
    //                                             (Figure 5)
    //                              Mathematical view of the Matrix Multiplication
    //
    //      LHS                           RHS                                           DST
    //    [  a1^T  ]            [ b1 b2 b3 b4 b5 b6 b7 ]                [ a1^Tb1  a1^Tb2  a1^Tb3 ... a1^Tb7 ]
    //    [  a2^T  ]                                    4 x 8           [ a2^Tb1  a2^Tb2  a2^Tb3 ... a2^Tb7 ]
    //    [  a3^T  ]                                                    [                                   ]
    //    [  a4^T  ]                                                =   [   .       .                       ]
    //    [  a5^T  ]        X                                           [   .          .                    ]
    //    [  a6^T  ]                                                    [   .             .                 ]
    //    [  a7^T  ]                                                    [                                   ]
    //    [  a8^T  ]                                                    [ a7^Tb1  a7^Tb2  a7^Tb3 ... a7^Tb7 ]
    //              8 x 4                                                                                     8 x 8
    //
    //
    //  For the first iteration, i.e. (m0, n0) = (0, 0), the arm_matrix_multiply would multiply the following matrices:
    //
    //    [  a1^T  ]            [  b1 b3 b5 b7 ]                [ a1^Tb1  a1^Tb3  a1^Tb5  a1^Tb7 ]
    //    [  a3^T  ]        x                   4 x 4     =     [ a3^Tb1  a1^Tb3  a1^Tb5  a1^Tb7 ]
    //    [  a5^T  ]                                            [ a5^Tb1  a1^Tb3  a1^Tb5  a1^Tb7 ]
    //    [  a7^T  ]                                            [ a7^Tb1  a7^Tb3  a7^Tb5  a7^Tb7 ]
    //              4 x 4                                                                         4 x 4
    //  The elements calculated in the 4x4 output block are the "interleaved" elements in the DST above.
    //  When we follow for each combination of (m0, n0), every element of the DST matrix "section" is filled.
    //

    // Get thread coordinates within an mmul block (of size MMUL_BLOCK_SIZE)
    // Since threads are grouped in x dimension, the modular of x-dim global id
    // wrt the MMUL_BLOCK_SIZE is the thread id in the group, ranging from 0 to
    // MMUL_BLOCK_SIZE-1. Because the thread numbering is in row-major order.
    const uint thread_id = (x0 % MMUL_BLOCK_SIZE);
    const uint thread_x  = thread_id % MMUL_N0;
    const uint thread_y  = (thread_id / MMUL_N0);

    // Starting destination coordinates
    // Note: We need to clamp dst_x and dst_y because we always need to execute a complete MMUL block! Only after the matrix multiplication
    // part can we exit the kernel if it is out-of-bound. Remember, we have a cooperative matrix multiplication. Therefore, we need a full block to get the correct results
    // Although we will never write out-of-bound, we still need this clamp to ensure that we do not read out-of-bound either.
    // The unclamped dst coordinates can be calculated easily from the output section coordinates and the thread coordinates (see above figure).

    // See Figure 1 & 2. Thread step size is N0 and M0,
    //                   Section step size is N0 x MMUL_N0 and M0 x MMUL_M0
    //                   respectively for x, y dimensions.
    const uint dst_x_unclamped = thread_x * N0 + section_x * N0 * MMUL_N0;
    const uint dst_y_unclamped = thread_y * M0 + section_y * M0 * MMUL_M0;
    const uint dst_x           = min(dst_x_unclamped, (uint)(N - N0));
    const uint dst_y           = min(dst_y_unclamped, (uint)(M - M0));

    // Starting LHS coordinates
    const uint lhs_x = thread_x;
    const uint lhs_y = dst_y;

    // Starting RHS coordinates
    const uint rhs_x = dst_x;
    const uint rhs_y = thread_y;

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y + z * rhs_stride_z;
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    // MMUL extension accumulate the result in F32 for both F32 and F16
    TILE(float, M0, N0, c_f32);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_f32[i].v = 0;
    })

    for(int k = 0; k < K; k += MMUL_K0)
    {
        // A tile of M0xK0 but K0 must be set to 1
        TILE(DATA_TYPE, M0, 1, a);
        // A tile of K0xN0 but K0 must be set to 1
        TILE(DATA_TYPE, 1, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, BUFFER, rhs, 0, 0, 1, rhs_stride_y, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                c_f32[m0].s[n0] = arm_matrix_multiply(a[m0].s[0], b[0].s[n0], c_f32[m0].s[n0]);
            })
        })

        lhs_offset_first_element_in_bytes += MMUL_K0 * sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += MMUL_K0 * rhs_stride_y;
    }

    // For threads "outside" of the dst bound, we do not write but we have to "read" (arm_matrix_multiply). That's why this needs to happen after arm_matrix_multiply
    if(dst_x_unclamped >= N || dst_y_unclamped >= M)
    {
        return;
    }

#if defined(HALF_PRECISION)
    TILE(DATA_TYPE, M0, N0, c);

    // Conversion required for the half precision
    LOOP_UNROLLING(int, m0, 0, 1, M0,
    {
        LOOP_UNROLLING(int, n0, 0, 1, N0,
        {
            c[m0].s[n0] = c_f32[m0].s[n0];
        })
    })
#else // defined(HALF_PRECISION)
#define c c_f32
#endif // defined(HALF_PRECISION)

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE(N0)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }
    else
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE_PARTIAL(N0, N0_LEFTOVER)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }

#undef MMUL_BLOCK_SIZE
}
#endif // defined(MAT_MUL_NATIVE_MMUL_NT_NT)

#if defined(MAT_MUL_NATIVE_MMUL_T_NT)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul) using MMUL: LHS transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The tile's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=1).
 * @note The number of leftover outputs rows/columns must be passed using -DN0_LEFTOVER and -DM0_LEFTOVER (e.g. -DN0_LEFTOVER=2, -DM0_LEFTOVER=3)
 * @note The MMUL block dimension (MMUL_M0, MMUL_N0, MMUL_K0) must be passed at compile time using -DMMUL_M0, -DMMUL_N0 and -DMMUL_K0 (e.g. -DMMUL_M0=4, -DMMUL_N0=4, -DMMUL_K0=4).
 * @note The dimension K must be passed at compile time using -DK (e.g. -DK=4). K must be a multiple of MMUL_K0
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_MMUL_T_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 8, 16
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_ptr                           Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                      Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                             The width of the rhs tensor
 * @param[in]  rhs_h                             The height of the rhs tensor
 * @param[in]  rhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the rhs matrix
 * @param[out] dst_ptr                           Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                             The width of the dst tensor
 * @param[in]  dst_h                             The height of the dst tensor
 * @param[in]  dst_n                             Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the dst matrix
 * @param[in]  M                                 Number of rows in DST matrix
 * @param[in]  N                                 Number of columns in DST matrix
 * @param[in]  K                                 Number of rows in LHS and RHS matrices, which is multiple of MMUL_K0.
 */
__kernel void mat_mul_native_mmul_t_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K)
{
#define MMUL_BLOCK_SIZE (MMUL_M0 * MMUL_N0)
    // For explanations on how this kernel works, please refer to NT/NT kernel. This kernel makes little modifications to it.

    const uint x0 = get_global_id(0); // [0, (N / N0) * MMUL_M0)
                                      // The upper limit is a simplified version of (N / N0) / MMUL_N0) * MMUL_BLOCK_SIZE)
    const uint y0 = get_global_id(1); // [0, (M / M0) / MMUL_M0)
    const uint z  = get_global_id(2); // Batch

    // Get section coordinates
    const uint section_x = (x0 / MMUL_BLOCK_SIZE);
    const uint section_y = y0;

    // Get thread coordinates
    uint thread_id = (x0 % MMUL_BLOCK_SIZE);
    uint thread_x  = thread_id % MMUL_N0;
    uint thread_y  = (thread_id / MMUL_N0);

    // See Nt/Nt kernel for explanations
    const uint dst_x_unclamped = thread_x * N0 + section_x * N0 * MMUL_N0;
    const uint dst_y_unclamped = thread_y * M0 + section_y * M0 * MMUL_M0;
    const uint dst_x           = min(dst_x_unclamped, (uint)(N - N0));
    const uint dst_y           = min(dst_y_unclamped, (uint)(M - M0));

    // Starting LHS coordinates
    uint lhs_x = dst_y;
    uint lhs_y = thread_x;

    // Starting RHS coordinates
    uint rhs_x = dst_x;
    uint rhs_y = thread_y;

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y + z * rhs_stride_z;
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    // MMUL extension accumulate the result in F32 for both F32 and F16
    TILE(float, M0, N0, c_f32);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_f32[i].v = 0;
    })

    for(int k = 0; k < K; k += MMUL_K0)
    {
        TILE(DATA_TYPE, 1, M0, a);
        TILE(DATA_TYPE, 1, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, 1, M0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, BUFFER, rhs, 0, 0, 1, rhs_stride_y, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                c_f32[m0].s[n0] = arm_matrix_multiply(a[0].s[m0], b[0].s[n0], c_f32[m0].s[n0]);
            })
        })

        lhs_offset_first_element_in_bytes += MMUL_K0 * lhs_stride_y;
        rhs_offset_first_element_in_bytes += MMUL_K0 * rhs_stride_y;
    }

    // For threads "outside" of the dst bound, we do not write but we have to "read" (arm_matrix_multiply). That's why this needs to happen after arm_matrix_multiply
    if(dst_x_unclamped >= N || dst_y_unclamped >= M)
    {
        return;
    }

#if defined(HALF_PRECISION)
    TILE(DATA_TYPE, M0, N0, c);

    // Conversion required for the half precision
    LOOP_UNROLLING(int, m0, 0, 1, M0,
    {
        LOOP_UNROLLING(int, n0, 0, 1, N0,
        {
            c[m0].s[n0] = c_f32[m0].s[n0];
        })
    })
#else // defined(HALF_PRECISION)
#define c c_f32
#endif // defined(HALF_PRECISION)

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE(N0)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }
    else
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE_PARTIAL(N0, N0_LEFTOVER)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }

#undef MMUL_BLOCK_SIZE
}
#endif // defined(MAT_MUL_NATIVE_MMUL_T_NT)

#if defined(MAT_MUL_NATIVE_MMUL_NT_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul) using MMUL: LHS non-transposed, RHS transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The tile's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=1).
 * @note The number of leftover outputs rows/columns must be passed using -DN0_LEFTOVER and -DM0_LEFTOVER (e.g. -DN0_LEFTOVER=2, -DM0_LEFTOVER=3)
 * @note The MMUL block dimension (MMUL_M0, MMUL_N0, MMUL_K0) must be passed at compile time using -DMMUL_M0, -DMMUL_N0 and -DMMUL_K0 (e.g. -DMMUL_M0=4, -DMMUL_N0=4, -DMMUL_K0=4).
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_MMUL_NT_T)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_ptr                           Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                      Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                             The width of the rhs tensor
 * @param[in]  rhs_h                             The height of the rhs tensor
 * @param[in]  rhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the rhs matrix
 * @param[out] dst_ptr                           Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                             The width of the dst tensor
 * @param[in]  dst_h                             The height of the dst tensor
 * @param[in]  dst_n                             Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the dst matrix
 * @param[in]  M                                 Number of rows in LHS matrix
 * @param[in]  N                                 Number of columns in RHS matrix
 * @param[in]  K                                 Number of columns in LHS matrix and columns in RHS matrix, which is multiple of MMUL_K0.
 */
__kernel void mat_mul_native_mmul_nt_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K)
{
#define MMUL_BLOCK_SIZE (MMUL_M0 * MMUL_N0)
    // For explanations on how this kernel works, please refer to NT/NT kernel. This kernel makes little modifications to it.

    const uint x0 = get_global_id(0); // [0, (N / N0) * MMUL_M0)
                                      // The upper limit is a simplified version of (N / N0) / MMUL_N0) * MMUL_BLOCK_SIZE)
    const uint y0 = get_global_id(1); // [0, (M / M0) / MMUL_M0)
    const uint z  = get_global_id(2); // Batch

    // Get block coordinates
    const uint section_x = (x0 / MMUL_BLOCK_SIZE);
    const uint section_y = y0;

    // Get thread coordinates within a block
    const uint thread_id = (x0 % MMUL_BLOCK_SIZE);
    const uint thread_x  = thread_id % MMUL_N0;
    const uint thread_y  = (thread_id / MMUL_N0);

    // Starting destination coordinates
    // Note: We need to clamp dst_x and dst_y because we always need to execute a complete MMUL block! Only after the matrix multiplication
    // part can we exit the kernel if it is out-of-bound. Remember, we have a cooperative matrix multiplication. Therefore, we need a full block to get the correct results
    // Although we will never write out-of-bound, we still need this clamp to ensure that we do not read out-of-bound either.
    const uint dst_x_unclamped = thread_x * N0 + section_x * N0 * MMUL_N0;
    const uint dst_y_unclamped = thread_y * M0 + section_y * M0 * MMUL_M0;
    const uint dst_x           = min(dst_x_unclamped, (uint)(N - N0));
    const uint dst_y           = min(dst_y_unclamped, (uint)(M - M0));

    // Starting LHS coordinates
    const uint lhs_x = thread_x;
    const uint lhs_y = dst_y;

    // Starting RHS coordinates
    const uint rhs_x = thread_y;
    const uint rhs_y = dst_x;

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y + z * rhs_stride_z;
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    // MMUL extension accumulate the result in F32 for both F32 and F16
    TILE(float, M0, N0, c_f32);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_f32[i].v = 0;
    })

    for(int k = 0; k < K; k += MMUL_K0)
    {
        // A tile of M0xK0 but K0 must be set to 1
        TILE(DATA_TYPE, M0, 1, a);
        // A tile of N0xK0 but K0 must be set to 1
        TILE(DATA_TYPE, N0, 1, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, 1, BUFFER, rhs, 0, 0, 1, rhs_stride_y, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                c_f32[m0].s[n0] = arm_matrix_multiply(a[m0].s[0], b[n0].s[0], c_f32[m0].s[n0]);
            })
        })

        lhs_offset_first_element_in_bytes += MMUL_K0 * sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += MMUL_N0 * sizeof(DATA_TYPE);
    }

    // For threads "outside" of the dst bound, we do not write but we have to "read" (arm_matrix_multiply). That's why this needs to happen after arm_matrix_multiply
    if(dst_x_unclamped >= N || dst_y_unclamped >= M)
    {
        return;
    }

#if defined(HALF_PRECISION)
    TILE(DATA_TYPE, M0, N0, c);

    // Conversion required for the half precision
    LOOP_UNROLLING(int, m0, 0, 1, M0,
    {
        LOOP_UNROLLING(int, n0, 0, 1, N0,
        {
            c[m0].s[n0] = c_f32[m0].s[n0];
        })
    })
#else // defined(HALF_PRECISION)
#define c c_f32
#endif // defined(HALF_PRECISION)

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE(N0)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }
    else
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE_PARTIAL(N0, N0_LEFTOVER)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }

#undef MMUL_BLOCK_SIZE
}
#endif // defined(MAT_MUL_NATIVE_MMUL_NT_T)

#if defined(MAT_MUL_NATIVE_MMUL_T_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul) using MMUL: LHS non-transposed, RHS transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The tile's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=1).
 * @note The number of leftover outputs rows/columns must be passed using -DN0_LEFTOVER and -DM0_LEFTOVER (e.g. -DN0_LEFTOVER=2, -DM0_LEFTOVER=3)
 * @note The MMUL block dimension (MMUL_M0, MMUL_N0, MMUL_K0) must be passed at compile time using -DMMUL_M0, -DMMUL_N0 and -DMMUL_K0 (e.g. -DMMUL_M0=4, -DMMUL_N0=4, -DMMUL_K0=4).
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_MMUL_NT_T)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 8, 16
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_ptr                           Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                      Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                             The width of the rhs tensor
 * @param[in]  rhs_h                             The height of the rhs tensor
 * @param[in]  rhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the rhs matrix
 * @param[out] dst_ptr                           Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                             The width of the dst tensor
 * @param[in]  dst_h                             The height of the dst tensor
 * @param[in]  dst_n                             Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the dst matrix
 * @param[in]  M                                 Number of rows in LHS matrix
 * @param[in]  N                                 Number of columns in RHS matrix
 * @param[in]  K                                 Number of rows in LHS matrix and columns in RHS matrix, which is multiple of MMUL_K0.
 */
__kernel void mat_mul_native_mmul_t_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K)
{
#define MMUL_BLOCK_SIZE (MMUL_M0 * MMUL_N0)
    // For explanations on how this kernel works, please refer to NT/NT kernel. This kernel makes little modifications to it.

    const uint x0 = get_global_id(0); // [0, (N / N0) * MMUL_M0)
                                      // The upper limit is a simplified version of (N / N0) / MMUL_N0) * MMUL_BLOCK_SIZE)
    const uint y0 = get_global_id(1); // [0, (M / M0) / MMUL_M0)
    const uint z  = get_global_id(2); // Batch

    // Get block coordinates
    const uint section_x = (x0 / MMUL_BLOCK_SIZE);
    const uint section_y = y0;

    // Get thread coordinates within a block
    const uint thread_id = (x0 % MMUL_BLOCK_SIZE);
    const uint thread_x  = thread_id % MMUL_N0;
    const uint thread_y  = (thread_id / MMUL_N0);

    // Starting destination coordinates
    // Note: We need to clamp dst_x and dst_y because we always need to execute a complete MMUL block! Only after the matrix multiplication
    // part can we exit the kernel if it is out-of-bound. Remember, we have a cooperative matrix multiplication. Therefore, we need a full block to get the correct results
    // Although we will never write out-of-bound, we still need this clamp to ensure that we do not read out-of-bound either.
    const uint dst_x_unclamped = thread_x * N0 + section_x * N0 * MMUL_N0;
    const uint dst_y_unclamped = thread_y * M0 + section_y * M0 * MMUL_M0;
    const uint dst_x           = min(dst_x_unclamped, (uint)(N - N0));
    const uint dst_y           = min(dst_y_unclamped, (uint)(M - M0));

    // Starting LHS coordinates
    const uint lhs_x = dst_y;
    const uint lhs_y = thread_x;

    // Starting RHS coordinates
    const uint rhs_x = thread_y;
    const uint rhs_y = dst_x;

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y + z * rhs_stride_z;
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    // MMUL extension accumulate the result in F32 for both F32 and F16
    TILE(float, M0, N0, c_f32);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_f32[i].v = 0;
    })

    for(int k = 0; k < K; k += MMUL_K0)
    {
        // A tile of K0xM0 but K0 must be set to 1
        TILE(DATA_TYPE, 1, M0, a);
        // A tile of N0xK0 but K0 must be set to 1
        TILE(DATA_TYPE, N0, 1, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, 1, M0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, 1, BUFFER, rhs, 0, 0, 1, rhs_stride_y, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                c_f32[m0].s[n0] = arm_matrix_multiply(a[0].s[m0], b[n0].s[0], c_f32[m0].s[n0]);
            })
        })

        lhs_offset_first_element_in_bytes += MMUL_K0 * lhs_stride_y;
        rhs_offset_first_element_in_bytes += MMUL_N0 * sizeof(DATA_TYPE);
    }

    // For threads "outside" of the dst bound, we do not write but we have to "read" (arm_matrix_multiply). That's why this needs to happen after arm_matrix_multiply
    if(dst_x_unclamped >= N || dst_y_unclamped >= M)
    {
        return;
    }

#if defined(HALF_PRECISION)
    TILE(DATA_TYPE, M0, N0, c);

    // Conversion required for the half precision
    LOOP_UNROLLING(int, m0, 0, 1, M0,
    {
        LOOP_UNROLLING(int, n0, 0, 1, N0,
        {
            c[m0].s[n0] = c_f32[m0].s[n0];
        })
    })
#else // defined(HALF_PRECISION)
#define c c_f32
#endif // defined(HALF_PRECISION)

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE(N0)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }
    else
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE_PARTIAL(N0, N0_LEFTOVER)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }

#undef MMUL_BLOCK_SIZE
}
#endif // defined(MAT_MUL_NATIVE_MMUL_T_T)
