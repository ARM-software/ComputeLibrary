/*
 * Copyright (c) 2024 Arm Limited.
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

// The below defines the various reduce operations for our purposes.
// Where a corresponds to the existing value, and b the new value.
#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))

#ifdef IS_FLOAT
#define MAX_OP(a, b) fmax(a, b)
#define MIN_OP(a, b) fmin(a, b)
#else // ifdef IS_FLOAT
#define MAX_OP(a, b) max(a, b)
#define MIN_OP(a, b) min(a, b)
#endif // ifdef IS_FLOAT

#define UPDATE_OP(a, b) (b)

#ifdef SCATTER_MP1D_2D_MPND

/** This kernel performs scatter operation
 *
 * @note Datatype should be given as a compile-time argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Number of indices should be given as a compile-time argument using -DNUM_INDICES, e.g. -DNUM_INDICES=3
 * @note Index length should be given as a compile-time argument using -DINDEX_LENGTH, e.g. -DINDEX_LENGTH=2
 * @note Outermost output shapes should be given as a compile-time argument using -DOUT_SHAPE_N_MINUS_X, where
 *       X must be 1,2,3,4,5, e.g. -DOUT_SHAPE_N_MINUS_1=3, ...
 * @note Number of elements to copy in a row should be given as a compile-time argument using -DN0, e.g. -DN0=4
 * @note Number of partial elements at the edge to copy in a row should be given as a compile-time argument using
 *       -DPARTIAL_N0, e.g. -DPARTIAL_N0=2
 * @note Scatter function should be given as a compile-time argument using -DSCATTER_FUNCTION, e.g. -DSCATTER_FUNCTION=ADD
 * @note If the kernel should skip reading the output tensor, -DSKIP_OUTPUT_READ option should be provided.
 * @note Kernel name in uppercase letters should be provided as a compile-time argument, e.g. -DSCATTER_MP1D_2D_MPND
 *
 * @param[in]  updates_ptr                           Pointer to the updates tensor. Data Types: F32
 * @param[in]  updates_stride_x                      Stride of the updates tensor in X dimension (in bytes)
 * @param[in]  updates_step_x                        updates_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  updates_stride_y                      Stride of the updates tensor in Y dimension (in bytes)
 * @param[in]  updates_step_y                        updates_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  updates_offset_first_element_in_bytes The offset of the first element in the updates tensor
 * @param[in]  indices_ptr                           Pointer to the indices tensor. Data Types: S32
 * @param[in]  indices_stride_x                      Stride of the indices tensor in X dimension (in bytes)
 * @param[in]  indices_step_x                        indices_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  indices_stride_y                      Stride of the indices tensor in Y dimension (in bytes)
 * @param[in]  indices_step_y                        indices_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  indices_offset_first_element_in_bytes The offset of the first element in the indices tensor
 * @param[out] output_ptr                            Pointer to the destination tensor. Same as @p upt_ptr
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  upt_block_stride                      Update tensor data block stride in bytes
 * @param[in]  out_block_stride                      Output tensor data block stride in bytes
 */
__kernel void scatter_mp1d_2d_mpnd(
    IMAGE_DECLARATION(updates),
    IMAGE_DECLARATION(indices),
    IMAGE_DECLARATION(output),
    int upt_block_stride,
    int out_block_stride
    )
{
    const int out_shape[5] = {OUT_SHAPE_N_MINUS_1, OUT_SHAPE_N_MINUS_2, OUT_SHAPE_N_MINUS_3,
        OUT_SHAPE_N_MINUS_4, OUT_SHAPE_N_MINUS_5};

    const int x = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // x-coordinate in the tensor
    const int y = get_global_id(1); // collapsed y-coordinate (ignoring the outermost dimensions)

    const bool x_cond = (PARTIAL_N0 != 0 && get_global_id(0) == 0);

    uchar *ind_ptr_raw = indices_ptr + indices_offset_first_element_in_bytes;
    const uchar *out_ptr_raw =  output_ptr + output_offset_first_element_in_bytes
        + x * sizeof(DATA_TYPE) + y * output_stride_y;

    const uchar *upt_ptr_raw = updates_ptr + updates_offset_first_element_in_bytes
        + x * sizeof(DATA_TYPE) + y * updates_stride_y;

    for(int index_element = 0; index_element < NUM_INDICES; ++index_element)
    {
        const int *ind_ptr = (const int *) (ind_ptr_raw);

        // Out of bounds check
        bool out_of_bounds = false;
        LOOP_UNROLLING(int, i, 0, 1, INDEX_LENGTH,
        {
            if(ind_ptr[i] >= out_shape[i] || ind_ptr[i] < 0)
            {
                out_of_bounds = true;
            }
        });

        ind_ptr_raw += indices_stride_y;

        if(out_of_bounds)
        {
            continue;
        }

        // Index calculation
        int index = 0;
        LOOP_UNROLLING(int, i, 0, 1, INDEX_LENGTH,
        {
            index = index * out_shape[i] + ind_ptr[i];
        });

        DATA_TYPE *out_ptr = (DATA_TYPE *) (out_ptr_raw + index * out_block_stride);

        const DATA_TYPE *upt_ptr = (const DATA_TYPE *) (upt_ptr_raw + index_element * upt_block_stride);

        VEC_DATA_TYPE(DATA_TYPE, N0) data_in0 = VLOAD(N0)(0, (__global DATA_TYPE *) upt_ptr);

#ifdef SKIP_OUTPUT_READ
        STORE_VECTOR_SELECT(data_in, DATA_TYPE, (__global DATA_TYPE *) out_ptr, N0, PARTIAL_N0, x_cond);
#else // ifdef SKIP_OUTPUT_READ
        VEC_DATA_TYPE(DATA_TYPE, N0) data_out0 = VLOAD(N0)(0, (__global DATA_TYPE *) out_ptr);
        data_out0 = SCATTER_FUNCTION(data_out0, data_in0);

        STORE_VECTOR_SELECT(data_out, DATA_TYPE, (__global DATA_TYPE *) out_ptr, N0, PARTIAL_N0, x_cond);
#endif // ifdef SKIP_OUTPUT_READ
    }
}

#endif // SCATTER_MP1D_2D_MPND

#ifdef SCATTER1D_PARALLEL

// NOTE : This code is non-deterministic and can only be excecuted with the "update" ScatterFunction
// This code is currently unusued as it requires changes to the existing test suite.
/** Performs the Scatter1D operation with multiple threads.
 *  Similar to @ref scatter1D()
 */
__kernel void scatter1D_parallel(
    TENSOR4D_DECLARATION(updates),
    TENSOR4D_DECLARATION(indices),
    TENSOR4D_DECLARATION(output))
{
    // Currently 1D - only iterate through x dimension of indices.
    const int px = get_global_id(0);
    const int index_value = *(uchar*)(indices_ptr + indices_offset_first_element_in_bytes + (sizeof(int) * px));

    if(index_value < OUT_SHAPE_X)
    {
        const DATA_TYPE update = *(DATA_TYPE *)(updates_ptr + updates_offset_first_element_in_bytes + (sizeof(DATA_TYPE) * px));
        __global uchar *out_addr = output_ptr + indices_offset_first_element_in_bytes + (sizeof(DATA_TYPE) * index_value);
        *(__global DATA_TYPE *)(out_addr) = update;
    }
}

#endif // SCATTER1D_PARALLEL
