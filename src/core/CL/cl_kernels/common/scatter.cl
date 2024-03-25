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

#if defined(INDICES_SHAPE_Y) && defined(DATA_TYPE) &&  defined(OUT_SHAPE_X) && defined(SCATTER_FUNCTION)

// The below defines the various reduce operations for our purposes.
// Where a corresponds to the existing value, and b the new value.
#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))
#define MAX_OP(a, b) fmax(a, b)
#define MIN_OP(a, b) fmin(a, b)
#define UPDATE_OP(a, b) (b)

/** Performs the ScatterND operation
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note the size of the dst tensor in the "x" dimension should be passed using -DOUT_SHAPE_X at compile time.
 * @note the number of values in the indices tensor in the y-dim should be passed with -DINDICES_SHAPE_Y at compile time.
 * @note Negative indices are treated as out of bounds.
 *
 * @param[in]  updates_ptr                           Pointer to the source tensor. Supported data types: All
 * @param[in]  updates_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  updates_step_x                        updates_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  updates_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  updates_step_y                        updates_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  updates_stride_z                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  updates_step_z                        updates_stride_z * number of elements along Z processed per work item (in bytes)
 * @param[in]  updates_stride_w                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  updates_step_w                        updates_stride_w * number of elements along W processed per work item (in bytes)
 * @param[in]  updates_offset_first_element_in_bytes Offset of the first element in the source tensor
 * @param[in]  indices_ptr                           Pointer to the indices vector. Supported data types: S32.
 * @param[in]  indices_stride_x                      Stride of the indices vector in X dimension (in bytes)
 * @param[in]  indices_step_x                        updates_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  indices_offset_first_element_in_bytes Offset of the first element in the indices vector
 * @param[out] output_ptr                            Pointer to the destination tensor. Supported data types: same as @p updates_ptr
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  output_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z processed per work item (in bytes)
 * @param[in]  output_stride_w                       Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  output_step_w                         output_stride_w * number of elements along W processed per work item (in bytes)
 * @param[in]  output_offset_first_element_in_bytes  Offset of the first element in the destination tensor
 */
// The below kernel code is expected to be excecuted sequentially with a single thread to ensure a deterministic outcome.
__kernel void scatter1D(
    TENSOR4D_DECLARATION(updates),
    TENSOR4D_DECLARATION(indices),
    TENSOR4D_DECLARATION(output))
{
    // Currently 1D - only iterate through y dimension of indices.
    unsigned int* indices_start_offset = (unsigned int*)(indices_ptr + indices_offset_first_element_in_bytes);
    DATA_TYPE* updates_start_offset = (DATA_TYPE*)(updates_ptr + updates_offset_first_element_in_bytes);
    DATA_TYPE* out_start_offset = (DATA_TYPE*)(output_ptr + output_offset_first_element_in_bytes);
    for (int px = 0; px < INDICES_SHAPE_Y; px++)
    {
        const int index_value = *(indices_start_offset);
        DATA_TYPE* out_addr = out_start_offset + index_value;
        if((index_value < OUT_SHAPE_X) && (index_value >= 0))
        {
            *(__global DATA_TYPE *)(out_addr) = SCATTER_FUNCTION(*(out_addr), *updates_start_offset);
        }
        // Increment pointers.
        indices_start_offset++;
        updates_start_offset++;
    }
}

#endif //defined(DATA_TYPE) && defined(SCATTER_FUNCTION) && defined(OUT_SHAPE_X) && defined(INDICES_SHAPE_Y)

#if defined(DATA_TYPE) && defined(SCATTER_FUNCTION) && defined(OUT_SHAPE_X) && !defined(INDICES_SHAPE_Y)

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

#endif //defined(DATA_TYPE) && defined(SCATTER_FUNCTION) && defined(OUT_SHAPE_X) && !defined(INDICES_SHAPE_Y)
