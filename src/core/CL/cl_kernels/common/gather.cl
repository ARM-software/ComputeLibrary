/*
 * Copyright (c) 2018-2021, 2023-2024 Arm Limited.
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

#if defined(DATA_TYPE) && defined(AXIS)

/** Performs the Gather operation along the chosen axis
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Axis should be given as a preprocessor argument using -DAXIS=axis. e.g. -DAXIS=1
 * @attention Output tensor depth should be given as a preprocessor argument using -DOUTPUT_DIM_Z=size. e.g. -DOUTPUT_DIM_Z=16
 *
 *
 * @param[in]  input_ptr                             Pointer to the source tensor. Supported data types: All
 * @param[in]  input_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                          input_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  input_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                          input_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  input_stride_z                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_z                          input_stride_z * number of elements along Z processed per work item (in bytes)
 * @param[in]  input_stride_w                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_w                          input_stride_w * number of elements along W processed per work item (in bytes)
 * @param[in]  input_offset_first_element_in_bytes   Offset of the first element in the source tensor
 * @param[in]  indices_ptr                           Pointer to the indices vector. Supported data types: S32/U32.
 * @param[in]  indices_stride_x                      Stride of the indices vector in X dimension (in bytes)
 * @param[in]  indices_step_x                        input_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  indices_offset_first_element_in_bytes Offset of the first element in the indices vector
 * @param[out] output_ptr                            Pointer to the destination tensor. Supported data types: same as @p input_ptr
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
__kernel void gather(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(indices),
    TENSOR4D_DECLARATION(output))
{
    const int px = get_global_id(0);
    const int py = get_global_id(1);
    const int pz = get_global_id(2) % OUTPUT_DIM_Z;
    const int pw = (get_global_id(2) / OUTPUT_DIM_Z );

    const Tensor4D input   = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input);
    const Tensor4D indices = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(indices);
    Tensor4D       output  = CONVERT_TO_TENSOR4D_STRUCT(output, OUTPUT_DIM_Z);

#if AXIS == 0
#if INDICES_DIMS == 1
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, px, 0, 0, 0);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
    __global const uchar *input_addr = tensor4D_offset(&input, safe_index, py, pz, pw);
#elif INDICES_DIMS == 2
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, px, py, 0, 0);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
    __global const uchar *input_addr = tensor4D_offset(&input, safe_index, pz, pw, 0);
#elif INDICES_DIMS == 3
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, px, py, pz, 0);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
    __global const uchar *input_addr = tensor4D_offset(&input, safe_index, pw, 0, 0);
#elif INDICES_DIMS == 4
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, px, py, pz, pw);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
    __global const uchar *input_addr = tensor4D_offset(&input, safe_index, 0, 0, 0);
#endif //INDICES_DIMS

#elif AXIS == 1
#if INDICES_DIMS == 1
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, py, 0, 0, 0);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
     __global const uchar *input_addr = tensor4D_offset(&input, px, safe_index, pz, pw);
#elif INDICES_DIMS == 2
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, py, pz, 0, 0);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
    __global const uchar *input_addr = tensor4D_offset(&input, px, safe_index, pw, 0);
#elif INDICES_DIMS == 3
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, py, pz, pw, 0);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
    __global const uchar *input_addr = tensor4D_offset(&input, px, safe_index, 0, 0);
#endif //INDICES_DIMS

#elif AXIS == 2
#if INDICES_DIMS == 1
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, pz, 0, 0, 0);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
     __global const uchar *input_addr = tensor4D_offset(&input, px, py, safe_index, pw);
#elif INDICES_DIMS == 2
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, pz, pw, 0, 0);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
    __global const uchar *input_addr = tensor4D_offset(&input, px, py, safe_index, 0);
#endif //INDICES_DIMS

#elif AXIS == 3
#if INDICES_DIMS == 1
    const uint index                 = *(__global const uint *)tensor4D_offset(&indices, pw, 0, 0, 0);
    const uint safe_index            = select((uint)0, index, index < INDEX_LIMIT);
     __global const uchar *input_addr = tensor4D_offset(&input, px, py, pz, safe_index);
#endif //INDICES_DIMS

#endif //AXIS

    *(__global DATA_TYPE *)output.ptr = select((DATA_TYPE)0, *((__global const DATA_TYPE *)input_addr), (DATA_TYPE)(index < INDEX_LIMIT));
}

#endif //defined(DATA_TYPE) && defined(AXIS)
