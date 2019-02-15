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
#include "helpers.h"

#if defined(DATA_TYPE) // Compile time constants

/** Performs a copy of input tensor to the output tensor.
 *
 * @param[in]  in_ptr                            Pointer to the source tensor. Supported data types: U16/S16/F16/U32/S32/F32
 * @param[in]  in_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  in_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  in_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  in_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: same as @p in_ptr
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  out_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  out_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  in_offset_y                       The initial offset of the input address along Y.
 * @param[in]  in_offset_z                       The initial offset of the input address along Z.
 */
__kernel void crop_tensor(
    TENSOR3D_DECLARATION(in),
    TENSOR3D_DECLARATION(out),
    int in_offset_y,
    int in_offset_z)
{
    Tensor3D in  = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(in);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

    const int in_x = get_global_id(0) * (in_step_x / in_stride_x);

#if defined(WIDTH_FLIPPED)
    const int in_y = in_offset_y - get_global_id(1);
#else  // defined(WIDTH_FLIPPED)
    const int in_y                 = in_offset_y + get_global_id(1);
#endif // defined(WIDTH_FLIPPED)

#if defined(HEIGHT_FLIPPED)
    const int in_z = in_offset_z - get_global_id(2);
#else  // defined(HEIGHT_FLIPPED)
    const int in_z                 = in_offset_z + get_global_id(2);
#endif // defined(HEIGHT_FLIPPED)

#if defined(VEC_SIZE)

#if defined(LAST_ACCESSED_X)
    // Check if access on width gets out of bounds
    // If it does then shift access vector to access elements within bounds
    const int shift = max((int)(get_global_id(0) * VEC_SIZE) - (int)LAST_ACCESSED_X, 0);
    in.ptr -= shift * in.stride_x;
    out.ptr -= shift * out.stride_x;
#endif // defined(LAST_ACCESSED_X)

    __global const uchar *input_addr = tensor3D_offset(&in, in_x, in_y, in_z);

    // Load data
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr);

    // Store result
    VSTORE(VEC_SIZE)
    (CONVERT(data, VEC_DATA_TYPE(float, VEC_SIZE)), 0, (__global float *)out.ptr);
#else  // defined(VEC_SIZE)
    *((__global float *)(out.ptr)) = CONVERT(*((__global DATA_TYPE *)tensor3D_offset(&in, in_x, in_y, in_z)), float);
#endif // defined(VEC_SIZE)
}

#endif // defined(DATA_TYPE) && defined(LAST_ACCESSED_X)