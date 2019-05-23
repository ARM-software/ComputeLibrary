/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#if defined(PAD00) && defined(PAD10) && defined(PAD20) && defined(PAD21) && defined(PAD30) && defined(DATA_TYPE) && defined(VEC_SIZE) // Compile time constants

/** Perform a padded copy of input tensor to the output tensor. Padding values are defined at compile time
 *
 * @attention The following variables must be passed at compile time:
 * -# -DPAD{d}{0,1} = padding before{0} and after{1} dimension d (d < 4)
 * -# -DDEPTH = The third dimension (depth) of the tensor (it is needed only if d == 3)
 * -# -DDATA_TYPE = Input and output datatypes.
 *
 * @param[in]  in_ptr                            Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
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
 */
__kernel void copy_pad_tensor(
    TENSOR3D_DECLARATION(in),
    TENSOR3D_DECLARATION(out))

{
    Tensor3D in  = CONVERT_TO_TENSOR3D_STRUCT(in);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

    const int offset_x = PAD00;
    const int offset_y = PAD10;
    const int offset_z = PAD20;

#if PAD30 > 0
    const size_t in_batch    = get_global_id(2) / DEPTH;
    const int    total_depth = DEPTH + PAD20 + PAD21;
    const int    offset_w    = PAD30 * total_depth + in_batch * (PAD20 + PAD21);
#else  // PAD30 == 0
    const int offset_w = 0;
#endif // PAD30

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in.ptr);

    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)tensor3D_offset(&out, offset_x, offset_y, offset_z + offset_w));
}
#endif // Compile time constants

#if defined(DATA_TYPE)
/** Performs a copy of input tensor to the output tensor.
 *
 * @param[in]  in_ptr                            Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
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
 */
__kernel void copy_tensor(
    TENSOR3D_DECLARATION(in),
    TENSOR3D_DECLARATION(out))
{
    Tensor3D in  = CONVERT_TO_TENSOR3D_STRUCT(in);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

#if defined(VEC_SIZE)

#if defined(LAST_ACCESSED_X)
    // Check if access on width gets out of bounds
    // If it does then shift access vector to access elements within bounds
    const int shift = max((int)(get_global_id(0) * VEC_SIZE) - (int)LAST_ACCESSED_X, 0);
    in.ptr -= shift * in.stride_x;
    out.ptr -= shift * out.stride_x;
#endif // defined(LAST_ACCESSED_X)

    // Load data
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in.ptr);

    // Store result
    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)out.ptr);
#else  // defined(VEC_SIZE)
    *((__global DATA_TYPE *)(out.ptr)) = *((__global DATA_TYPE *)(in.ptr));
#endif // defined(VEC_SIZE)
}
#endif // defined(DATA_TYPE)