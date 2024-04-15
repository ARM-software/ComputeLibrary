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

#if defined(DATA_TYPE) && defined(OPERATION)
// Calculate reverse square root
#define rsqrt_op(input) rsqrt(input)
#if defined(VEC_SIZE)
#define VEC_FLOAT VEC_DATA_TYPE(float, VEC_SIZE)
#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE)
#define VEC_TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#endif // defined(VEC_SIZE)

/** Applies element wise unary operator in a tensor.
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: QASYMM8/QASYMM8_SIGNED.
 * @param[in]  in_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  in_step_y                         in_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  in_step_z                         in_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  Offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: QASYMM8/QASYMM8_SIGNED.
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_step_y                        Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes Offset of the first element in the destination image
 */
__kernel void elementwise_unary_quantized(
    TENSOR3D_DECLARATION(in),
    TENSOR3D_DECLARATION(out))
{
    Tensor3D in  = CONVERT_TO_TENSOR3D_STRUCT(in);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

    // Check if access on width gets out of bounds
    // If it does shift access vector to access elements within bounds
    const int xi = (int)(get_global_id(0) * VEC_SIZE);
    in.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * in_stride_x;
    out.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * out_stride_x;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in.ptr);
    VEC_DATA_TYPE(float, VEC_SIZE)
    data_f32                = CONVERT(data, VEC_FLOAT);
    data_f32                = (data_f32 - (float)OFFSET_IN) * (float)SCALE_IN;
    VEC_INT        qres_int = CONVERT_SAT((OPERATION(data_f32) / ((VEC_FLOAT)(float)SCALE_OUT)), VEC_INT) + ((VEC_INT)((int)OFFSET_OUT));
    const VEC_TYPE qres     = CONVERT_SAT(qres_int, VEC_TYPE);
    VSTORE(VEC_SIZE)
    (qres, 0, (__global DATA_TYPE *)out.ptr);
}
#endif // defined(DATA_TYPE) && defined(OPERATION)
