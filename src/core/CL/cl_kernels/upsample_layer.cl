/*
 * Copyright (c) 2018 ARM Limited.
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

/** This function applies upsample on an input image. (NCHW)
 *
 * @attention The following variables must be passed at compile time:
 * -# -DDATA_TYPE = Tensor data type. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * -# -DVEC_SIZE_IN = Input vector size
 * -# -DVEC_SIZE_OUT = Output vector size
 * -# -DLAST_ACCESSED_X_IN = The input element that is on the X border (threads trying to set this, might need to step back a bit)
 * -# -DLAST_ACCESSED_X_OUT = The output element that is on the X border (threads trying to set this, might need to step back a bit)
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void upsample_layer_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

#if defined(VEC_SIZE_IN) && defined(VEC_SIZE_OUT) && defined(LAST_ACCESSED_X_IN) && defined(LAST_ACCESSED_X_OUT)
    // Check if access on width gets out of bounds
    // If it does shift access vector to access elements within bounds
    const int xi_in  = (int)(get_global_id(0) * VEC_SIZE_IN);
    const int xi_out = (int)(get_global_id(0) * VEC_SIZE_OUT);
    src.ptr -= max(xi_in - (int)LAST_ACCESSED_X_IN, 0) * src_stride_x;
    dst.ptr -= max(xi_out - (int)LAST_ACCESSED_X_OUT, 0) * dst_stride_x;

    VEC_DATA_TYPE(DATA_TYPE, 8)
    data = vload8(0, (__global DATA_TYPE *)src.ptr);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    data_out = (VEC_DATA_TYPE(DATA_TYPE, 16))(data.s0, data.s0, data.s1, data.s1, data.s2, data.s2, data.s3, data.s3, data.s4, data.s4, data.s5, data.s5, data.s6, data.s6, data.s7, data.s7);

    vstore16(data_out, 0, (__global DATA_TYPE *)dst.ptr);
    vstore16(data_out, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, 0, 1, 0));
#else  // !defined(VEC_SIZE_IN) && defined(VEC_SIZE_OUT) && defined(LAST_ACCESSED_X_IN) && defined(LAST_ACCESSED_X_OUT)
    *((__global DATA_TYPE *)tensor3D_offset(&dst, 0, 0, 0)) = *((__global DATA_TYPE *)src.ptr);
    *((__global DATA_TYPE *)tensor3D_offset(&dst, 0, 1, 0)) = *((__global DATA_TYPE *)src.ptr);
#endif // defined(VEC_SIZE_IN) && defined(VEC_SIZE_OUT) && defined(LAST_ACCESSED_X_IN) && defined(LAST_ACCESSED_X_OUT)
}

/** This function applies upsample on an input image. (NHWC)
 *
 * @attention The following variables must be passed at compile time:
 * -# -DDATA_TYPE = Tensor data type. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * -# -DVEC_SIZE_IN = Input vector size
 * -# -DVEC_SIZE_OUT = Output vector size
 * -# -DLAST_ACCESSED_X_IN = The input element that is on the X border (threads trying to set this, might need to step back a bit)
 * -# -DLAST_ACCESSED_X_OUT = The output element that is on the X border (threads trying to set this, might need to step back a bit)
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void upsample_layer_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

#if defined(VEC_SIZE_IN) && defined(VEC_SIZE_OUT) && defined(LAST_ACCESSED_X_IN) && defined(LAST_ACCESSED_X_OUT)
    // Check if access on width gets out of bounds
    // If it does shift access vector to access elements within bounds
    const int xi_in  = (int)(get_global_id(0) * VEC_SIZE_IN);
    const int xi_out = (int)(get_global_id(0) * VEC_SIZE_OUT);
    src.ptr -= max(xi_in - (int)LAST_ACCESSED_X_IN, 0) * src_stride_x;
    dst.ptr -= max(xi_out - (int)LAST_ACCESSED_X_OUT, 0) * dst_stride_x;

    VEC_DATA_TYPE(DATA_TYPE, 16)
    data = vload16(0, (__global DATA_TYPE *)src.ptr);

    vstore16(data, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, 0, 0, 0));
    vstore16(data, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, 0, 1, 0));
    vstore16(data, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, 0, 0, 1));
    vstore16(data, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, 0, 1, 1));
#else  // !defined(VEC_SIZE_IN) && defined(VEC_SIZE_OUT) && defined(LAST_ACCESSED_X_IN) && defined(LAST_ACCESSED_X_OUT)
    *((__global DATA_TYPE *)tensor3D_offset(&dst, 0, 0, 0)) = *((__global DATA_TYPE *)src.ptr);
    *((__global DATA_TYPE *)tensor3D_offset(&dst, 0, 1, 0)) = *((__global DATA_TYPE *)src.ptr);
    *((__global DATA_TYPE *)tensor3D_offset(&dst, 0, 0, 1)) = *((__global DATA_TYPE *)src.ptr);
    *((__global DATA_TYPE *)tensor3D_offset(&dst, 0, 1, 1)) = *((__global DATA_TYPE *)src.ptr);
#endif // defined(VEC_SIZE_IN) && defined(VEC_SIZE_OUT) && defined(LAST_ACCESSED_X_IN) && defined(LAST_ACCESSED_X_OUT)
}
