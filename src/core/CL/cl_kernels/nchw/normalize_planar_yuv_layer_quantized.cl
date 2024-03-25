/*
 * Copyright (c) 2018-2021 Arm Limited.
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

#if defined(DATA_TYPE) && defined(VEC_SIZE) && defined(OFFSET) && defined(SCALE)

#define TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#define OFFSET_FLT ((float)OFFSET)
#define SCALE_FLT ((float)SCALE)

#if defined(NUM_CHANNELS)

/** Apply normalize_planar_yuv layer on tensors with NCHW data layout.
 *
 * @note Data type should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE e.g. -DVEC_SIZE=8
 * @note The depth of the input tensor should be given as a preprocessor argument using -DNUM_CHANNELS e.g. -DNUM_CHANNELS=8
 * @note The quantization offset should be given as a preprocessor argument using -DOFFSET e.g. -DOFFSET=8
 * @note The quantization scale should be given as a preprocessor argument using -DSCALE e.g. -DSCALE=8
 *
 * @param[in]  src_ptr                            Pointer to the first source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED
 * @param[in]  src_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  mean_ptr                           Pointer to the mean source tensor. Supported data types: same as @p src_ptr
 * @param[in]  mean_stride_x                      Stride of the mean source tensor in X dimension (in bytes)
 * @param[in]  mean_step_x                        mean_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mean_offset_first_element_in_bytes The offset of the first element in the mean source tensor
 * @param[in]  std_ptr                            Pointer to the std tensor. Supported data types: same as @p src_ptr
 * @param[in]  std_stride_x                       Stride of the std tensor in X dimension (in bytes)
 * @param[in]  std_step_x                         std_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  std_offset_first_element_in_bytes  The offset of the first element in the var source tensor
 */
__kernel void normalize_planar_yuv_layer_q8_nchw(TENSOR3D_DECLARATION(src),
                                                 TENSOR3D_DECLARATION(dst),
                                                 VECTOR_DECLARATION(mean),
                                                 VECTOR_DECLARATION(std))
{
    Tensor3D src  = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst  = CONVERT_TO_TENSOR3D_STRUCT(dst);
    Vector   mean = CONVERT_TO_VECTOR_STRUCT(mean);
    Vector   std  = CONVERT_TO_VECTOR_STRUCT(std);

    const uint current_slice = get_global_id(2) % NUM_CHANNELS;

    VEC_DATA_TYPE(float, VEC_SIZE)
    curr_mean_flt = (VEC_DATA_TYPE(float, VEC_SIZE))(*((__global DATA_TYPE *)(mean.ptr + current_slice * sizeof(DATA_TYPE))));
    curr_mean_flt = round(curr_mean_flt - OFFSET_FLT) * SCALE_FLT;

    VEC_DATA_TYPE(float, VEC_SIZE)
    curr_std_flt = (VEC_DATA_TYPE(float, VEC_SIZE))(*((__global DATA_TYPE *)(std.ptr + current_slice * sizeof(DATA_TYPE))));
    curr_std_flt = round(curr_std_flt - OFFSET_FLT) * SCALE_FLT;

    VEC_DATA_TYPE(float, VEC_SIZE)
    data_flt = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src.ptr), VEC_DATA_TYPE(float, VEC_SIZE));
    data_flt = round(data_flt - OFFSET_FLT) * SCALE_FLT;

    // Perform normalization
    VEC_DATA_TYPE(float, VEC_SIZE)
    res_flt = (data_flt - curr_mean_flt) / curr_std_flt;

    const TYPE res_u8 = CONVERT_SAT(round(res_flt / SCALE_FLT) + OFFSET_FLT, TYPE);
    VSTORE(VEC_SIZE)
    (res_u8, 0, (__global DATA_TYPE *)dst.ptr);
}

#endif // defined(NUM_CHANNELS)
#endif // defined(DATA_TYPE) && defined(VEC_SIZE) && defined(OFFSET) && defined(SCALE)