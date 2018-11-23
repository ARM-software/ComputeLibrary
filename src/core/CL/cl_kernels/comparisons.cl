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

#define EQUAL(x, y) ((x) == (y))
#define NOTEQUAL(x, y) ((x) != (y))
#define GREATER(x, y) ((x) > (y))
#define GREATEREQUAL(x, y) ((x) >= (y))
#define LESS(x, y) ((x) < (y))
#define LESSEQUAL(x, y) ((x) <= (y))

#define DEFINE_KERNEL_STR(name) compare_##name
#define DEFINE_KERNEL(name) DEFINE_KERNEL_STR(name)

#define DEFINE_KERNEL_QUANTIZED_STR(name) compare_##name##_quantized
#define DEFINE_KERNEL_QUANTIZED(name) DEFINE_KERNEL_QUANTIZED_STR(name)

#if defined(DATA_TYPE) && defined(VEC_SIZE) && defined(OP) && defined(OP_NAME)
/** This function compares two tensors.
 *
 * @attention The inputs' data type need to be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @attention The comparison operation should be given as a preprocessor argument using -DOP=operation. e.g. -DOP=LESS
 *
 * @param[in]  in1_ptr                           Pointer to the source tensor. Supported data types: U8/S16/F16/F32
 * @param[in]  in1_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  in1_step_z                        in1_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  in2_ptr                           Pointer to the source tensor. Supported data types: U8/S16/F16/F32
 * @param[in]  in2_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  in2_step_z                        in2_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: U8 (only if both inputs are U8), S16/F16/F32
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void DEFINE_KERNEL(OP_NAME)(
    TENSOR3D_DECLARATION(in1),
    TENSOR3D_DECLARATION(in2),
    TENSOR3D_DECLARATION(out))
{
    // Get pixels pointer
    Tensor3D in1 = CONVERT_TO_TENSOR3D_STRUCT(in1);
    Tensor3D in2 = CONVERT_TO_TENSOR3D_STRUCT(in2);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

    // Load values
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_a = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in1.ptr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_b = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in2.ptr);

    // Calculate and store result
    VSTORE(VEC_SIZE)
    (CONVERT(OP(in_a, in_b), VEC_DATA_TYPE(uchar, VEC_SIZE)), 0, (__global uchar *)out.ptr);
}
#endif /* defined(DATA_TYPE) && defined(VEC_SIZE) && defined(OP) && defined(OP_NAME) */

#if defined(OFFSET_IN1) && defined(OFFSET_IN2) && defined(SCALE_IN1) && defined(SCALE_IN2)
/** This function compares two quantized tensors.
 *
 * @note The quantization offset of the first operand must be passed at compile time using -DOFFSET_IN1, i.e. -DOFFSET_IN1=10
 * @note The quantization offset of the second operand must be passed at compile time using -DOFFSET_IN2, i.e. -DOFFSET_IN2=10
 * @note The quantization scale of the first operand must be passed at compile time using -DSCALE_IN1, i.e. -DSCALE_IN1=10
 * @note The quantization scale of the second operand must be passed at compile time using -DSCALE_IN2, i.e. -DSCALE_IN2=10
 *
 * @param[in]  in1_ptr                           Pointer to the source tensor. Supported data types: QASYMM8
 * @param[in]  in1_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  in1_step_z                        in1_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  in2_ptr                           Pointer to the source tensor. Supported data types: same as @p in1_ptr
 * @param[in]  in2_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  in2_step_z                        in2_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: same as @p in1_ptr
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void DEFINE_KERNEL_QUANTIZED(OP_NAME)(
    TENSOR3D_DECLARATION(in1),
    TENSOR3D_DECLARATION(in2),
    TENSOR3D_DECLARATION(out))
{
    // Get pixels pointer
    Tensor3D in1 = CONVERT_TO_TENSOR3D_STRUCT(in1);
    Tensor3D in2 = CONVERT_TO_TENSOR3D_STRUCT(in2);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

    int16 in_a = CONVERT(vload16(0, (__global uchar *)in1.ptr), int16);
    int16 in_b = CONVERT(vload16(0, (__global uchar *)in2.ptr), int16);

    in_a = in_a - (int16)((int)OFFSET_IN1);
    in_b = in_b - (int16)((int)OFFSET_IN2);

    const float16 in1f32 = convert_float16(in_a) * (float16)((float)SCALE_IN1);
    const float16 in2f32 = convert_float16(in_b) * (float16)((float)SCALE_IN2);
    const int16   res    = OP(in1f32, in2f32);

    // Store result
    vstore16(convert_uchar16(res), 0, (__global uchar *)out.ptr);
}
#endif /* defined(OFFSET_IN1) && defined(OFFSET_IN2) && defined(SCALE_IN1) && defined(SCALE_IN2) */