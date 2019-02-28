/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
#define VEC_FLOAT VEC_DATA_TYPE(float, VEC_SIZE)
#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE)
#define VEC_UCHAR VEC_DATA_TYPE(uchar, VEC_SIZE)
#define CONVERT_RTE(x, type) (convert_##type##_rte((x)))
#define CONVERT_DOWN(x, type) CONVERT_RTE(x, type)
inline VEC_UCHAR requantize(VEC_UCHAR input, float in_offset, float out_offset, float in_scale, float out_scale)
{
    const VEC_FLOAT in_f32  = (CONVERT(input, VEC_FLOAT) - (VEC_FLOAT)((float)in_offset)) * (VEC_FLOAT)((float)in_scale);
    const VEC_FLOAT out_f32 = in_f32 / ((VEC_FLOAT)(float)out_scale) + ((VEC_FLOAT)((float)out_offset));
    const VEC_UCHAR res_u8  = CONVERT_SAT(CONVERT_DOWN(out_f32, VEC_INT), VEC_UCHAR);
    return res_u8;
}
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

#if defined(DATA_TYPE) && defined(VEC_SIZE)
#if defined(DEPTH) && defined(ELEMENT_SIZE)

#if defined(INPUT1_WIDTH)

#if ELEMENT_SIZE == 1
#define COND_DATA_TYPE char
#elif ELEMENT_SIZE == 2
#define COND_DATA_TYPE short
#elif ELEMENT_SIZE == 4
#define COND_DATA_TYPE int
#else // ELEMENT_SIZE
#error "Element size not supported"
#endif // ELEMENT_SIZE

#if VEC_SIZE == 2
#define SEQ ((int2)(0, 1))
#elif VEC_SIZE == 4
#define SEQ ((int4)(0, 1, 2, 3))
#elif VEC_SIZE == 8
#define SEQ ((int8)(0, 1, 2, 3, 4, 5, 6, 7))
#elif VEC_SIZE == 16
#define SEQ ((int16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
#else // VEC_SIZE
#error "Vector size not supported"
#endif // VEC_SIZE

/** This kernel concatenates two input tensors into the output tensor along the first dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note Vector size has to be passed at compile time using -DVEC_SIZE. i.e. -DVEC_SIZE=16
 * @note The offset for the first spatial dimension has to be passed at compile time using -DWIDTH_OFFSET. i.e. -DWIDTH_OFFSET=128
 * @note Tensor depth should be given as a preprocessor argument using -DDEPTH=size. e.g. -DDEPTH=16
 * @note First input tensor width should be given as a preprocessor argument using -DINPUT1_WIDTH=width. e.g. -DINPUT1_WIDTH=8
 *
 * @param[in]  src1_ptr                           Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/F32
 * @param[in]  src1_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src1_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src1_stride_w                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src1_step_w                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  src2_ptr                           Pointer to the source tensor. Supported data types: same as @p src1_ptr
 * @param[in]  src2_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src2_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src2_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src2_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src2_stride_w                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src2_step_w                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor. Supported data types: same as @p src1_ptr
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                         dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_w                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  src1_pad_right                     Right paddings of the first input tensor in unit of elements
 * @param[in]  src1_pad_left                      Left paddings of the second input tensor in unit of elements
 */
__kernel void concatenate_width_x2(
    TENSOR4D_DECLARATION(src1),
    TENSOR4D_DECLARATION(src2),
    TENSOR4D_DECLARATION(dst),
    uint src1_pad_right,
    uint src2_pad_left)
{
    Tensor4D dst = CONVERT_TO_TENSOR4D_STRUCT(dst, DEPTH);

    // Calculate input indices
    const int x  = get_global_id(0) * (int)VEC_SIZE;
    const int y  = get_global_id(1);
    const int z  = get_global_id(2) % (int)DEPTH;
    const int w  = get_global_id(2) / (int)DEPTH;
    const int x1 = min(x, (int)INPUT1_WIDTH + (int)src1_pad_right - (int)VEC_SIZE);
    const int x2 = max(x - (int)INPUT1_WIDTH, -(int)src2_pad_left);

    // Calculate inputs and output addresses
    const __global uchar *in1_ptr = src1_ptr + (int)src1_offset_first_element_in_bytes + x1 * (int)src1_stride_x + y * (int)src1_stride_y + z * (int)src1_stride_z + w * (int)src1_stride_w;
    const __global uchar *in2_ptr = src2_ptr + (int)src2_offset_first_element_in_bytes + x2 * (int)src2_stride_x + y * (int)src2_stride_y + z * (int)src2_stride_z + w * (int)src2_stride_w;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    src1_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in1_ptr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    src2_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in2_ptr);

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) && defined(OFFSET_IN2) && defined(SCALE_IN2)
    src1_values = requantize(src1_values, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT);
    src2_values = requantize(src2_values, OFFSET_IN2, OFFSET_OUT, SCALE_IN2, SCALE_OUT);
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) && defined(OFFSET_IN2) && defined(SCALE_IN2) */
    const VEC_DATA_TYPE(int, VEC_SIZE) x_coords        = SEQ + (VEC_DATA_TYPE(int, VEC_SIZE))(x);
    const VEC_DATA_TYPE(COND_DATA_TYPE, VEC_SIZE) cond = CONVERT(x_coords < (VEC_DATA_TYPE(int, VEC_SIZE))(INPUT1_WIDTH), VEC_DATA_TYPE(COND_DATA_TYPE, VEC_SIZE));
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) values    = select(src2_values, src1_values, cond);

    VSTORE(VEC_SIZE)
    (values, 0, (__global DATA_TYPE *)dst.ptr);
}

#if defined(INPUT2_WIDTH) && defined(INPUT3_WIDTH)
/** This kernel concatenates four input tensors into the output tensor along the first dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note Vector size has to be passed at compile time using -DVEC_SIZE. i.e. -DVEC_SIZE=16
 * @note The offset for the first spatial dimension has to be passed at compile time using -DWIDTH_OFFSET. i.e. -DWIDTH_OFFSET=128
 * @note Tensor depth should be given as a preprocessor argument using -DDEPTH=size. e.g. -DDEPTH=16
 * @note First input tensor width should be given as a preprocessor argument using -DINPUT1_WIDTH=width. e.g. -DINPUT1_WIDTH=8
 * @note Second input tensor width should be given as a preprocessor argument using -DINPUT2_WIDTH=width. e.g. -DINPUT2_WIDTH=8
 * @note Third input tensor width should be given as a preprocessor argument using -DINPUT3_WIDTH=width. e.g. -DINPUT3_WIDTH=8
 *
 * @param[in]  src1_ptr                           Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/F32
 * @param[in]  src1_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src1_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src1_stride_w                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src1_step_w                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  src2_ptr                           Pointer to the source tensor. Supported data types: same as @p src1_ptr
 * @param[in]  src2_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src2_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src2_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src2_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src2_stride_w                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src2_step_w                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  src3_ptr                           Pointer to the source tensor. Supported data types: same as @p src1_ptr
 * @param[in]  src3_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src3_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src3_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src3_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src3_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src3_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src3_stride_w                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src3_step_w                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src3_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  src4_ptr                           Pointer to the source tensor. Supported data types: same as @p src1_ptr
 * @param[in]  src4_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src4_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src4_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src4_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src4_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src4_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src4_stride_w                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src4_step_w                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src4_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor. Supported data types: same as @p src1_ptr
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                         dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_w                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  src1_pad_right                     Right paddings of the first input tensor in unit of elements
 * @param[in]  src2_pad_left                      Left paddings of the second input tensor in unit of elements
 * @param[in]  src2_pad_right                     Right paddings of the second input tensor in unit of elements
 * @param[in]  src3_pad_left                      Left paddings of the third input tensor in unit of elements
 * @param[in]  src3_pad_right                     Right paddings of the third input tensor in unit of elements
 * @param[in]  src4_pad_left                      Left paddings of the fourth input tensor in unit of elements
 */
__kernel void concatenate_width_x4(
    TENSOR4D_DECLARATION(src1),
    TENSOR4D_DECLARATION(src2),
    TENSOR4D_DECLARATION(src3),
    TENSOR4D_DECLARATION(src4),
    TENSOR4D_DECLARATION(dst),
    uint src1_pad_right,
    uint src2_pad_left,
    uint src2_pad_right,
    uint src3_pad_left,
    uint src3_pad_right,
    uint src4_pad_left)
{
    Tensor4D dst = CONVERT_TO_TENSOR4D_STRUCT(dst, DEPTH);

    // Calculate input indices
    const int x = get_global_id(0) * (int)VEC_SIZE;
    const int y = get_global_id(1);
    const int z = get_global_id(2) % (int)DEPTH;
    const int w = get_global_id(2) / (int)DEPTH;

    const int x1 = min(x, (int)INPUT1_WIDTH + (int)src1_pad_right - (int)VEC_SIZE);
    const int x2 = min(max(x - (int)INPUT1_WIDTH, -(int)src2_pad_left), (int)INPUT2_WIDTH + (int)src2_pad_right - (int)VEC_SIZE);
    const int x3 = min(max(x - (int)INPUT1_WIDTH - (int)INPUT2_WIDTH, -(int)src3_pad_left), (int)INPUT3_WIDTH + (int)src3_pad_right - (int)VEC_SIZE);
    const int x4 = max(x - (int)INPUT1_WIDTH - (int)INPUT2_WIDTH - (int)INPUT3_WIDTH, -(int)src4_pad_left);

    // Calculate inputs and output addresses
    const __global uchar *in1_ptr = src1_ptr + (int)src1_offset_first_element_in_bytes + x1 * (int)src1_stride_x + y * (int)src1_stride_y + z * (int)src1_stride_z + w * (int)src1_stride_w;
    const __global uchar *in2_ptr = src2_ptr + (int)src2_offset_first_element_in_bytes + x2 * (int)src2_stride_x + y * (int)src2_stride_y + z * (int)src2_stride_z + w * (int)src2_stride_w;
    const __global uchar *in3_ptr = src3_ptr + (int)src3_offset_first_element_in_bytes + x3 * (int)src3_stride_x + y * (int)src3_stride_y + z * (int)src3_stride_z + w * (int)src3_stride_w;
    const __global uchar *in4_ptr = src4_ptr + (int)src4_offset_first_element_in_bytes + x4 * (int)src4_stride_x + y * (int)src4_stride_y + z * (int)src4_stride_z + w * (int)src4_stride_w;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    src1_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in1_ptr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    src2_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in2_ptr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    src3_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in3_ptr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    src4_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in4_ptr);

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) && defined(OFFSET_IN2) && defined(SCALE_IN2) && defined(OFFSET_IN3) && defined(SCALE_IN3) && defined(OFFSET_IN4) && defined(SCALE_IN4)
    src1_values = requantize(src1_values, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT);
    src2_values = requantize(src2_values, OFFSET_IN2, OFFSET_OUT, SCALE_IN2, SCALE_OUT);
    src3_values = requantize(src3_values, OFFSET_IN3, OFFSET_OUT, SCALE_IN3, SCALE_OUT);
    src4_values = requantize(src4_values, OFFSET_IN4, OFFSET_OUT, SCALE_IN4, SCALE_OUT);
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) && defined(OFFSET_IN2) && defined(SCALE_IN2) && defined(OFFSET_IN3) && defined(SCALE_IN3) && defined(OFFSET_IN4) && defined(SCALE_IN4) */

    const VEC_DATA_TYPE(int, VEC_SIZE) x_coords = SEQ + (VEC_DATA_TYPE(int, VEC_SIZE))(x);

    const VEC_DATA_TYPE(COND_DATA_TYPE, VEC_SIZE) cond_in2 = CONVERT(x_coords < (VEC_DATA_TYPE(int, VEC_SIZE))(INPUT1_WIDTH), VEC_DATA_TYPE(COND_DATA_TYPE, VEC_SIZE));
    const VEC_DATA_TYPE(COND_DATA_TYPE, VEC_SIZE) cond_in3 = CONVERT(x_coords < (VEC_DATA_TYPE(int, VEC_SIZE))(INPUT1_WIDTH + INPUT2_WIDTH), VEC_DATA_TYPE(COND_DATA_TYPE, VEC_SIZE));
    const VEC_DATA_TYPE(COND_DATA_TYPE, VEC_SIZE) cond_in4 = CONVERT(x_coords < (VEC_DATA_TYPE(int, VEC_SIZE))(INPUT1_WIDTH + INPUT2_WIDTH + INPUT3_WIDTH), VEC_DATA_TYPE(COND_DATA_TYPE, VEC_SIZE));

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    values = select(src2_values, src1_values, cond_in2);
    values = select(src3_values, values, cond_in3);
    values = select(src4_values, values, cond_in4);

    VSTORE(VEC_SIZE)
    (values, 0, (__global DATA_TYPE *)dst.ptr);
}
#endif /* defined(INPUT2_WIDTH) && defined(INPUT3_WIDTH) */
#endif /* defined(INPUT1_WIDTH) */
#endif /* defined(DEPTH) && defined(ELEMENT_SIZE) */

#if defined(WIDTH_OFFSET) && defined(DEPTH)
/** This kernel concatenates the input tensor into the output tensor along the first dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note Vector size has to be passed at compile time using -DVEC_SIZE. i.e. -DVEC_SIZE=16
 * @note The offset for the first spatial dimension has to be passed at compile time using -DWIDTH_OFFSET. i.e. -DWIDTH_OFFSET=128
 * @note Tensor depth should be given as a preprocessor argument using -DDEPTH=size. e.g. -DDEPTH=16
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_w                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */

__kernel void concatenate_width(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst))
{
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, DEPTH);
    Tensor4D dst = CONVERT_TO_TENSOR4D_STRUCT(dst, DEPTH);

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    source_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src.ptr);

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
    const VEC_UCHAR out = requantize(source_values, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT);
    VSTORE(VEC_SIZE)
    (out, 0, (__global DATA_TYPE *)(dst.ptr) + WIDTH_OFFSET);
#else  /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */
    VSTORE(VEC_SIZE)
    (source_values, 0, (__global DATA_TYPE *)(dst.ptr) + WIDTH_OFFSET);
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */
}

#endif /* defined(WIDTH_OFFSET) && defined(DEPTH) */

/** This kernel concatenates the input tensor into the output tensor along the third dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note Vector size has to be passed at compile time using -DVEC_SIZE. i.e. -DVEC_SIZE=16
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16, F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  offsets                           The offsets to the first valid element of the output tensor in bytes
 */
__kernel void concatenate_depth(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    int3 offsets)
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    source_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, -offsets.x, -offsets.y, 0));

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
    source_values = requantize(source_values, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT);
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

    VSTORE(VEC_SIZE)
    (source_values, 0, (__global DATA_TYPE *)(dst.ptr + offsets.z));

}
#endif /* defined(DATA_TYPE) && defined(VEC_SIZE) */
