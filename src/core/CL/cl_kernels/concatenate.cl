/*
 * Copyright (c) 2017-2020 Arm Limited.
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

#if defined(VEC_SIZE)
#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE)

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
#define VEC_FLOAT VEC_DATA_TYPE(float, VEC_SIZE)
#define VEC_QUANT VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#define CONVERT_RTE(x, type) (convert_##type##_rte((x)))
#define CONVERT_DOWN(x, type) CONVERT_RTE(x, type)
inline VEC_QUANT requantize(VEC_QUANT input, float in_offset, float out_offset, float in_scale, float out_scale)
{
    const VEC_FLOAT in_f32  = (CONVERT(input, VEC_FLOAT) - (VEC_FLOAT)((float)in_offset)) * (VEC_FLOAT)((float)in_scale);
    const VEC_FLOAT out_f32 = in_f32 / ((VEC_FLOAT)(float)out_scale) + ((VEC_FLOAT)((float)out_offset));
    const VEC_QUANT res_q8  = CONVERT_SAT(CONVERT_DOWN(out_f32, VEC_INT), VEC_QUANT);
    return res_q8;
}
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

#if defined(DATA_TYPE)
#define VEC_TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)

#if defined(DEPTH) && defined(ELEMENT_SIZE)
#if defined(INPUT1_WIDTH)

#define SELECT_TYPE SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#define SEQ VEC_OFFS(int, VEC_SIZE)

/** This kernel concatenates two input tensors into the output tensor along the first dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note Vector size has to be passed at compile time using -DVEC_SIZE. i.e. -DVEC_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note Tensor depth should be given as a preprocessor argument using -DDEPTH=size. e.g. -DDEPTH=16
 * @note First input tensor width should be given as a preprocessor argument using -DINPUT1_WIDTH=width. e.g. -DINPUT1_WIDTH=8
 *
 * @param[in]  src1_ptr                           Pointer to the source tensor. Supported data types: All.
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
 */
__kernel void concatenate_width_x2(
    TENSOR4D_DECLARATION(src1),
    TENSOR4D_DECLARATION(src2),
    TENSOR4D_DECLARATION(dst))
{
    // Calculate input indices
    const int x  = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    const int y  = get_global_id(1);
    const int z  = get_global_id(2) % (int)DEPTH;
    const int w  = get_global_id(2) / (int)DEPTH;
    const int x1 = min(x, (int)INPUT1_WIDTH - (int)VEC_SIZE);
    const int x2 = max(x - (int)INPUT1_WIDTH, 0);

    // Calculate inputs and output addresses
    const __global uchar *dst_addr  = dst_ptr + (int)dst_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * (int)dst_stride_y + z * (int)dst_stride_z + w * (int)dst_stride_w;
    const __global uchar *src1_addr = src1_ptr + (int)src1_offset_first_element_in_bytes + x1 * sizeof(DATA_TYPE) + y * (int)src1_stride_y + z * (int)src1_stride_z + w * (int)src1_stride_w;
    const __global uchar *src2_addr = src2_ptr + (int)src2_offset_first_element_in_bytes + x2 * sizeof(DATA_TYPE) + y * (int)src2_stride_y + z * (int)src2_stride_z + w * (int)src2_stride_w;

    VEC_TYPE src1_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src1_addr);
    VEC_TYPE src2_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src2_addr);

#if defined(OFFSET_IN1) && defined(OFFSET_IN2) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_IN2) && defined(SCALE_OUT)
    src1_values = requantize(src1_values, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT);
    src2_values = requantize(src2_values, OFFSET_IN2, OFFSET_OUT, SCALE_IN2, SCALE_OUT);
#endif /* defined(OFFSET_IN1) && defined(OFFSET_IN2) && defined(OFFSET_OUT) && defined(SCALE_IN1)  && defined(SCALE_IN2) && defined(SCALE_OUT) */
    const VEC_INT x_coords = SEQ + (VEC_INT)(x);

    // Rotate src1/2_values, if values0 is a combination of src1_values and src2_values.
    SELECT_TYPE cond = CONVERT(((VEC_INT)x < (VEC_INT)INPUT1_WIDTH) && ((VEC_INT)x > (VEC_INT)(INPUT1_WIDTH - VEC_SIZE)), SELECT_TYPE);
    src1_values      = select(src1_values, ROTATE(src1_values, VEC_SIZE, INPUT1_ROTATE_N), cond);
    src2_values      = select(src2_values, ROTATE(src2_values, VEC_SIZE, INPUT1_ROTATE_N), cond);

    cond                   = CONVERT(x_coords < (VEC_INT)(INPUT1_WIDTH), SELECT_TYPE);
    const VEC_TYPE values0 = select(src2_values, src1_values, cond);

    STORE_VECTOR_SELECT(values, DATA_TYPE, dst_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
}

#if defined(INPUT2_WIDTH) && defined(INPUT3_WIDTH)
/** This kernel concatenates four input tensors into the output tensor along the first dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note Vector size has to be passed at compile time using -DVEC_SIZE. i.e. -DVEC_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note Tensor depth should be given as a preprocessor argument using -DDEPTH=size. e.g. -DDEPTH=16
 * @note First input tensor width should be given as a preprocessor argument using -DINPUT1_WIDTH=width. e.g. -DINPUT1_WIDTH=8
 * @note Second input tensor width should be given as a preprocessor argument using -DINPUT2_WIDTH=width. e.g. -DINPUT2_WIDTH=8
 * @note Third input tensor width should be given as a preprocessor argument using -DINPUT3_WIDTH=width. e.g. -DINPUT3_WIDTH=8
 *
 * @param[in]  src1_ptr                           Pointer to the source tensor. Supported data types: All
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
 */
__kernel void concatenate_width_x4(
    TENSOR4D_DECLARATION(src1),
    TENSOR4D_DECLARATION(src2),
    TENSOR4D_DECLARATION(src3),
    TENSOR4D_DECLARATION(src4),
    TENSOR4D_DECLARATION(dst))
{
    // Calculate input indices
    const int x = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    const int y = get_global_id(1);
    const int z = get_global_id(2) % (int)DEPTH;
    const int w = get_global_id(2) / (int)DEPTH;

    const int x1 = min(x, (int)INPUT1_WIDTH - (int)VEC_SIZE);
    const int x2 = min(max(x - (int)INPUT1_WIDTH, 0), (int)INPUT2_WIDTH - (int)VEC_SIZE);
    const int x3 = min(max(x - (int)INPUT1_WIDTH - (int)INPUT2_WIDTH, 0), (int)INPUT3_WIDTH - (int)VEC_SIZE);
    const int x4 = max(x - (int)INPUT1_WIDTH - (int)INPUT2_WIDTH - (int)INPUT3_WIDTH, 0);

    // Calculate inputs and output addresses
    const __global uchar *dst_addr  = dst_ptr + (int)dst_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * (int)dst_stride_y + z * (int)dst_stride_z + w * (int)dst_stride_w;
    const __global uchar *src1_addr = src1_ptr + (int)src1_offset_first_element_in_bytes + x1 * sizeof(DATA_TYPE) + y * (int)src1_stride_y + z * (int)src1_stride_z + w * (int)src1_stride_w;
    const __global uchar *src2_addr = src2_ptr + (int)src2_offset_first_element_in_bytes + x2 * sizeof(DATA_TYPE) + y * (int)src2_stride_y + z * (int)src2_stride_z + w * (int)src2_stride_w;
    const __global uchar *src3_addr = src3_ptr + (int)src3_offset_first_element_in_bytes + x3 * sizeof(DATA_TYPE) + y * (int)src3_stride_y + z * (int)src3_stride_z + w * (int)src3_stride_w;
    const __global uchar *src4_addr = src4_ptr + (int)src4_offset_first_element_in_bytes + x4 * sizeof(DATA_TYPE) + y * (int)src4_stride_y + z * (int)src4_stride_z + w * (int)src4_stride_w;

    VEC_TYPE src1_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src1_addr);
    VEC_TYPE src2_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src2_addr);
    VEC_TYPE src3_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src3_addr);
    VEC_TYPE src4_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src4_addr);

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) && defined(OFFSET_IN2) && defined(SCALE_IN2) && defined(OFFSET_IN3) && defined(SCALE_IN3) && defined(OFFSET_IN4) && defined(SCALE_IN4)
    src1_values = requantize(src1_values, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT);
    src2_values = requantize(src2_values, OFFSET_IN2, OFFSET_OUT, SCALE_IN2, SCALE_OUT);
    src3_values = requantize(src3_values, OFFSET_IN3, OFFSET_OUT, SCALE_IN3, SCALE_OUT);
    src4_values = requantize(src4_values, OFFSET_IN4, OFFSET_OUT, SCALE_IN4, SCALE_OUT);
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) && defined(OFFSET_IN2) && defined(SCALE_IN2) && defined(OFFSET_IN3) && defined(SCALE_IN3) && defined(OFFSET_IN4) && defined(SCALE_IN4) */

    const VEC_INT x_coords = SEQ + (VEC_INT)(x);

    SELECT_TYPE cond_in2 = CONVERT(((VEC_INT)x < (VEC_INT)INPUT1_WIDTH && (VEC_INT)x > (VEC_INT)(INPUT1_WIDTH - VEC_SIZE)), SELECT_TYPE);
    SELECT_TYPE cond_in3 = CONVERT(((VEC_INT)x < (VEC_INT)(INPUT1_WIDTH + INPUT2_WIDTH) && (VEC_INT)x > (VEC_INT)(INPUT1_WIDTH + INPUT2_WIDTH - VEC_SIZE)), SELECT_TYPE);
    SELECT_TYPE cond_in4 = CONVERT(((VEC_INT)x < (VEC_INT)(INPUT1_WIDTH + INPUT2_WIDTH + INPUT3_WIDTH) && (VEC_INT)x > (VEC_INT)(INPUT1_WIDTH + INPUT2_WIDTH + INPUT3_WIDTH - VEC_SIZE)), SELECT_TYPE);

    // Rotate src1/2_values, if values0 is a combination of src1_values and src2_values.
    src1_values = select(src1_values, ROTATE(src1_values, VEC_SIZE, INPUT1_ROTATE_N), cond_in2);
    src2_values = select(src2_values, ROTATE(src2_values, VEC_SIZE, INPUT1_ROTATE_N), cond_in2);
    // Rotate src2/3_values, if values0 is a combination of src2_values and src3_values.
    src2_values = select(src2_values, ROTATE(src2_values, VEC_SIZE, INPUT2_ROTATE_N), cond_in3);
    src3_values = select(src3_values, ROTATE(src3_values, VEC_SIZE, INPUT2_ROTATE_N), cond_in3);
    // Rotate src3/4_values, if values0 is a combination of src3_values and src4_values.
    src3_values = select(src3_values, ROTATE(src3_values, VEC_SIZE, INPUT3_ROTATE_N), cond_in4);
    src4_values = select(src4_values, ROTATE(src4_values, VEC_SIZE, INPUT3_ROTATE_N), cond_in4);

    cond_in2 = CONVERT(x_coords < (VEC_INT)(INPUT1_WIDTH), SELECT_TYPE);
    cond_in3 = CONVERT(x_coords < (VEC_INT)(INPUT1_WIDTH + INPUT2_WIDTH), SELECT_TYPE);
    cond_in4 = CONVERT(x_coords < (VEC_INT)(INPUT1_WIDTH + INPUT2_WIDTH + INPUT3_WIDTH), SELECT_TYPE);

    VEC_TYPE values0 = select(src2_values, src1_values, cond_in2);
    values0          = select(src3_values, values0, cond_in3);
    values0          = select(src4_values, values0, cond_in4);

    STORE_VECTOR_SELECT(values, DATA_TYPE, dst_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
}
#endif /* defined(INPUT2_WIDTH) && defined(INPUT3_WIDTH) */
#endif /* defined(INPUT1_WIDTH) */
#endif /* defined(DEPTH) && defined(ELEMENT_SIZE) */

#if defined(WIDTH_OFFSET) && defined(DEPTH) && defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER)
/** This kernel concatenates the input tensor into the output tensor along the first dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note Vector size has to be passed at compile time using -DVEC_SIZE. i.e. -DVEC_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
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
    // Calculate input indices
    const int x = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    const int y = get_global_id(1);
    const int z = get_global_id(2) % (int)DEPTH;
    const int w = get_global_id(2) / (int)DEPTH;

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * src_stride_y + z * src_stride_z + w * src_stride_w;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z + w * dst_stride_w;

    VEC_TYPE source_values0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src_addr);

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
    const VEC_QUANT out0 = requantize(source_values0, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT);
    STORE_VECTOR_SELECT(out, DATA_TYPE, dst_addr + WIDTH_OFFSET * sizeof(DATA_TYPE), VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#else  /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */
    STORE_VECTOR_SELECT(source_values, DATA_TYPE, dst_addr + WIDTH_OFFSET * sizeof(DATA_TYPE), VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */
}

#endif /* defined(WIDTH_OFFSET) && defined(DEPTH) && defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER)*/

#if defined(VEC_SIZE_LEFTOVER)

#if defined(HEIGHT_OFFSET) && defined(DEPTH) && defined(VEC_SIZE)
/** This kernel concatenates the input tensor into the output tensor along the second dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note Vector size has to be passed at compile time using -DVEC_SIZE. i.e. -DVEC_SIZE=16
 * @note Vector sizes supported are 2,4,8 and 16.
 * @note The offset for the second spatial dimension has to be passed at compile time using -DHEIGHT_OFFSET. i.e. -DHEIGHT_OFFSET=128
 * @note Tensor depth should be given as a preprocessor argument using -DDEPTH=size. e.g. -DDEPTH=16
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
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

__kernel void concatenate_height(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst))
{
    const int x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0) * sizeof(DATA_TYPE);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x_offs + get_global_id(1) * src_stride_y + (get_global_id(2) % DEPTH) * src_stride_z + (get_global_id(
                                   2) / DEPTH) * src_stride_w;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x_offs + get_global_id(1) * dst_stride_y + (get_global_id(2) % DEPTH) * dst_stride_z + (get_global_id(
                                   2) / DEPTH) * dst_stride_w;

    VEC_TYPE source_values0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src_addr);

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
    const VEC_QUANT out0 = requantize(source_values0, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT);
    STORE_VECTOR_SELECT(out, DATA_TYPE, dst_addr + HEIGHT_OFFSET * dst_stride_y, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#else  /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */
    STORE_VECTOR_SELECT(source_values, DATA_TYPE, dst_addr + HEIGHT_OFFSET * dst_stride_y, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */
}

#endif /* defined(HEIGHT_OFFSET) && defined(DEPTH) */

/** This kernel concatenates the input tensor into the output tensor along the third dimension
 *
 * @note The data type has to be passed at compile time using -DDATA_TYPE. i.e. -DDATA_TYPE=float
 * @note Vector size has to be passed at compile time using -DVEC_SIZE. i.e. -DVEC_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: All
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
__kernel void concatenate(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    int offset)
{
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE * sizeof(DATA_TYPE) - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE * sizeof(DATA_TYPE)), 0);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x_offs + get_global_id(1) * src_stride_y + get_global_id(2) * src_stride_z;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x_offs + get_global_id(1) * dst_stride_y + get_global_id(2) * dst_stride_z;

    VEC_TYPE source_values0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src_addr);

#if defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT)
    source_values0 = requantize(source_values0, OFFSET_IN1, OFFSET_OUT, SCALE_IN1, SCALE_OUT);
#endif /* defined(OFFSET_IN1) && defined(OFFSET_OUT) && defined(SCALE_IN1) && defined(SCALE_OUT) */

    STORE_VECTOR_SELECT(source_values, DATA_TYPE, dst_addr + offset, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
}
#endif /* defined(VEC_SIZE_LEFTOVER) */
#endif /* defined(DATA_TYPE) */
#endif /* defined(VEC_SIZE) */
