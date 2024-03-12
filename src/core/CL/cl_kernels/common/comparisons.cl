/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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

#ifdef IS_QUANTIZED
#  define DEFINE_KERNEL_STR(name) compare_##name##_quantized
#else // IS_QUANTIZED
#  define DEFINE_KERNEL_STR(name) compare_##name
#endif // IS_QUANTIZED

#define DEFINE_KERNEL(name) DEFINE_KERNEL_STR(name)

#if defined(DATA_TYPE) && defined(VEC_SIZE) && defined(OP) && defined(OP_NAME)
/** This function compares two tensors.
 *
 * @attention The inputs' data type need to be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @attention The comparison operation should be given as a preprocessor argument using -DOP=operation. e.g. -DOP=LESS
 *
 * @param[in]  in1_ptr                           Pointer to the source tensor. Supported data types: All non-quantized data types.
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
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: U8
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
    int dst_x = max((int)get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE, 0);

#if VEC_SIZE_IN1 == 1
    int in1_x = 0;
#else // VEC_SIZE_IN1 == 1
    int in1_x = dst_x;
#endif // VEC_SIZE_IN1 == 1

#if VEC_SIZE_IN2 == 1
    int in2_x = 0;
#else // VEC_SIZE_IN2 == 1
    int in2_x = dst_x;
#endif // VEC_SIZE_IN2 == 1

    int y = get_global_id(1);
    int z = get_global_id(2);

    in1_ptr += in1_offset_first_element_in_bytes + z * in1_stride_z + y * in1_stride_y + in1_x * sizeof(DATA_TYPE);
    in2_ptr += in2_offset_first_element_in_bytes + z * in2_stride_z + y * in2_stride_y + in2_x * sizeof(DATA_TYPE);
    out_ptr += out_offset_first_element_in_bytes + z * out_stride_z + y * out_stride_y + dst_x * sizeof(uchar);

    // Load values
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) in_a = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))VLOAD(VEC_SIZE_IN1)(0, (__global DATA_TYPE *)in1_ptr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) in_b = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))VLOAD(VEC_SIZE_IN2)(0, (__global DATA_TYPE *)in2_ptr);

    // Calculate and store result
#ifdef IS_QUANTIZED
    VEC_DATA_TYPE(int, VEC_SIZE) in_a_i32 = CONVERT(in_a, VEC_DATA_TYPE(int, VEC_SIZE));
    VEC_DATA_TYPE(int, VEC_SIZE) in_b_i32 = CONVERT(in_b, VEC_DATA_TYPE(int, VEC_SIZE));

    VEC_DATA_TYPE(float, VEC_SIZE) in_a_fp = CONVERT(in_a_i32 - OFFSET_IN1, VEC_DATA_TYPE(float, VEC_SIZE)) * SCALE_IN1;
    VEC_DATA_TYPE(float, VEC_SIZE) in_b_fp = CONVERT(in_b_i32 - OFFSET_IN2, VEC_DATA_TYPE(float, VEC_SIZE)) * SCALE_IN2;
#else // IS_QUANTIZED
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) in_a_fp = in_a;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) in_b_fp = in_b;
#endif // IS_QUANTIZED

#if VEC_SIZE == 1
    uchar res0 = (uchar)select(0, 255, OP(in_a_fp, in_b_fp));
#else // VEC_SIZE == 1
    VEC_DATA_TYPE(uchar, VEC_SIZE) res0 = CONVERT(OP(in_a_fp, in_b_fp), VEC_DATA_TYPE(uchar, VEC_SIZE));
#endif // VEC_SIZE == 1

    STORE_VECTOR_SELECT(res, uchar, out_ptr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
}
#endif /* defined(DATA_TYPE) && defined(VEC_SIZE) && defined(OP) && defined(OP_NAME) */
