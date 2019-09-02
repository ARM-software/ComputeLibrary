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

/** List of all the operations supported by this kernel.
 * @note ADD and SUB operations, when executed on integers, support saturation */
#ifdef SATURATE
#define ADD(x, y) add_sat((x), (y))
#define SUB(x, y) sub_sat((x), (y))
#else /* SATURATE */
#define ADD(x, y) (x) + (y)
#define SUB(x, y) (x) - (y)
#endif /* SATURATE */

#define MAX(x, y) max(x, y)
#define MIN(x, y) min(x, y)
#define SQUARED_DIFF(x, y) (x - y) * (x - y)
#define DIV(x, y) (x / y)
#define POWER(x, y) pow(x, y)
#define PRELU(x, y) (select(y * x, x, x > (DATA_TYPE_OUT)0))

#define OP_FUN_NAME_STR(op) elementwise_operation_##op
#define OP_FUN_NAME(op) OP_FUN_NAME_STR(op)

#if defined(OP) && defined(DATA_TYPE_IN1) && defined(DATA_TYPE_IN2) && defined(DATA_TYPE_OUT) && defined(VEC_SIZE)
/** This function executes an element-wise operation among two tensors.
 *
 * @attention The input and output data_types need to be passed at compile time using -DDATA_TYPE_IN1, -DDATA_TYPE_IN2 and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN1=uchar -DDATA_TYPE_IN2=uchar -DDATA_TYPE_OUT=short
 * @attention To perform saturating operation -DSATURATE has to be passed to the compiler otherwise wrapping policy will be used.
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @attention The element-wise operation to be executed has to be passed at compile time using -DOP (e.g., -DOP=ADD)
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
__kernel void OP_FUN_NAME(OP)(
    TENSOR3D_DECLARATION(in1),
    TENSOR3D_DECLARATION(in2),
    TENSOR3D_DECLARATION(out))
{
    // Get pixels pointer
    Tensor3D in1 = CONVERT_TO_TENSOR3D_STRUCT(in1);
    Tensor3D in2 = CONVERT_TO_TENSOR3D_STRUCT(in2);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

    // Load values
    VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)
    in_a = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN1 *)in1.ptr), VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE));
    VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)
    in_b = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN2 *)in2.ptr), VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE));

    // Calculate and store result
    VSTORE(VEC_SIZE)
    (OP(in_a, in_b), 0, (__global DATA_TYPE_OUT *)out.ptr);
}
#endif /* defined(DATA_TYPE_IN1) && defined(DATA_TYPE_IN2) && defined(DATA_TYPE_OUT) && defined(VEC_SIZE) */
