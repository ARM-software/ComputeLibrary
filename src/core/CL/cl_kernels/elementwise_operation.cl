/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#define PRELU(x, y) (select(y * x, x, CONVERT((x > (DATA_TYPE_OUT)0), SELECT_VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT))))

#if defined(VEC_SIZE_OUT) && defined(DATA_TYPE_OUT)
#define AND(x, y) (CONVERT((x && y), VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT)) & 1)
#define OR(x, y) (CONVERT((x || y), VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT)) & 1)
#endif // defined(VEC_SIZE_OUT) && defined(DATA_TYPE_OUT)

#define OP_FUN_NAME_STR(op) elementwise_operation_##op
#define OP_FUN_NAME(op) OP_FUN_NAME_STR(op)

#if defined(OP) && defined(VEC_SIZE_IN1) && defined(VEC_SIZE_IN2) && defined(VEC_SIZE_OUT) && defined(DATA_TYPE_IN1) && defined(DATA_TYPE_IN2) && defined(DATA_TYPE_OUT)

#if defined(ACTIVATION_TYPE)
#include "activation_float_helpers.h"
#endif // defined(ACTIVATION_TYPE)

/** This function executes an element-wise operation among two tensors.
 *
 * @note Vector sizes of inputs and output have to be passed at compile time using -DVEC_SIZE_IN1, -DVEC_SIZE_IN2, -DVEC_SIZE_OUT.
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_OUT=3. It is defined as the remainder between the input's first dimension and VEC_SIZE_OUT
 * @note The input and output data_types need to be passed at compile time using -DDATA_TYPE_IN1, -DDATA_TYPE_IN2 and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN1=uchar -DDATA_TYPE_IN2=uchar -DDATA_TYPE_OUT=short
 * @note To perform saturating operation -DSATURATE has to be passed to the compiler otherwise wrapping policy will be used.
 * @note The element-wise operation to be executed has to be passed at compile time using -DOP (e.g., -DOP=ADD)
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
#if VEC_SIZE_IN1 == 1
    uint in1_x_offs = 0;
#else  // VEC_SIZE_IN1 == 1
    uint in1_x_offs = max((int)(get_global_id(0) * VEC_SIZE_IN1 - (VEC_SIZE_IN1 - VEC_SIZE_LEFTOVER) % VEC_SIZE_IN1), 0);
#endif // VEC_SIZE_IN1 == 1
#if VEC_SIZE_IN2 == 1
    uint in2_x_offs = 0;
#else  // VEC_SIZE_IN2 == 1
    uint in2_x_offs = max((int)(get_global_id(0) * VEC_SIZE_IN2 - (VEC_SIZE_IN2 - VEC_SIZE_LEFTOVER) % VEC_SIZE_IN2), 0);
#endif // VEC_SIZE_IN2 == 1
    uint out_x_offs = max((int)(get_global_id(0) * VEC_SIZE_OUT - (VEC_SIZE_OUT - VEC_SIZE_LEFTOVER) % VEC_SIZE_OUT), 0);

    // Get pixels pointer
    __global uchar *in1_addr = in1_ptr + in1_offset_first_element_in_bytes + in1_x_offs * sizeof(DATA_TYPE_IN1) + get_global_id(1) * in1_step_y + get_global_id(2) * in1_step_z;
    __global uchar *in2_addr = in2_ptr + in2_offset_first_element_in_bytes + in2_x_offs * sizeof(DATA_TYPE_IN2) + get_global_id(1) * in2_step_y + get_global_id(2) * in2_step_z;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + out_x_offs * sizeof(DATA_TYPE_OUT) + get_global_id(1) * out_step_y + get_global_id(2) * out_step_z;

    // Load values
    VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT)
    in_a = CONVERT((VEC_DATA_TYPE(DATA_TYPE_IN1, VEC_SIZE_OUT))(VLOAD(VEC_SIZE_IN1)(0, (__global DATA_TYPE_IN1 *)in1_addr)), VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT));
    VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT)
    in_b = CONVERT((VEC_DATA_TYPE(DATA_TYPE_IN2, VEC_SIZE_OUT))(VLOAD(VEC_SIZE_IN2)(0, (__global DATA_TYPE_IN2 *)in2_addr)), VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT));

    // Calculate and store result
    VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT)
    res0 = OP(in_a, in_b);
#if defined(ACTIVATION_TYPE)
    res0 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE_OUT, VEC_SIZE_OUT, res0, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    STORE_VECTOR_SELECT(res, DATA_TYPE_OUT, out_addr, VEC_SIZE_OUT, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
}
#endif /* defined(OP) && defined(VEC_SIZE_IN1) && defined(VEC_SIZE_IN2) && defined(VEC_SIZE_OUT) && defined(DATA_TYPE_IN1) && defined(DATA_TYPE_IN2) && defined(DATA_TYPE_OUT) */
