/*
 * Copyright (c) 2016-2020 Arm Limited.
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

#if defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER)

/** This function computes the bitwise OR of two input images.
 *
 * @note The following variables must be passed at compile time:
 * -# -DVEC_SIZE         : The number of elements processed in X dimension
 * -# -DVEC_SIZE_LEFTOVER: Leftover size in the X dimension; x_dimension % VEC_SIZE
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void bitwise_or(
    IMAGE_DECLARATION(in1),
    IMAGE_DECLARATION(in2),
    IMAGE_DECLARATION(out))
{
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);

    // Get pixels pointer
    __global uchar *in1_addr = in1_ptr + in1_offset_first_element_in_bytes + x_offs + get_global_id(1) * in1_step_y;
    __global uchar *in2_addr = in2_ptr + in2_offset_first_element_in_bytes + x_offs + get_global_id(1) * in2_step_y;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + x_offs + get_global_id(1) * out_step_y;

    // Load data
    VEC_DATA_TYPE(uchar, VEC_SIZE)
    in_a = VLOAD(VEC_SIZE)(0, (__global uchar *)in1_addr);
    VEC_DATA_TYPE(uchar, VEC_SIZE)
    in_b = VLOAD(VEC_SIZE)(0, (__global uchar *)in2_addr);

    VEC_DATA_TYPE(uchar, VEC_SIZE)
    data0 = in_a | in_b;

    // Boundary-aware store
    STORE_VECTOR_SELECT(data, uchar, (__global uchar *)out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}

/** This function computes the bitwise AND of two input images.
 *
 * @note The following variables must be passed at compile time:
 * -# -DVEC_SIZE         : The number of elements processed in X dimension
 * -# -DVEC_SIZE_LEFTOVER: Leftover size in the X dimension; x_dimension % VEC_SIZE
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void bitwise_and(
    IMAGE_DECLARATION(in1),
    IMAGE_DECLARATION(in2),
    IMAGE_DECLARATION(out))
{
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);

    // Get pixels pointer
    __global uchar *in1_addr = in1_ptr + in1_offset_first_element_in_bytes + x_offs + get_global_id(1) * in1_step_y;
    __global uchar *in2_addr = in2_ptr + in2_offset_first_element_in_bytes + x_offs + get_global_id(1) * in2_step_y;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + x_offs + get_global_id(1) * out_step_y;

    // Load data
    VEC_DATA_TYPE(uchar, VEC_SIZE)
    in_a = VLOAD(VEC_SIZE)(0, (__global uchar *)in1_addr);
    VEC_DATA_TYPE(uchar, VEC_SIZE)
    in_b = VLOAD(VEC_SIZE)(0, (__global uchar *)in2_addr);

    VEC_DATA_TYPE(uchar, VEC_SIZE)
    data0 = in_a & in_b;

    // Boundary-aware store
    STORE_VECTOR_SELECT(data, uchar, (__global uchar *)out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}

/** This function computes the bitwise XOR of two input images.
 *
 * @note The following variables must be passed at compile time:
 * -# -DVEC_SIZE         : The number of elements processed in X dimension
 * -# -DVEC_SIZE_LEFTOVER: Leftover size in the X dimension; x_dimension % VEC_SIZE
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void bitwise_xor(
    IMAGE_DECLARATION(in1),
    IMAGE_DECLARATION(in2),
    IMAGE_DECLARATION(out))
{
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);

    // Get pixels pointer
    __global uchar *in1_addr = in1_ptr + in1_offset_first_element_in_bytes + x_offs + get_global_id(1) * in1_step_y;
    __global uchar *in2_addr = in2_ptr + in2_offset_first_element_in_bytes + x_offs + get_global_id(1) * in2_step_y;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + x_offs + get_global_id(1) * out_step_y;

    // Load data
    VEC_DATA_TYPE(uchar, VEC_SIZE)
    in_a = VLOAD(VEC_SIZE)(0, (__global uchar *)in1_addr);
    VEC_DATA_TYPE(uchar, VEC_SIZE)
    in_b = VLOAD(VEC_SIZE)(0, (__global uchar *)in2_addr);

    VEC_DATA_TYPE(uchar, VEC_SIZE)
    data0 = in_a ^ in_b;

    // Boundary-aware store
    STORE_VECTOR_SELECT(data, uchar, (__global uchar *)out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}

/** This function computes the bitwise NOT of an images.
 *
 * @note The following variables must be passed at compile time:
 * -# -DVEC_SIZE         : The number of elements processed in X dimension
 * -# -DVEC_SIZE_LEFTOVER: Leftover size in the X dimension; x_dimension % VEC_SIZE
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         in_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void bitwise_not(
    IMAGE_DECLARATION(in1),
    IMAGE_DECLARATION(out))
{
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);

    // Get pixels pointer
    __global uchar *in1_addr = in1_ptr + in1_offset_first_element_in_bytes + x_offs + get_global_id(1) * in1_step_y;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + x_offs + get_global_id(1) * out_step_y;

    // Load data
    VEC_DATA_TYPE(uchar, VEC_SIZE)
    in_a = VLOAD(VEC_SIZE)(0, (__global uchar *)in1_addr);

    VEC_DATA_TYPE(uchar, VEC_SIZE)
    data0 = ~in_a;

    // Boundary-aware store
    STORE_VECTOR_SELECT(data, uchar, (__global uchar *)out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}

#endif // defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER)