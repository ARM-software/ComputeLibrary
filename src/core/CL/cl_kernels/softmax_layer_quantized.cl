/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "helpers_asymm.h"

#define MAX_OP(x, y, type, size) max((x), (y))
#define ADD_OP(x, y, type, size) ((x) + (y))

/* Number of workitems in dimension 0. */
#if !defined(GRID_SIZE)
#define GRID_SIZE 1
#endif /* !defined(GRID_SIZE) */

#if VECTOR_SIZE == 2
__constant uint2 idx__ = (uint2)(0, 1);
#define asymm_mult(a, b) ASYMM_MULT(a, b, 2)
#define asymm_exp_on_negative_values(a, k_integer_bits) ASYMM_EXP_ON_NEGATIVE_VALUES(a, k_integer_bits, 2)
#define asymm_rescale(value, src_integer_bits, dst_integer_bits) ASYMM_RESCALE(value, src_integer_bits, dst_integer_bits, 2)

#elif VECTOR_SIZE == 4
__constant uint4 idx__ = (uint4)(0, 1, 2, 3);
#define asymm_mult(a, b) ASYMM_MULT(a, b, 4)
#define asymm_exp_on_negative_values(a, k_integer_bits) ASYMM_EXP_ON_NEGATIVE_VALUES(a, k_integer_bits, 4)
#define asymm_rescale(value, src_integer_bits, dst_integer_bits) ASYMM_RESCALE(value, src_integer_bits, dst_integer_bits, 4)

#elif VECTOR_SIZE == 8
__constant uint8 idx__ = (uint8)(0, 1, 2, 3, 4, 5, 6, 7);
#define asymm_mult(a, b) ASYMM_MULT(a, b, 8)
#define asymm_exp_on_negative_values(a, k_integer_bits) ASYMM_EXP_ON_NEGATIVE_VALUES(a, k_integer_bits, 8)
#define asymm_rescale(value, src_integer_bits, dst_integer_bits) ASYMM_RESCALE(value, src_integer_bits, dst_integer_bits, 8)

#else /* VECTOR_SIZE DEFAULT */
#define VECTOR_SIZE 16
#define LOG_VECTOR_SIZE 4
__constant uint16 idx__ = (uint16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#define asymm_mult(a, b) ASYMM_MULT(a, b, 16)
#define asymm_exp_on_negative_values(a, k_integer_bits) ASYMM_EXP_ON_NEGATIVE_VALUES(a, k_integer_bits, 16)
#define asymm_rescale(value, src_integer_bits, dst_integer_bits) ASYMM_RESCALE(value, src_integer_bits, dst_integer_bits, 16)

#endif /* VECTOR_SIZE END */

#define VEC_UCHAR VEC_DATA_TYPE(uchar, VECTOR_SIZE)
#define VEC_UINT VEC_DATA_TYPE(uint, VECTOR_SIZE)
#define VEC_INT VEC_DATA_TYPE(int, VECTOR_SIZE)

#if defined(DIFF_MIN)

VEC_INT mult_by_quantized_multiplier_serial(VEC_INT data)
{
#if defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT)
    if(INPUT_BETA_MULTIPLIER > 1)
    {
        return asymm_mult(data * (1 << INPUT_BETA_LEFT_SHIFT), INPUT_BETA_MULTIPLIER);
    }
#endif /* defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT) */
    return data;
}

int4 mult_by_quantized_multiplier_parallel(int4 data)
{
#if defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT)
    if(INPUT_BETA_MULTIPLIER > 1)
    {
        return ASYMM_MULT(data * (1 << INPUT_BETA_LEFT_SHIFT), INPUT_BETA_MULTIPLIER, 4);
    }
#endif /* defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT) */
    return data;
}

/** Shifts the values of the input tensor by the max calculated in softmax_layer_max kernel,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note In case the input is not multiple of 16 -DNON_MULTIPLE_OF_VECTOR_SIZE must be passed.
 * @note Quantized beta can be optionally passed at compile time using -DINPUT_BETA_MULTIPLIER and -DINPUT_BETA_LEFT_SHIFT (if undefined, assume beta equals 1.0)
 * @note -DDIFF_MIN must be passed at compile time. It is threshold difference between maximum value of input data and current processed value, it defines whether the value will be taken into account or not.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: QASYMM8
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  max_ptr                           Pointer to the max values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  max_stride_x                      Stride of the max values tensor in X dimension (in bytes)
 * @param[in]  max_step_x                        max_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  max_stride_y                      Stride of the max values tensor in Y dimension (in bytes)
 * @param[in]  max_step_y                        max_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  max_stride_z                      Stride of the max values tensor in Z dimension (in bytes)
 * @param[in]  max_step_z                        max_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  max_offset_first_element_in_bytes The offset of the first element in the max values tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: S32
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[out] sum_ptr                           Pointer to the sum values tensor slice. Supported data types: same as @p dst_ptr
 * @param[in]  sum_stride_x                      Stride of the sum values tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the sum values tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_stride_z                      Stride of the sum values tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the sum values tensor
 * @param[in]  width                             Input image width
 */
__kernel void softmax_layer_max_shift_exp_sum_quantized_serial(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(maxo),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(sum),
    uint width)
{
    Image src  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image maxo = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(maxo);
    Image sum  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

    VEC_UCHAR max_val_vec = 0;

    // Calculate max of row
    const uint width4 = width >> LOG_VECTOR_SIZE;
    for(uint i = 0; i < width4; i++)
    {
        VEC_UCHAR data = VLOAD(VECTOR_SIZE)(0, (__global uchar *)offset(&src, i << LOG_VECTOR_SIZE, 0));
        max_val_vec    = MAX_OP(data, max_val_vec, uchar, 16);
    }

#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    // Handle non multiple of 16
    VEC_UCHAR uchar_min = (VEC_UCHAR)0;
    VEC_UCHAR data      = VLOAD(VECTOR_SIZE)(0, (__global uchar *)offset(&src, width4 << LOG_VECTOR_SIZE, 0));
    VEC_UCHAR widx      = CONVERT(((VEC_UINT)(width4 << LOG_VECTOR_SIZE) + idx__) < width, VEC_UCHAR);
    max_val_vec         = MAX_OP(max_val_vec, select(uchar_min, data, widx), uchar, 16);
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */

    // Perform max reduction
#if VECTOR_SIZE == 16
    max_val_vec.s01234567 = MAX_OP(max_val_vec.s01234567, max_val_vec.s89ABCDEF, uchar, 8);
#endif /* VECTOR SIZE 16 END */
#if VECTOR_SIZE >= 8
    max_val_vec.s0123 = MAX_OP(max_val_vec.s0123, max_val_vec.s4567, uchar, 4);
#endif /* VECTOR SIZE 8 END */
#if VECTOR_SIZE >= 4
    max_val_vec.s01 = MAX_OP(max_val_vec.s01, max_val_vec.s23, uchar, 2);
#endif /* VECTOR SIZE 4 END */
    max_val_vec.s0 = MAX_OP(max_val_vec.s0, max_val_vec.s1, uchar, 1);

    // Store result
    *((__global uchar *)maxo.ptr) = max_val_vec.s0;

    // Second part

    // Load max value of 1D logits vector (row)
    int max_val = convert_int(*((__global uchar *)offset(&maxo, 0, 0)));

    // Set sum vector, Q(EXP_ACCUMULATION_INT_BITS)
    VEC_INT sum1D = 0;

    // Shift values, exp and sum
    for(uint i = 0; i < width4; i++)
    {
        VEC_UCHAR data         = VLOAD(VECTOR_SIZE)(0, (__global uchar *)offset(&src, i << LOG_VECTOR_SIZE, 0));
        VEC_INT data_fp        = CONVERT(data, VEC_INT);
        VEC_INT data_diff      = data_fp - max_val;
        VEC_INT data_diff_mult = mult_by_quantized_multiplier_serial(data_diff);
        data_fp                = asymm_exp_on_negative_values(data_diff_mult, SCALED_DIFF_INT_BITS);
        data_fp                = asymm_rescale(data_fp, 0, EXP_ACCUMULATION_INT_BITS);
        VSTORE(VECTOR_SIZE)
        (data_diff, 0, (__global int *)offset(&dst, i << LOG_VECTOR_SIZE, 0));
        sum1D = sum1D + select(0, data_fp, data_diff >= (VEC_INT)(DIFF_MIN));
    }

#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    // Handle non multiple of 16
    data                   = VLOAD(VECTOR_SIZE)(0, (__global uchar *)offset(&src, width4 << LOG_VECTOR_SIZE, 0));
    VEC_INT data_fp        = CONVERT(data, VEC_INT);
    VEC_INT data_diff      = data_fp - max_val;
    VEC_INT data_diff_mult = mult_by_quantized_multiplier_serial(data_diff);
    data_fp                = asymm_exp_on_negative_values(data_diff_mult, SCALED_DIFF_INT_BITS);
    data_fp                = asymm_rescale(data_fp, 0, EXP_ACCUMULATION_INT_BITS);
    VEC_INT widx_          = CONVERT(((VEC_UINT)(width4 << LOG_VECTOR_SIZE) + idx__) < width, VEC_INT);
    VSTORE(VECTOR_SIZE)
    (data_diff, 0, (__global int *)offset(&dst, width4 << LOG_VECTOR_SIZE, 0));
    data_fp = select(0, data_fp, data_diff >= (VEC_INT)(DIFF_MIN));
    sum1D   = sum1D + select(0, data_fp, widx_);
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */

    // Perform sum reduction
#if VECTOR_SIZE == 16
    sum1D.s01234567 = ADD_OP(sum1D.s01234567, sum1D.s89ABCDEF, uchar, 8);
#endif /* VECTOR SIZE 16 END */
#if VECTOR_SIZE >= 8
    sum1D.s0123 = ADD_OP(sum1D.s0123, sum1D.s4567, uchar, 4);
#endif /* VECTOR SIZE 8 END */
#if VECTOR_SIZE >= 4
    sum1D.s01 = ADD_OP(sum1D.s01, sum1D.s23, uchar, 2);
#endif /* VECTOR SIZE 4 END */
    sum1D.s0 = ADD_OP(sum1D.s0, sum1D.s1, uchar, 1);

    // Calculate and store result
    *((__global int *)sum.ptr) = sum1D.s0;
}

/** Identifies the maximum value across the 1st dimension and shifts the values of the input tensor by this maximum value,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Fixed point position must be given as a preprocessor argument using -DFIXED_POINT_POSITION=pos. e.g. DFIXED_POINT_POSITION=4
 * @note In case the input is not a multiple of VECTOR_SIZE (2,4,8,16) -DNON_MULTIPLE_OF_VECTOR_SIZE must be passed.
 *
 * @param[in]  src_ptr                            Pointer to the source tensor slice. Supported data types: QS8/QS16/F16/F32
 * @param[in]  src_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                         src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in]  maxo_ptr                           Pointer to the max values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  maxo_stride_x                      Stride of the max values tensor in X dimension (in bytes)
 * @param[in]  maxo_step_x                        max_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  maxo_stride_y                      Stride of the max values tensor in Y dimension (in bytes)
 * @param[in]  maxo_step_y                        max_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  maxo_stride_z                      Stride of the max values tensor in Z dimension (in bytes)
 * @param[in]  maxo_step_z                        max_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  maxo_offset_first_element_in_bytes The offset of the first element in the max values tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                         dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[out] sum_ptr                            Pointer to the sum values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  sum_stride_x                       Stride of the sum values tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                         sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                       Stride of the sum values tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                         sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_stride_z                       Stride of the sum values tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                         sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes  The offset of the first element in the sum values tensor
 * @param[in]  width                              Input image width
 */
__kernel void softmax_layer_max_shift_exp_sum_quantized_parallel(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(maxo),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(sum),
    uint width)
{
    Image src  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image maxo = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(maxo);
    Image sum  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

    const uint4 idx4 = (uint4)(0, 1, 2, 3);
    const uint  lid  = get_local_id(0);

    // Define one temporary vector per work-item.
    __local int4 tmp_local[GRID_SIZE];
    __local uchar max_local;

    uchar4 uchar_min   = (uchar4)0;
    uchar4 max_val_vec = uchar_min;

    // Number of elements per work-item.
    const uint row = width / GRID_SIZE;
    // Number of iterations per work-item.
    const uint width_ = row >> 2;
    // Calculate max of row
    uint i = 0;
    for(; i < width_; i++)
    {
        uchar4 data_max = vload4(0, (__global uchar *)offset(&src, i * GRID_SIZE * 4, 0));
        max_val_vec     = MAX_OP(data_max, max_val_vec, uchar, 4);
    }
#ifdef NON_MULTIPLE_OF_GRID_SIZE
    // How many work-items needed to complete the computation.
    int boundary_workitems = (width % (GRID_SIZE * 4)) / 4;
    if(lid < boundary_workitems)
    {
        uchar4 data_max = vload4(0, (__global uchar *)offset(&src, i * GRID_SIZE * 4, 0));
        max_val_vec     = MAX_OP(data_max, max_val_vec, uchar, 4);
    }
#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    if(boundary_workitems == 0)
    {
        boundary_workitems = GRID_SIZE;
        i--;
    }
    if(lid == (boundary_workitems - 1))
    {
        // Handle non multiple of 4
        uchar4 data_max = vload4(0, (__global uchar *)offset(&src, (GRID_SIZE * i * 4) + 4, 0));
        uchar4 widx     = convert_uchar4(((uint4)(GRID_SIZE * i * 4) + boundary_workitems * 4 + idx4) < width);
        max_val_vec     = MAX_OP(max_val_vec, select(uchar_min, data_max, widx), uchar, 4);
    }
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */
#endif /* NON_MULTIPLE_OF_GRID_SIZE */
    tmp_local[lid] = convert_int4(max_val_vec);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(GRID_SIZE >= 256)
    {
        if(lid < 128)
        {
            tmp_local[lid] = MAX_OP(tmp_local[lid + 128], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 128)
    {
        if(lid < 64)
        {
            tmp_local[lid] = MAX_OP(tmp_local[lid + 64], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 64)
    {
        if(lid < 32)
        {
            tmp_local[lid] = MAX_OP(tmp_local[lid + 32], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 32)
    {
        if(lid < 16)
        {
            tmp_local[lid] = MAX_OP(tmp_local[lid + 16], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 16)
    {
        if(lid < 8)
        {
            tmp_local[lid] = MAX_OP(tmp_local[lid + 8], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 8)
    {
        if(lid < 4)
        {
            tmp_local[lid] = MAX_OP(tmp_local[lid + 4], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 4)
    {
        if(lid < 2)
        {
            tmp_local[lid] = MAX_OP(tmp_local[lid + 2], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
        max_val_vec     = MAX_OP(convert_uchar4(tmp_local[lid + 1]), convert_uchar4(tmp_local[lid]), uchar, 4);
        max_val_vec.s01 = MAX_OP(max_val_vec.s01, max_val_vec.s23, uchar, 2);
        max_val_vec.s0  = MAX_OP(max_val_vec.s0, max_val_vec.s1, uchar, 1);
        max_local       = max_val_vec.s0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Second section */

    // Set sum vector
    int4 sum1D   = 0;
    int  max_val = convert_int(max_local);

    // Shift values, exp and sum
    for(i = 0; i < width_; i++)
    {
        uchar4 data         = vload4(0, (__global uchar *)offset(&src, i * GRID_SIZE * 4, 0));
        int4 data_fp        = convert_int4(data);
        int4 data_diff      = data_fp - max_val;
        int4 data_diff_mult = mult_by_quantized_multiplier_parallel(data_diff);
        data_fp             = ASYMM_EXP_ON_NEGATIVE_VALUES(data_diff_mult, SCALED_DIFF_INT_BITS, 4);
        data_fp             = ASYMM_RESCALE(data_fp, 0, EXP_ACCUMULATION_INT_BITS, 4);
        vstore4(data_diff, 0, (__global int *)offset(&dst, i * GRID_SIZE * 4, 0));
        sum1D = sum1D + select(0, data_fp, data_diff >= (int4)(DIFF_MIN));
    }
#ifdef NON_MULTIPLE_OF_GRID_SIZE
    boundary_workitems = (width % (GRID_SIZE * 4)) / 4;
    if(lid < boundary_workitems)
    {
        uchar4 data         = vload4(0, (__global uchar *)offset(&src, i * GRID_SIZE * 4, 0));
        int4 data_fp        = convert_int4(data);
        int4 data_diff      = data_fp - max_val;
        int4 data_diff_mult = mult_by_quantized_multiplier_parallel(data_diff);
        data_fp             = ASYMM_EXP_ON_NEGATIVE_VALUES(data_diff_mult, SCALED_DIFF_INT_BITS, 4);
        data_fp             = ASYMM_RESCALE(data_fp, 0, EXP_ACCUMULATION_INT_BITS, 4);
        vstore4(data_diff, 0, (__global int *)offset(&dst, i * GRID_SIZE * 4, 0));
        sum1D = sum1D + select(0, data_fp, data_diff >= (int4)(DIFF_MIN));
    }
#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    if(boundary_workitems == 0)
    {
        boundary_workitems = GRID_SIZE;
        i--;
    }
    if(lid == (boundary_workitems - 1))
    {
        // Handle non multiple of vector size ((GRID_SIZE * i * 4) + 4, 0); move 4 float positions ahead, *4 is due to the stride
        uchar4 data         = vload4(0, (__global uchar *)offset(&src, i * GRID_SIZE * 4 + 4, 0));
        int4 data_fp        = convert_int4(data);
        int4 data_diff      = data_fp - max_val;
        int4 data_diff_mult = mult_by_quantized_multiplier_parallel(data_diff);
        data_fp             = ASYMM_EXP_ON_NEGATIVE_VALUES(data_diff_mult, SCALED_DIFF_INT_BITS, 4);
        data_fp             = ASYMM_RESCALE(data_fp, 0, EXP_ACCUMULATION_INT_BITS, 4);
        int4 widx           = convert_int4(((uint4)(GRID_SIZE * i * 4) + boundary_workitems * 4 + idx4) < width);
        data_fp             = select(0, data_fp, widx);
        vstore4(data_diff, 0, (__global int *)offset(&dst, i * GRID_SIZE * 4 + 4, 0));
        sum1D = sum1D + select(0, data_fp, data_diff >= (int4)(DIFF_MIN));
    }
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */
#endif /* NON_MULTIPLE_OF_GRID_SIZE */
    tmp_local[lid] = sum1D;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(GRID_SIZE >= 256)
    {
        if(lid < 128)
        {
            tmp_local[lid] = ADD_OP(tmp_local[lid + 128], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 128)
    {
        if(lid < 64)
        {
            tmp_local[lid] = ADD_OP(tmp_local[lid + 64], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 64)
    {
        if(lid < 32)
        {
            tmp_local[lid] = ADD_OP(tmp_local[lid + 32], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 32)
    {
        if(lid < 16)
        {
            tmp_local[lid] = ADD_OP(tmp_local[lid + 16], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 16)
    {
        if(lid < 8)
        {
            tmp_local[lid] = ADD_OP(tmp_local[lid + 8], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 8)
    {
        if(lid < 4)
        {
            tmp_local[lid] = ADD_OP(tmp_local[lid + 4], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 4)
    {
        if(lid < 2)
        {
            tmp_local[lid] = ADD_OP(tmp_local[lid + 2], tmp_local[lid], int, 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
        sum1D = ADD_OP(tmp_local[lid + 1], tmp_local[lid], int, 4);
        // Perform max reduction
        sum1D.s01                  = ADD_OP(sum1D.s01, sum1D.s23, int, 2);
        sum1D.s0                   = ADD_OP(sum1D.s0, sum1D.s1, int, 1);
        *((__global int *)sum.ptr) = sum1D.s0;
    }
}

/** Divides all the values of the input tensor by the sum calculated from softmax_layer_shift_exp_sum kernel.
 *
 * @note Fixed point position must be given as a preprocessor argument using -DFIXED_POINT_POSITION=pos. e.g. DFIXED_POINT_POSITION=4
 * @note Quantized beta can be optionally passed at compile time using -DINPUT_BETA_MULTIPLIER and -DINPUT_BETA_LEFT_SHIFT (if undefined, assume beta equals 1.0)
 * @note -DDIFF_MIN must be passed at compile time. It is threshold difference between maximum value of input data and current processed value, it defines whether the value will be taken into account or not.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: S32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  sum_ptr                           Pointer to the sum values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  sum_stride_x                      Stride of the sum values tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the sum values tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_stride_z                      Stride of the sum values tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the sum values tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: QASYMM8
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void softmax_layer_norm_quantized(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(sum),
    TENSOR3D_DECLARATION(dst))
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(sum);

    // Load max value of 1D logits vector (row)
    int sum_val = *((__global int *)offset(&sum, 0, get_global_id(1)));

    // It will be better to calculate this in prev layer and pass here as parameter
    uint  sum_val_u               = convert_uint(sum_val);
    int   headroom_plus_one       = clz(sum_val_u);
    int   num_bits_over_unit      = EXP_ACCUMULATION_INT_BITS - headroom_plus_one;
    int   shifted_sum_minus_one_1 = convert_int((sum_val_u << headroom_plus_one) - (1u << 31));
    int16 shifted_sum_minus_one   = shifted_sum_minus_one_1;
    int16 shifted_scale           = ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1(shifted_sum_minus_one, 16);

    // It was already calculated in prev layer, should be stored into tmp output and reused
    int16 data_diff      = vload16(0, (__global int *)offset(&src, 0, 0));
    int16 data_diff_mult = data_diff;
#if defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT)
    if(INPUT_BETA_MULTIPLIER > 1)
    {
        data_diff_mult = ASYMM_MULT(data_diff * (1 << INPUT_BETA_LEFT_SHIFT), INPUT_BETA_MULTIPLIER, 16);
    }
#endif /* defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT) */
    int16 data = ASYMM_EXP_ON_NEGATIVE_VALUES(data_diff_mult, SCALED_DIFF_INT_BITS, 16);

    data = ASYMM_MULT(shifted_scale, data, 16);
    data = ASYMM_ROUNDING_DIVIDE_BY_POW2(data, num_bits_over_unit + 31 - 8, 16);
    data = select(0, data, data_diff >= (int16)(DIFF_MIN));
    vstore16(convert_uchar16_sat(data), 0, (__global uchar *)offset(&dst, 0, 0));
}

#endif /* defined(DIFF_MIN) */
