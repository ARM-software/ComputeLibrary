/*
 * Copyright (c) 2017-2021 Arm Limited.
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

#if defined(DATA_TYPE) && defined(MIN_VALUE) && defined(VECTOR_SIZE) && defined(VECTOR_SIZE_LEFTOVER) && defined(DIFF_MIN)

#define VEC_BASE VEC_DATA_TYPE(DATA_TYPE, VECTOR_SIZE)
#define VEC_INT VEC_DATA_TYPE(int, VECTOR_SIZE)
#define VEC_FLOAT VEC_DATA_TYPE(float, VECTOR_SIZE)

/** Divides all the values of the input tensor by the sum calculated from softmax_layer_shift_exp_sum kernel.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE, e.g. -DDATA_TYPE=uchar
 * @note The zero value for the given data type must be given as a preprocessor argument using -DMIN_VALUE, e.g. -DMIN_VALUE=-128
 * @note Vector size should be given as a preprocessor argument using -DVECTOR_SIZE=size. e.g. -DVECTOR_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVECTOR_SIZE_LEFTOVER. e.g. -DVECTOR_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VECTOR_SIZE
 * @note Quantized beta can be optionally passed at compile time using -DINPUT_BETA_MULTIPLIER and -DINPUT_BETA_LEFT_SHIFT (if undefined, assume beta equals 1.0)
 * @note Additional quantization data must be passed at compile time using -DSCALED_DIFF_INT_BITS and -DEXP_ACCUMULATION_INT_BITS.
 * @note -DDIFF_MIN must be passed at compile time. It is threshold difference between maximum value of input data and current processed value, it defines whether the value will be taken into account or not.
 * @note In case the input's data type is QASYMM8_SIGNED, -DQASYMM8_SIGNED must be passed.
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
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: QASYMM8/QASYMM8_SIGNED
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
    const int x_offs = max((int)(get_global_id(0) * VECTOR_SIZE - (VECTOR_SIZE - VECTOR_SIZE_LEFTOVER) % VECTOR_SIZE), 0);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x_offs * sizeof(int) + get_global_id(1) * src_stride_y + get_global_id(2) * src_stride_z;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * dst_stride_y + get_global_id(2) * dst_stride_z;

    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(sum);

#ifdef BETA
    // Initialize beta
    VEC_FLOAT beta       = (VEC_FLOAT)BETA;
    VEC_FLOAT scale_beta = -BETA * SCALE;
#else  /* BETA */
    VEC_FLOAT scale_beta = -SCALE;
#endif /* BETA */

    // Load max value of 1D logits vector (row)
    float sum_val         = *((__global float *)offset(&sum, 0, get_global_id(1)));
    float sum_val_inverse = 256.f / sum_val;

    VEC_INT   data_diff   = VLOAD(VECTOR_SIZE)(0, (__global int *)src_addr);
    VEC_FLOAT data_diff_f = CONVERT(data_diff, VEC_FLOAT);

    data_diff_f *= scale_beta;
    data_diff_f = exp(data_diff_f);
    data_diff_f *= sum_val_inverse;

#ifdef QASYMM8_SIGNED
    data_diff_f -= 128.f;
#endif /* QASYMM8_SIGNED */
    VEC_INT  data  = CONVERT(data_diff_f, VEC_INT);
    VEC_BASE data0 = CONVERT_SAT(data, VEC_DATA_TYPE(DATA_TYPE, VECTOR_SIZE));
    STORE_VECTOR_SELECT(data, DATA_TYPE, dst_addr, VECTOR_SIZE, VECTOR_SIZE_LEFTOVER, VECTOR_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}

#if defined(SRC_WIDTH) && defined(LOG_VECTOR_SIZE)

/* Number of workitems in dimension 0. */
#if !defined(GRID_SIZE)
#define GRID_SIZE 1
#endif /* !defined(GRID_SIZE) */

#define VEC_UINT VEC_DATA_TYPE(uint, VECTOR_SIZE)

VEC_INT mult_by_quantized_multiplier(VEC_INT data)
{
#if defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT)
    if(INPUT_BETA_MULTIPLIER > 1)
    {
        return ASYMM_MULT(data * (1 << INPUT_BETA_LEFT_SHIFT), INPUT_BETA_MULTIPLIER, VECTOR_SIZE);
    }
#endif /* defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT) */
    return data;
}

/** Shifts the values of the input tensor by the max calculated in softmax_layer_max kernel,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE, e.g. -DDATA_TYPE=uchar
 * @note The zero value for the given data type must be given as a preprocessor argument using -DMIN_VALUE, e.g. -DMIN_VALUE=-128
 * @note Vector size should be given as a preprocessor argument using -DVECTOR_SIZE=size. e.g. -DVECTOR_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVECTOR_SIZE_LEFTOVER. e.g. -DVECTOR_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VECTOR_SIZE
 * @note In case the input is not multiple of VECTOR_SIZE -DNON_MULTIPLE_OF_VECTOR_SIZE must be passed.
 * @note Quantized beta can be optionally passed at compile time using -DINPUT_BETA_MULTIPLIER and -DINPUT_BETA_LEFT_SHIFT (if undefined, assume beta equals 1.0)
 * @note Additional quantization data must be passed at compile time using -DSCALED_DIFF_INT_BITS and -DEXP_ACCUMULATION_INT_BITS.
 * @note -DDIFF_MIN must be passed at compile time. It is threshold difference between maximum value of input data and current processed value, it defines whether the value will be taken into account or not.
 * @note In case the input's data type is QASYMM8_SIGNED, -DQASYMM8_SIGNED must be passed.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: QASYMM8/QASYMM8_SIGNED
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
 */
__kernel void softmax_layer_max_shift_exp_sum_quantized_serial(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(maxo),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(sum))
{
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + get_global_id(1) * src_stride_y + get_global_id(2) * src_stride_z;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + get_global_id(1) * dst_stride_y + get_global_id(2) * dst_stride_z;

    Image maxo = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(maxo);
    Image sum  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

    VEC_BASE max_val_vec = (VEC_BASE)(MIN_VALUE);

#ifdef BETA
    // Initialize beta
    VEC_FLOAT beta       = (VEC_FLOAT)BETA;
    VEC_FLOAT scale_beta = -BETA * SCALE;
#else  /* BETA */
    VEC_FLOAT scale_beta = -SCALE;
#endif /* BETA */

    // Calculate max of row
#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    VEC_BASE vec_min_val = (VEC_BASE)(MIN_VALUE);
    VEC_BASE data        = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)src_addr);
    VEC_INT widx         = (VEC_INT)VECTOR_SIZE_LEFTOVER > VEC_OFFS(int, VECTOR_SIZE);
    max_val_vec          = max(max_val_vec, select(vec_min_val, data, CONVERT(widx, VEC_BASE)));
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */

    for(uint i = VECTOR_SIZE_LEFTOVER; i < SRC_WIDTH; i += VECTOR_SIZE)
    {
        VEC_BASE data = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + i * sizeof(DATA_TYPE)));
        max_val_vec   = max(data, max_val_vec);
    }

    // Perform max reduction
    DATA_TYPE max_local               = MAX_REDUCE(max_val_vec, VECTOR_SIZE);
    *((__global DATA_TYPE *)maxo.ptr) = max_local;

    // Second part

    // Load max value of 1D logits vector (row)
    int       max_val = convert_int(max_local);
    VEC_FLOAT sum1D_f = 0.f;
    // Start with the leftover items
#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    VEC_INT   data_fp   = CONVERT(data, VEC_INT);
    VEC_INT   data_diff = max_val - data_fp;
    VEC_FLOAT data_fp_f = CONVERT(data_diff, VEC_FLOAT);
    data_fp_f *= scale_beta;
    data_fp_f = exp(data_fp_f);
    data_fp_f = select(0, data_fp_f, widx);
    VSTORE_PARTIAL(VECTOR_SIZE, VECTOR_SIZE_LEFTOVER)
    (data_diff, 0, (__global int *)dst_addr);
    sum1D_f += data_fp_f;
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */
    // Do the rest and compute exp and sum
    for(uint i = VECTOR_SIZE_LEFTOVER; i < SRC_WIDTH; i += VECTOR_SIZE)
    {
        VEC_BASE data       = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + i * sizeof(DATA_TYPE)));
        VEC_INT   data_fp   = CONVERT(data, VEC_INT);
        VEC_INT   data_diff = max_val - data_fp;
        VEC_FLOAT data_fp_f = CONVERT(data_diff, VEC_FLOAT);
        data_fp_f *= scale_beta;
        data_fp_f = exp(data_fp_f);
        sum1D_f += data_fp_f;
        VSTORE(VECTOR_SIZE)
        (data_diff, 0, (__global int *)(dst_addr + i * sizeof(int)));
    }
    // Perform sum reduction
    *((__global float *)sum.ptr) = SUM_REDUCE(sum1D_f, VECTOR_SIZE);
}

/** Identifies the maximum value across the 1st dimension and shifts the values of the input tensor by this maximum value,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE, e.g. -DDATA_TYPE=uchar
 * @note The zero value for the given data type must be given as a preprocessor argument using -DMIN_VALUE, e.g. -DMIN_VALUE=-128
 * @note Vector size should be given as a preprocessor argument using -DVECTOR_SIZE=size. e.g. -DVECTOR_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVECTOR_SIZE_LEFTOVER. e.g. -DVECTOR_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VECTOR_SIZE
 * @note In case the input is not a multiple of VECTOR_SIZE (2,4,8,16) -DNON_MULTIPLE_OF_VECTOR_SIZE must be passed.
 * @note Quantized beta can be optionally passed at compile time using -DINPUT_BETA_MULTIPLIER and -DINPUT_BETA_LEFT_SHIFT (if undefined, assume beta equals 1.0)
 * @note Additional quantization data must be passed at compile time using -DSCALED_DIFF_INT_BITS and -DEXP_ACCUMULATION_INT_BITS.
 * @note -DDIFF_MIN must be passed at compile time. It is threshold difference between maximum value of input data and current processed value, it defines whether the value will be taken into account or not.
 * @note In case the input's data type is QASYMM8_SIGNED, -DQASYMM8_SIGNED must be passed.
 *
 * @param[in]  src_ptr                            Pointer to the source tensor slice. Supported data types: F16/F32
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
 */
__kernel void softmax_layer_max_shift_exp_sum_quantized_parallel(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(maxo),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(sum))
{
    const uint lid    = get_local_id(0);
    const uint x_offs = (VECTOR_SIZE_LEFTOVER + lid * VECTOR_SIZE);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * src_stride_y + get_global_id(2) * src_stride_z;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x_offs * sizeof(int) + get_global_id(1) * dst_stride_y + get_global_id(2) * dst_stride_z;

    Image maxo = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(maxo);
    Image sum  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

    // Define one temporary vector per work-item.
    __local VEC_INT tmp_local[GRID_SIZE];
    __local DATA_TYPE max_local;

    VEC_BASE vec_min_val = (VEC_BASE)(MIN_VALUE);
    VEC_BASE max_val_vec = vec_min_val;

    // Number of iterations per work-item.
    const uint width = (SRC_WIDTH / GRID_SIZE) >> LOG_VECTOR_SIZE;
    // Calculate max of row
    uint i = 0;
    for(; i < width; ++i)
    {
        VEC_BASE data_max = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        max_val_vec       = max(data_max, max_val_vec);
    }
#ifdef NON_MULTIPLE_OF_GRID_SIZE
    // How many work-items needed to complete the computation.
    int boundary_workitems = (SRC_WIDTH % (GRID_SIZE * VECTOR_SIZE)) / VECTOR_SIZE;
    if(lid < boundary_workitems)
    {
        VEC_BASE data_max = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        max_val_vec       = max(data_max, max_val_vec);
    }
#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    VEC_INT widx;
    if(lid == 0)
    {
        // Handle non multiple of 4
        VEC_BASE data_max = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr - VECTOR_SIZE_LEFTOVER * sizeof(DATA_TYPE)));
        widx              = (VEC_INT)VECTOR_SIZE_LEFTOVER > VEC_OFFS(int, VECTOR_SIZE);
        max_val_vec       = max(max_val_vec, select(vec_min_val, data_max, CONVERT(widx, VEC_BASE)));
    }
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */
#endif /* NON_MULTIPLE_OF_GRID_SIZE */
    tmp_local[lid] = CONVERT(max_val_vec, VEC_INT);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(GRID_SIZE >= 256)
    {
        if(lid < 128)
        {
            tmp_local[lid] = max(tmp_local[lid + 128], tmp_local[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 128)
    {
        if(lid < 64)
        {
            tmp_local[lid] = max(tmp_local[lid + 64], tmp_local[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 64)
    {
        if(lid < 32)
        {
            tmp_local[lid] = max(tmp_local[lid + 32], tmp_local[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 32)
    {
        if(lid < 16)
        {
            tmp_local[lid] = max(tmp_local[lid + 16], tmp_local[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 16)
    {
        if(lid < 8)
        {
            tmp_local[lid] = max(tmp_local[lid + 8], tmp_local[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 8)
    {
        if(lid < 4)
        {
            tmp_local[lid] = max(tmp_local[lid + 4], tmp_local[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 4)
    {
        if(lid < 2)
        {
            tmp_local[lid] = max(tmp_local[lid + 2], tmp_local[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
        max_val_vec = max(CONVERT((tmp_local[lid + 1]), VEC_BASE), CONVERT((tmp_local[lid]), VEC_BASE));
        max_local   = MAX_REDUCE(max_val_vec, VECTOR_SIZE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Second section */

    // Set sum vector
    VEC_INT sum1D   = 0;
    int     max_val = convert_int(max_local);

    // Shift values, exp and sum
    for(i = 0; i < width; ++i)
    {
        VEC_BASE data          = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        VEC_INT data_fp        = CONVERT(data, VEC_INT);
        VEC_INT data_diff      = data_fp - max_val;
        VEC_INT data_diff_mult = mult_by_quantized_multiplier(data_diff);
        data_fp                = ASYMM_EXP_ON_NEGATIVE_VALUES(data_diff_mult, SCALED_DIFF_INT_BITS, VECTOR_SIZE);
        data_fp                = ASYMM_RESCALE(data_fp, 0, EXP_ACCUMULATION_INT_BITS, VECTOR_SIZE);
        VSTORE(VECTOR_SIZE)
        (data_diff, 0, (__global int *)(dst_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(int)));
        sum1D = sum1D + select(0, data_fp, data_diff >= (VEC_INT)(DIFF_MIN));
    }
#ifdef NON_MULTIPLE_OF_GRID_SIZE
    boundary_workitems = (SRC_WIDTH % (GRID_SIZE * VECTOR_SIZE)) / VECTOR_SIZE;
    if(lid < boundary_workitems)
    {
        VEC_BASE data          = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        VEC_INT data_fp        = CONVERT(data, VEC_INT);
        VEC_INT data_diff      = data_fp - max_val;
        VEC_INT data_diff_mult = mult_by_quantized_multiplier(data_diff);
        data_fp                = ASYMM_EXP_ON_NEGATIVE_VALUES(data_diff_mult, SCALED_DIFF_INT_BITS, VECTOR_SIZE);
        data_fp                = ASYMM_RESCALE(data_fp, 0, EXP_ACCUMULATION_INT_BITS, VECTOR_SIZE);
        VSTORE(VECTOR_SIZE)
        (data_diff, 0, (__global int *)(dst_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(int)));
        sum1D = sum1D + select(0, data_fp, data_diff >= (VEC_INT)(DIFF_MIN));
    }
#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    if(lid == 0)
    {
        // Handle non multiple of vector size ((GRID_SIZE * i * 4) + 4, 0); move 4 float positions ahead, *4 is due to the stride
        VEC_BASE data          = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr - VECTOR_SIZE_LEFTOVER * sizeof(DATA_TYPE)));
        VEC_INT data_fp        = CONVERT(data, VEC_INT);
        VEC_INT data_diff      = data_fp - max_val;
        VEC_INT data_diff_mult = mult_by_quantized_multiplier(data_diff);
        data_fp                = ASYMM_EXP_ON_NEGATIVE_VALUES(data_diff_mult, SCALED_DIFF_INT_BITS, VECTOR_SIZE);
        data_fp                = ASYMM_RESCALE(data_fp, 0, EXP_ACCUMULATION_INT_BITS, VECTOR_SIZE);
        VSTORE_PARTIAL(VECTOR_SIZE, VECTOR_SIZE_LEFTOVER)
        (data_diff, 0, (__global int *)(dst_addr - VECTOR_SIZE_LEFTOVER * sizeof(int)));
        data_fp = select(MIN_VALUE, data_fp, data_diff >= (VEC_INT)(DIFF_MIN));
        data_fp = select(0, data_fp, widx);
        sum1D   = sum1D + data_fp;
    }
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */
#endif /* NON_MULTIPLE_OF_GRID_SIZE */
    tmp_local[lid] = sum1D;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(GRID_SIZE >= 256)
    {
        if(lid < 128)
        {
            tmp_local[lid] += tmp_local[lid + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 128)
    {
        if(lid < 64)
        {
            tmp_local[lid] += tmp_local[lid + 64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 64)
    {
        if(lid < 32)
        {
            tmp_local[lid] += tmp_local[lid + 32];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 32)
    {
        if(lid < 16)
        {
            tmp_local[lid] += tmp_local[lid + 16];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 16)
    {
        if(lid < 8)
        {
            tmp_local[lid] += tmp_local[lid + 8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 8)
    {
        if(lid < 4)
        {
            tmp_local[lid] += tmp_local[lid + 4];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(GRID_SIZE >= 4)
    {
        if(lid < 2)
        {
            tmp_local[lid] += tmp_local[lid + 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
        sum1D = (tmp_local[lid + 1] + tmp_local[lid]);
        // Perform sum reduction
        *((__global int *)sum.ptr) = SUM_REDUCE(sum1D, VECTOR_SIZE);
    }
}
#endif // #if defined(SRC_WIDTH) && defined(LOG_VECTOR_SIZE)
#endif /* defined(DATA_TYPE) && defined(DIFF_MIN) && defined(VECTOR_SIZE) && defined(VECTOR_SIZE_LEFTOVER) && defined(MIN_VALUE) */
