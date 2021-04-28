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
#include "helpers.h"

#if defined(DATA_TYPE) && defined(MIN_VALUE) && defined(VECTOR_SIZE) && defined(VECTOR_SIZE_LEFTOVER)

/** Divides all the values of the input tensor by the sum calculated from softmax_layer_shift_exp_sum kernel.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE, e.g. -DDATA_TYPE=float
 * @note The zero value for the given data type must be given as a preprocessor argument using -DMIN_VALUE, e.g. -DMIN_VALUE=0
 * @note Vector size should be given as a preprocessor argument using -DVECTOR_SIZE=size. e.g. -DVECTOR_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVECTOR_SIZE_LEFTOVER. e.g. -DVECTOR_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VECTOR_SIZE
 * @note In case of log softmax, -DLOG_SOFTMAX must be passed.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: F16/F32
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
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void softmax_layer_norm(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(sum),
    TENSOR3D_DECLARATION(dst))
{
    const int x_offs = max((int)(get_global_id(0) * VECTOR_SIZE - (VECTOR_SIZE - VECTOR_SIZE_LEFTOVER) % VECTOR_SIZE), 0) * sizeof(DATA_TYPE);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x_offs + get_global_id(1) * src_stride_y + get_global_id(2) * src_stride_z;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x_offs + get_global_id(1) * dst_stride_y + get_global_id(2) * dst_stride_z;

    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(sum);

    // Load max value of 1D logits vector (row)
    DATA_TYPE sum_val = *((__global DATA_TYPE *)offset(&sum, 0, get_global_id(1)));
    VEC_DATA_TYPE(DATA_TYPE, VECTOR_SIZE)
    data0 = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)src_addr);

#if defined(LOG_SOFTMAX)
    sum_val = log(sum_val);
    data0 -= sum_val;
#else  // defined(LOG_SOFTMAX)
    data0 /= sum_val;
#endif // defined(LOG_SOFTMAX)

    STORE_VECTOR_SELECT(data, DATA_TYPE, dst_addr, VECTOR_SIZE, VECTOR_SIZE_LEFTOVER, VECTOR_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
}

#if defined(SRC_WIDTH) && defined(LOG_VECTOR_SIZE) && defined(MINVAL)

/* Number of workitems in dimension 0. */
#if !defined(GRID_SIZE)
#define GRID_SIZE 1
#endif /* !defined(GRID_SIZE) */

#define VEC_TYPE VEC_DATA_TYPE(DATA_TYPE, VECTOR_SIZE)
#define SELECT_TYPE SELECT_VEC_DATA_TYPE(DATA_TYPE, VECTOR_SIZE)

/** Identifies the maximum value across the 1st dimension and shifts the values of the input tensor by this maximum value,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE, e.g. -DDATA_TYPE=float
 * @note The zero value for the given data type must be given as a preprocessor argument using -DMIN_VALUE, e.g. -DMIN_VALUE=0
 * @note Vector size should be given as a preprocessor argument using -DVECTOR_SIZE=size. e.g. -DVECTOR_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVECTOR_SIZE_LEFTOVER. e.g. -DVECTOR_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VECTOR_SIZE
 * @note In case the input is not a multiple of VECTOR_SIZE (2,4,8,16) -DNON_MULTIPLE_OF_VECTOR_SIZE must be passed.
 * @note Beta can be optionally passed at compile time using -DBETA (by default, it is 1.0).
 * @note In case of log softmax, -DLOG_SOFTMAX must be passed.
 * @note Based on the data type, the minimum possible value must be passed using -DMINVAL. For float it should be defined as -FLT_MAX, while for half it should be -HALF_MAX
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
__kernel void softmax_layer_max_shift_exp_sum_serial(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(maxo),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(sum))
{
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + get_global_id(1) * src_stride_y + get_global_id(2) * src_stride_z;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + get_global_id(1) * dst_stride_y + get_global_id(2) * dst_stride_z;

    Image maxo = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(maxo);
    Image sum  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

#ifdef BETA
    // Initialize beta
    VEC_TYPE beta = (VEC_TYPE)BETA;
#endif /* BETA */

    // Initialize local maximum
    VEC_TYPE max_val_vec = (VEC_TYPE)(MINVAL);

#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    VEC_TYPE data    = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)src_addr);
    SELECT_TYPE widx = (SELECT_TYPE)VECTOR_SIZE_LEFTOVER > VEC_OFFS(SELECT_DATA_TYPE(DATA_TYPE), VECTOR_SIZE);
    max_val_vec      = max(max_val_vec, select((VEC_TYPE)(MINVAL), data, widx));
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */

    for(uint i = VECTOR_SIZE_LEFTOVER; i < SRC_WIDTH; i += VECTOR_SIZE)
    {
        VEC_TYPE data = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + i * sizeof(DATA_TYPE)));
        max_val_vec   = max(data, max_val_vec);
    }

    // Perform max reduction
    DATA_TYPE max_val                 = MAX_REDUCE(max_val_vec, VECTOR_SIZE);
    *((__global DATA_TYPE *)maxo.ptr) = max_val;

    /* Second section */

    // Set sum vector
    VEC_TYPE sum1D = 0;

#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    data -= max_val;
#ifdef BETA
    data *= beta;
#endif /* BETA */
#ifdef LOG_SOFTMAX
    VSTORE_PARTIAL(VECTOR_SIZE, VECTOR_SIZE_LEFTOVER)
    (data, 0, (__global DATA_TYPE *)dst_addr);
    data = exp(data);
    data = select(0, data, widx);
#else  /* LOG_SOFTMAX */
    data = exp(data);
    data = select(0, data, widx);
    VSTORE_PARTIAL(VECTOR_SIZE, VECTOR_SIZE_LEFTOVER)
    (data, 0, (__global DATA_TYPE *)dst_addr);
#endif /* LOG_SOFTMAX */
    sum1D += data;
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */

    // Shift values, exp and sum
    for(uint i = VECTOR_SIZE_LEFTOVER; i < SRC_WIDTH; i += VECTOR_SIZE)
    {
        VEC_TYPE data = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + i * sizeof(DATA_TYPE)));
        data -= max_val;
#ifdef BETA
        data *= beta;
#endif /* BETA */
#ifdef LOG_SOFTMAX
        VSTORE(VECTOR_SIZE)
        (data, 0, (__global DATA_TYPE *)(dst_addr + i * sizeof(DATA_TYPE)));
        data = exp(data);
#else  /* LOG_SOFTMAX */
        data = exp(data);
        VSTORE(VECTOR_SIZE)
        (data, 0, (__global DATA_TYPE *)(dst_addr + i * sizeof(DATA_TYPE)));
#endif /* LOG_SOFTMAX */
        sum1D += data;
    }

    // Perform sum reduction
    *((__global DATA_TYPE *)sum.ptr) = SUM_REDUCE(sum1D, VECTOR_SIZE);
}

/** Identifies the maximum value across the 1st dimension and shifts the values of the input tensor by this maximum value,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE, e.g. -DDATA_TYPE=float
 * @note The zero value for the given data type must be given as a preprocessor argument using -DMIN_VALUE, e.g. -DMIN_VALUE=0
 * @note Vector size should be given as a preprocessor argument using -DVECTOR_SIZE=size. e.g. -DVECTOR_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVECTOR_SIZE_LEFTOVER. e.g. -DVECTOR_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VECTOR_SIZE
 * @note In case the input is not a multiple of VECTOR_SIZE (2,4,8,16) -DNON_MULTIPLE_OF_VECTOR_SIZE must be passed.
 * @note Beta can be optionally passed at compile time using -DBETA (by default, it is 1.0).
 * @note In case of log softmax, -DLOG_SOFTMAX must be passed.
 * @note Based on the data type, the minimum possible value must be passed using -DMINVAL. For float it should be defined as -FLT_MAX, while for half it should be -HALF_MAX
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
__kernel void softmax_layer_max_shift_exp_sum_parallel(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(maxo),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(sum))
{
    const uint lid    = get_local_id(0);
    const uint x_offs = (VECTOR_SIZE_LEFTOVER + lid * VECTOR_SIZE) * sizeof(DATA_TYPE);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x_offs + get_global_id(1) * src_stride_y + get_global_id(2) * src_stride_z;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x_offs + get_global_id(1) * dst_stride_y + get_global_id(2) * dst_stride_z;

    Image maxo = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(maxo);
    Image sum  = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

#ifdef BETA
    // Initialize beta
    VEC_TYPE beta = (VEC_TYPE)BETA;
#endif /* BETA */

    // Define one temporary vector per work-item.
    __local VEC_TYPE tmp_local[GRID_SIZE];
    __local DATA_TYPE max_local;

    VEC_TYPE max_val_vec = (VEC_TYPE)(MINVAL);

    // Number of iterations per work-item.
    const uint width = (SRC_WIDTH / GRID_SIZE) >> LOG_VECTOR_SIZE;
    // Calculate max of row
    uint i = 0;
    for(; i < width; ++i)
    {
        VEC_TYPE data_max = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        max_val_vec       = max(data_max, max_val_vec);
    }
#ifdef NON_MULTIPLE_OF_GRID_SIZE
    // How many work-items needed to complete the computation.
    int boundary_workitems = (SRC_WIDTH % (GRID_SIZE * VECTOR_SIZE)) / VECTOR_SIZE;
    if(lid < boundary_workitems)
    {
        VEC_TYPE data_max = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        max_val_vec       = max(data_max, max_val_vec);
    }
#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    SELECT_TYPE widx;
    if(lid == 0)
    {
        // Handle non multiple of 4
        VEC_TYPE data_max = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr - VECTOR_SIZE_LEFTOVER * sizeof(DATA_TYPE)));
        widx              = (SELECT_TYPE)VECTOR_SIZE_LEFTOVER > VEC_OFFS(SELECT_DATA_TYPE(DATA_TYPE), VECTOR_SIZE);
        max_val_vec       = max(max_val_vec, select((VEC_TYPE)(MINVAL), data_max, widx));
    }
#endif /* NON_MULTIPLE_OF_VECTOR_SIZE */
#endif /* NON_MULTIPLE_OF_GRID_SIZE */
    tmp_local[lid] = max_val_vec;

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
        max_val_vec = max(tmp_local[lid + 1], tmp_local[lid]);
        max_local   = MAX_REDUCE(max_val_vec, VECTOR_SIZE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Second section */

    // Set sum vector
    VEC_TYPE  sum1D   = 0;
    DATA_TYPE max_val = max_local;

    // Shift values, exp and sum
    for(i = 0; i < width; ++i)
    {
        VEC_TYPE data = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        data -= max_val;
#ifdef BETA
        data *= beta;
#endif /* BETA */
#ifdef LOG_SOFTMAX
        VSTORE(VECTOR_SIZE)
        (data, 0, (__global DATA_TYPE *)(dst_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        data = exp(data);
#else  /* LOG_SOFTMAX */
        data = exp(data);
        VSTORE(VECTOR_SIZE)
        (data, 0, (__global DATA_TYPE *)(dst_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
#endif /* LOG_SOFTMAX */
        sum1D += data;
    }
#ifdef NON_MULTIPLE_OF_GRID_SIZE
    boundary_workitems = (SRC_WIDTH % (GRID_SIZE * VECTOR_SIZE)) / VECTOR_SIZE;
    if(lid < boundary_workitems)
    {
        VEC_TYPE data = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(__global DATA_TYPE *)(src_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        data -= max_val;
#ifdef BETA
        data *= beta;
#endif /* BETA */
#ifdef LOG_SOFTMAX
        VSTORE(VECTOR_SIZE)
        (data, 0, (__global DATA_TYPE *)(dst_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
        data = exp(data);
#else  /* LOG_SOFTMAX */
        data = exp(data);
        VSTORE(VECTOR_SIZE)
        (data, 0, (__global DATA_TYPE *)(dst_addr + (i * GRID_SIZE * VECTOR_SIZE) * sizeof(DATA_TYPE)));
#endif /* LOG_SOFTMAX */
        sum1D += data;
    }
#ifdef NON_MULTIPLE_OF_VECTOR_SIZE
    if(lid == 0)
    {
        // Handle non multiple of vector size ((GRID_SIZE * i * 4) + 4, 0); move 4 float positions ahead, *4 is due to the stride
        VEC_TYPE data = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)(src_addr - VECTOR_SIZE_LEFTOVER * sizeof(DATA_TYPE)));
        data -= max_val;
#ifdef BETA
        data *= beta;
#endif /* BETA */
#ifdef LOG_SOFTMAX
        VSTORE_PARTIAL(VECTOR_SIZE, VECTOR_SIZE_LEFTOVER)
        (data, 0, (__global DATA_TYPE *)(dst_addr - VECTOR_SIZE_LEFTOVER * sizeof(DATA_TYPE)));
        data = exp(data);
        data = select(0, data, widx);
#else  /* LOG_SOFTMAX */
        data = exp(data);
        data = select(0, data, widx);
        VSTORE_PARTIAL(VECTOR_SIZE, VECTOR_SIZE_LEFTOVER)
        (data, 0, (__global DATA_TYPE *)(dst_addr - VECTOR_SIZE_LEFTOVER * sizeof(DATA_TYPE)));
#endif /* LOG_SOFTMAX */
        sum1D += data;
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
        *((__global DATA_TYPE *)sum.ptr) = SUM_REDUCE(sum1D, VECTOR_SIZE);
    }
}

#endif // defined(SRC_WIDTH) && defined(LOG_VECTOR_SIZE) && defined(MINVAL)
#endif // defined(DATA_TYPE) && defined(MIN_VALUE) && defined(VECTOR_SIZE) && defined(VECTOR_SIZE_LEFTOVER)