/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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
#include "tile_helpers.h"

#if defined(VEC_SIZE) && defined(DATA_TYPE) && defined(DATA_TYPE_OUTPUT)

#define VEC_TYPE_IN VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#define VEC_TYPE_OUT VEC_DATA_TYPE(DATA_TYPE_OUTPUT, VEC_SIZE)
#define VEC_SELECT_IN SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#define VEC_SIGNED_INT_IN SIGNED_INT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)

#if defined(FLOAT_DATA_TYPE)
#define ISGREATER(x, y) (VEC_SELECT_IN) isgreater(x, y)
#define ISLESS(x, y) (VEC_SELECT_IN) isless(x, y)
#else // !FLOAT_DATA_TYPE
#if defined(WIDTH)
#define ISGREATER(x, y) (x > y) ? 1 : 0
#define ISLESS(x, y) (x < y) ? 1 : 0
#else // !defined(WIDTH)
#define ISGREATER(x, y) select((VEC_SIGNED_INT_IN)0, (VEC_SIGNED_INT_IN)-1, (VEC_SIGNED_INT_IN)(x > y))
#define ISLESS(x, y) select((VEC_SIGNED_INT_IN)0, (VEC_SIGNED_INT_IN)-1, (VEC_SIGNED_INT_IN)(x < y))
#endif // defined(WIDTH)
#endif // defined(FLOAT_DATA_TYPE)

#if defined(ARG_MAX)
#define CONDITION_TO_USE(x, y) ISGREATER(x, y)
#elif defined(ARG_MIN)
#define CONDITION_TO_USE(x, y) ISLESS(x, y)
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#error "Unsupported reduction operation!"
#endif // defined(ARG_MAX)

#if defined(WIDTH)

#if defined(ARG_MAX)
#define VECTOR_PREDICATE_EQ(x, y) ((x) >= (y))
#define VECTOR_PREDICATE(x, y) ((x) > (y))
#define SCALAR_SELECT_OP(x, y) ((x) > (y)) ? (x) : (y);
#elif defined(ARG_MIN)
#define VECTOR_PREDICATE_EQ(x, y) ((x) <= (y))
#define VECTOR_PREDICATE(x, y) ((x) < (y))
#define SCALAR_SELECT_OP(x, y) ((x) < (y)) ? (x) : (y);
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#error "Unsupported reduction operation!"
#endif // defined(ARG_MAX)

inline DATA_TYPE_OUTPUT vectorized_compute_arg_min_max_2(DATA_TYPE *min_max_val, DATA_TYPE_OUTPUT *min_max_idx, VEC_DATA_TYPE(DATA_TYPE, 2) in, VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 2) res)
{
    if( VECTOR_PREDICATE_EQ(in.s0,in.s1) )
    {
        *min_max_val  = in.s0;
        *min_max_idx  = res.s0;
    }
    else
    {
        *min_max_val  = in.s1;
        *min_max_idx  = res.s1;
    }
}

inline DATA_TYPE_OUTPUT vectorized_compute_arg_min_max_4(DATA_TYPE *min_max_val, DATA_TYPE_OUTPUT *min_max_idx, VEC_DATA_TYPE(DATA_TYPE, 4) in, VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 4) res)
{
    VEC_DATA_TYPE(COND_DATA_TYPE, 2)
    idx_sel       = VECTOR_PREDICATE_EQ(in.s01, in.s23);
    in.s01      = select(in.s23, in.s01, idx_sel);
    res.s01     = select(res.s23, res.s01, CONVERT(idx_sel, int2));
    idx_sel.s0    = VECTOR_PREDICATE(in.s0, in.s1) || (in.s0 == in.s1 && CONVERT((res.s0 < res.s1), COND_DATA_TYPE));
    res.s0        = select(res.s1, res.s0, CONVERT(idx_sel.s0, int));
    *min_max_val  = SCALAR_SELECT_OP(in.s0, in.s1);
    *min_max_idx  = res.s0;
}

inline DATA_TYPE_OUTPUT vectorized_compute_arg_min_max_8(DATA_TYPE *min_max_val, DATA_TYPE_OUTPUT *min_max_idx, VEC_DATA_TYPE(DATA_TYPE, 8) in, VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 8) res)
{
    VEC_DATA_TYPE(COND_DATA_TYPE, 4)
    idx_sel       = VECTOR_PREDICATE_EQ(in.s0123, in.s4567);
    in.s0123      = select(in.s4567, in.s0123, idx_sel);
    res.s0123     = select(res.s4567, res.s0123, CONVERT(idx_sel, int4));
    idx_sel.s01   = (VECTOR_PREDICATE(in.s01, in.s23)) || (in.s01 == in.s23 && CONVERT(((res.s01 < res.s23)), VEC_DATA_TYPE(COND_DATA_TYPE, 2)));
    in.s01        = select(in.s23, in.s01, idx_sel.s01);
    res.s01       = select(res.s23, res.s01, CONVERT(idx_sel.s01, int2));
    idx_sel.s0    = VECTOR_PREDICATE(in.s0, in.s1) || (in.s0 == in.s1 && CONVERT((res.s0 < res.s1), COND_DATA_TYPE));
    res.s0        = select(res.s1, res.s0, CONVERT(idx_sel.s0, int));
    *min_max_val  = SCALAR_SELECT_OP(in.s0, in.s1);
    *min_max_idx  = res.s0;
}

inline DATA_TYPE_OUTPUT vectorized_compute_arg_min_max_16(DATA_TYPE *min_max_val, DATA_TYPE_OUTPUT *min_max_idx, VEC_DATA_TYPE(DATA_TYPE, 16) in, VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16) res)
{
    VEC_DATA_TYPE(COND_DATA_TYPE, 8)
    idx_sel       = VECTOR_PREDICATE_EQ(in.s01234567, in.s89abcdef);
    in.s01234567  = select(in.s89abcdef, in.s01234567, idx_sel);
    res.s01234567 = select(res.s89abcdef, res.s01234567, CONVERT(idx_sel, int8));
    idx_sel.s0123 = VECTOR_PREDICATE(in.s0123, in.s4567) || (in.s0123 == in.s4567 && CONVERT(((res.s0123 < res.s4567)), VEC_DATA_TYPE(COND_DATA_TYPE, 4)));
    in.s0123      = select(in.s4567, in.s0123, idx_sel.s0123);
    res.s0123     = select(res.s4567, res.s0123, CONVERT(idx_sel.s0123, int4));
    idx_sel.s01   = (VECTOR_PREDICATE(in.s01, in.s23)) || (in.s01 == in.s23 && CONVERT(((res.s01 < res.s23)), VEC_DATA_TYPE(COND_DATA_TYPE, 2)));
    in.s01        = select(in.s23, in.s01, idx_sel.s01);
    res.s01       = select(res.s23, res.s01, CONVERT(idx_sel.s01, int2));
    idx_sel.s0    = VECTOR_PREDICATE(in.s0, in.s1) || (in.s0 == in.s1 && CONVERT((res.s0 < res.s1), COND_DATA_TYPE));
    res.s0        = select(res.s1, res.s0, CONVERT(idx_sel.s0, int));
    *min_max_val  = SCALAR_SELECT_OP(in.s0, in.s1);
    *min_max_idx  = res.s0;
}



inline void scalar_compute_global_min_max(DATA_TYPE in_val, int idx, DATA_TYPE *out_min_max_val, DATA_TYPE_OUTPUT *out_idx)
{
#if defined(ARG_MAX)
    if(in_val > *out_min_max_val)
#else  // defined(ARG_MAX)
    if(in_val < *out_min_max_val)
#endif // defined(ARG_MAX)
    {
        *out_min_max_val = in_val;
        *out_idx         = idx;
    }
}

#if VEC_SIZE > 1
#if VEC_SIZE == 16
    #define VECTORIZED_OP(min_max_val,min_max_idx,in,res) vectorized_compute_arg_min_max_16(min_max_val,min_max_idx,in,res)
#elif VEC_SIZE == 8 // #if VEC_SIZE == 16
    #define VECTORIZED_OP(min_max_val,min_max_idx,in,res) vectorized_compute_arg_min_max_8(min_max_val,min_max_idx,in,res)
#elif VEC_SIZE == 4 // # elif VEC_SIZE == 8
    #define VECTORIZED_OP(min_max_val,min_max_idx,in,res) vectorized_compute_arg_min_max_4(min_max_val,min_max_idx,in,res)
#elif VEC_SIZE == 2 // elif VEC_SIZE == 4
    #define VECTORIZED_OP(min_max_val,min_max_idx,in,res) vectorized_compute_arg_min_max_2(min_max_val,min_max_idx,in,res)
#else // elif VEC_SIZE == 2
    #error "Not supported"
#endif // #if VEC_SIZE == 16

inline VEC_DATA_TYPE(DATA_TYPE_OUTPUT, VEC_SIZE) init_idx_vector()
{
#if VEC_SIZE == 16
    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, VEC_SIZE)
    vidx = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
#elif VEC_SIZE == 8 // #if VEC_SIZE == 16
    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, VEC_SIZE)
    vidx = { 0, 1, 2, 3, 4, 5, 6, 7 };
#elif VEC_SIZE == 4 // elif VEC_SIZE == 8
    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, VEC_SIZE)
    vidx = { 0, 1, 2, 3 };
#elif VEC_SIZE == 2 // elif VEC_SIZE == 4
    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, VEC_SIZE)
    vidx = { 0, 1 };
#else  // elif VEC_SIZE == 2
#error "Not supported"
#endif // #if VEC_SIZE == 16
    return vidx;
}
#endif // VEC_SIZE > 1

/** This kernel performs reduction on x-axis.
 *
 * @note The input data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data type of the output must be passed at compile time using -DDATA_TYPE_OUTPUT: e.g. -DDATA_TYPE_OUTPUT=uint
 * @note The data type used for the comparing indexe must be passed at compile type using -DCOND_DATA_TYPE: e.g -DCOND_DATA_TYPE=uint
 * @note The height size must be passed at compile time using -DHEIGHT e.g. -DHEIGHT=128
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: U32/S32
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void arg_min_max_x(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    __global DATA_TYPE *input_addr         = (__global DATA_TYPE *)(input_ptr + input_offset_first_element_in_bytes + get_global_id(1) * input_stride_y);
    __global DATA_TYPE_OUTPUT *output_addr = (__global DATA_TYPE_OUTPUT *)(output_ptr + output_offset_first_element_in_bytes + get_global_id(1) * output_stride_y);

    DATA_TYPE        final_value = input_addr[0];
    DATA_TYPE_OUTPUT final_idx   = 0;

#if VEC_SIZE > 1
    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, VEC_SIZE)
    vidx = init_idx_vector();

    int x = 0;
    for(; x <= (WIDTH - VEC_SIZE); x += VEC_SIZE)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        vals = VLOAD(VEC_SIZE)(0, (input_addr + x));
        DATA_TYPE        local_min_max_value;
        DATA_TYPE_OUTPUT local_min_max_idx;

        VECTORIZED_OP(&local_min_max_value, &local_min_max_idx, vals, vidx);
        local_min_max_idx += x;
        scalar_compute_global_min_max(local_min_max_value, local_min_max_idx, &final_value, &final_idx);
    }
#endif // VEC_SIZE > 1

#if(WIDTH % VEC_SIZE)
    LOOP_UNROLLING(int, j, 0, 1, WIDTH % VEC_SIZE,
    {
        scalar_compute_global_min_max(*(input_addr + j + x), j + x, &final_value, &final_idx);
    })
#endif // (WIDTH % VEC_SIZE)

    output_addr[0] = final_idx;
}
#endif // defined(WIDTH)

#if defined(HEIGHT)
/** This kernel performs reduction on y-axis.
 *
 * @note The input data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note The data type of the output must be passed at compile time using -DDATA_TYPE_OUTPUT: e.g. -DDATA_TYPE_OUTPUT=uint
 * @note The height size must be passed at compile time using -DHEIGHT e.g. -DHEIGHT=128
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: U32/S32
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void arg_min_max_y(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    const int x_offs            = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * input_stride_y;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE_OUTPUT) + get_global_id(1) * output_stride_y;

    VEC_TYPE_IN res = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr), VEC_TYPE_IN);

    VEC_TYPE_OUT indx0 = 0;
    for(DATA_TYPE_OUTPUT y = 1; y < HEIGHT; ++y)
    {
        VEC_TYPE_IN in = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + y * input_stride_y)), VEC_TYPE_IN);

        VEC_TYPE_OUT cond_conv = CONVERT(CONDITION_TO_USE(in, res), VEC_TYPE_OUT);
        indx0                  = select(indx0, (VEC_TYPE_OUT)y, cond_conv);
        res                    = select(res, in, CONDITION_TO_USE(in, res));
    }

    // Store result
    STORE_VECTOR_SELECT(indx, DATA_TYPE_OUTPUT, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif // defined(HEIGHT)

#if defined(DEPTH) && !defined(BATCH)
/** This kernel performs reduction on z-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note The depth size must be passed at compile time using -DDEPTH e.g. -DDEPTH=128
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: U32/S32
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_stride_z                      Stride of the output tensor in Z dimension (in bytes)
 * @param[in] output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void arg_min_max_z(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    const int x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);

    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * input_stride_y + get_global_id(2) * input_stride_z;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE_OUTPUT) + get_global_id(1) * output_stride_y + get_global_id(2) * output_stride_z;

    VEC_TYPE_IN res = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr), VEC_TYPE_IN);

    VEC_TYPE_OUT indx0 = 0;
    for(DATA_TYPE_OUTPUT z = 1; z < DEPTH; ++z)
    {
        VEC_TYPE_IN in = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + z * input_stride_z)), VEC_TYPE_IN);

        VEC_TYPE_OUT cond_conv = CONVERT(CONDITION_TO_USE(in, res), VEC_TYPE_OUT);
        indx0                  = select(indx0, (VEC_TYPE_OUT)z, cond_conv);
        res                    = select(res, in, CONDITION_TO_USE(in, res));
    }

    // Store result
    STORE_VECTOR_SELECT(indx, DATA_TYPE_OUTPUT, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif /* defined(DEPTH)  && !defined(BATCH) */

#if defined(BATCH) && defined(DEPTH)
/** This kernel performs reduction on w-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note The batch size must be passed at compile time using -DBATCH e.g. -DBATCH=128
 * @note The depth size must be passed at compile time using -DBATCH e.g. -DDEPTH=128
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] input_stride_w                       Stride of the source tensor in W dimension (in bytes)
 * @param[in] input_step_w                         input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: U32/S32
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_stride_z                      Stride of the output tensor in Z dimension (in bytes)
 * @param[in] output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] output_stride_w                      Stride of the output tensor in W dimension (in bytes)
 * @param[in] output_step_w                        output_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void arg_min_max_w(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output))
{
    const int x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);

    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * input_stride_y + (get_global_id(2) % DEPTH) * input_stride_z +
                                  (get_global_id(2) / DEPTH) * input_stride_w;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE_OUTPUT) + get_global_id(1) * output_stride_y + (get_global_id(
                                      2) % DEPTH) * output_stride_z + (get_global_id(2) / DEPTH) * output_stride_w;

    VEC_TYPE_IN res = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr), VEC_TYPE_IN);

    VEC_TYPE_OUT indx0 = 0;
    for(DATA_TYPE_OUTPUT w = 1; w < BATCH; ++w)
    {
        VEC_TYPE_IN in = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + w * input_stride_w)), VEC_TYPE_IN);

        VEC_TYPE_OUT cond_conv = CONVERT(CONDITION_TO_USE(in, res), VEC_TYPE_OUT);
        indx0                  = select(indx0, (VEC_TYPE_OUT)w, cond_conv);
        res                    = select(res, in, CONDITION_TO_USE(in, res));
    }

    // Store result
    STORE_VECTOR_SELECT(indx, DATA_TYPE_OUTPUT, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif /* defined(BATCH) && defined(DEPTH) */
#endif // defined(VEC_SIZE) && defined(DATA_TYPE) && defined(DATA_TYPE_OUTPUT)
