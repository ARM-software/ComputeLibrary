/*
 * Copyright (c) 2019-2020 ARM Limited.
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

#if defined(FLOAT_DATA_TYPE)
#define ISGREATER(x, y) isgreater(x, y)
#define ISLESS(x, y) isless(x, y)
#else // !FLOAT_DATA_TYPE
#if defined(WIDTH)
#define ISGREATER(x, y) (x > y) ? 1 : 0
#define ISLESS(x, y) (x < y) ? 1 : 0
#else // !defined(WIDTH)
#define ISGREATER(x, y) select((VEC_DATA_TYPE(DATA_TYPE_SELECT, 16))0, (VEC_DATA_TYPE(DATA_TYPE_SELECT, 16)) - 1, x > y)
#define ISLESS(x, y) select((VEC_DATA_TYPE(DATA_TYPE_SELECT, 16))0, (VEC_DATA_TYPE(DATA_TYPE_SELECT, 16)) - 1, x < y)
#endif // defined(WIDTH)
#endif // defined(FLOAT_DATA_TYPE)

#if defined(ARG_MAX)
#define CONDITION_TO_USE(x, y) ISGREATER(x, y)
#elif defined(ARG_MIN)
#define CONDITION_TO_USE(x, y) ISLESS(x, y)
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#error "Unsupported reduction operation!"
#endif // defined(ARG_MAX)

#if defined(DATA_TYPE_OUTPUT) && defined(DATA_TYPE_SELECT)
#if defined(WIDTH)
#if defined(ARG_MIN)
#if defined(PREV_OUTPUT)
/** Find index minimum value of a vector
 *
 * @param[in] input Pointer to the first value.
 *
 * @return index of the vector.
 */
inline DATA_TYPE_OUTPUT arg_idx_min_prev_out(__global const DATA_TYPE *input, __global const DATA_TYPE_OUTPUT *prev_res, const int x_idx)
{
    int end_elem = (x_idx + 1) * 16;
    if(end_elem > WIDTH)
    {
        end_elem = WIDTH - x_idx * 16;
    }
    DATA_TYPE_OUTPUT res = prev_res[0];
    for(int x_v = 1; x_v < end_elem; ++x_v)
    {
        res = select(res, prev_res[x_v], *(input + prev_res[x_v]) < * (input + res));
    }
    return res;
}
#else // !defined(PREV_OUTPUT)
/** Find index minimum value of a vector
 *
 * @param[in] input Pointer to the first value.
 *
 * @return index of the vector.
 */
inline DATA_TYPE_OUTPUT arg_idx_min(__global const DATA_TYPE *input, const int x_idx)
{
#if WIDTH < 16
    DATA_TYPE_OUTPUT res = 0;
    for(DATA_TYPE_OUTPUT x_v = res + 1; x_v < WIDTH; ++x_v)
    {
        res = select(res, x_v, *(input + x_v) < * (input + res));
    }
    return res;
#else  // WIDTH >= 16
    int       x_elem   = x_idx * 16;
    const int x_goback = select(0, 16 - WIDTH % 16, x_elem + 16 > WIDTH);
    x_elem -= x_goback;

    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, input - x_goback);
    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16)
    res = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    VEC_DATA_TYPE(DATA_TYPE_SELECT, 8)
    idx_sel       = (in.s01234567 <= in.s89abcdef);
    in.s01234567  = select(in.s89abcdef, in.s01234567, idx_sel);
    res.s01234567 = select(res.s89abcdef, res.s01234567, CONVERT(idx_sel, int8));

    idx_sel.s0123 = (in.s0123 < in.s4567) || (in.s0123 == in.s4567 && CONVERT((res.s0123 < res.s4567), VEC_DATA_TYPE(DATA_TYPE_SELECT, 4)));
    in.s0123      = select(in.s4567, in.s0123, idx_sel.s0123);
    res.s0123     = select(res.s4567, res.s0123, CONVERT(idx_sel.s0123, int4));

    idx_sel.s01 = (in.s01 < in.s23) || (in.s01 == in.s23 && CONVERT((res.s01 < res.s23), VEC_DATA_TYPE(DATA_TYPE_SELECT, 2)));
    in.s01      = select(in.s23, in.s01, idx_sel.s01);
    res.s01     = select(res.s23, res.s01, CONVERT(idx_sel.s01, int2));

    idx_sel.s0 = (in.s0 < in.s1) || (in.s0 == in.s1 && CONVERT((res.s0 < res.s1), DATA_TYPE_SELECT));
    res.s0     = select(res.s1, res.s0, CONVERT(idx_sel.s0, int));

    return res.s0 + x_elem;
#endif // WIDTH < 16
}
#endif // defined(PREV_OUTPUT)
#endif // defined(ARG_MIN)
#if defined(ARG_MAX)
#if defined(PREV_OUTPUT)
/** Find index maximum value of a vector
 *
 * @param[in] input Pointer to the first value.
 *
 * @return index of the vector.
 */
inline DATA_TYPE_OUTPUT arg_idx_max_prev_out(__global const DATA_TYPE *input, __global const DATA_TYPE_OUTPUT *prev_res, const int x_idx)
{
    int end_elem = (x_idx + 1) * 16;
    if(end_elem > WIDTH)
    {
        end_elem = WIDTH - x_idx * 16;
    }
    DATA_TYPE_OUTPUT res = prev_res[0];
    for(int x_v = 1; x_v < end_elem; ++x_v)
    {
        res = select(res, prev_res[x_v], *(input + prev_res[x_v]) > *(input + res));
    }
    return res;
}
#else // !defined(PREV_OUTPUT)
/** Find index maximum value of a vector
 *
 * @param[in] input Pointer to the first value.
 *
 * @return index of the vector.
 */
inline DATA_TYPE_OUTPUT arg_idx_max(__global const DATA_TYPE *input, const int x_idx)
{
#if WIDTH < 16
    DATA_TYPE_OUTPUT res = 0;
    for(DATA_TYPE_OUTPUT x_v = res + 1; x_v < WIDTH; ++x_v)
    {
        res = select(res, x_v, *(input + x_v) > *(input + res));
    }
    return res;
#else  // WIDTH >= 16
    int       x_elem   = x_idx * 16;
    const int x_goback = select(0, 16 - WIDTH % 16, x_elem + 16 > WIDTH);
    x_elem -= x_goback;

    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, input - x_goback);
    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16)
    res = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    VEC_DATA_TYPE(DATA_TYPE_SELECT, 8)
    idx_sel       = (in.s01234567 >= in.s89abcdef);
    in.s01234567  = select(in.s89abcdef, in.s01234567, idx_sel);
    res.s01234567 = select(res.s89abcdef, res.s01234567, CONVERT(idx_sel, int8));

    idx_sel.s0123 = (in.s0123 > in.s4567) || (in.s0123 == in.s4567 && CONVERT((res.s0123 < res.s4567), VEC_DATA_TYPE(DATA_TYPE_SELECT, 4)));
    in.s0123      = select(in.s4567, in.s0123, idx_sel.s0123);
    res.s0123     = select(res.s4567, res.s0123, CONVERT(idx_sel.s0123, int4));

    idx_sel.s01 = (in.s01 > in.s23) || (in.s01 == in.s23 && CONVERT((res.s01 < res.s23), VEC_DATA_TYPE(DATA_TYPE_SELECT, 2)));
    in.s01      = select(in.s23, in.s01, idx_sel.s01);
    res.s01     = select(res.s23, res.s01, CONVERT(idx_sel.s01, int2));

    idx_sel.s0 = (in.s0 > in.s1) || (in.s0 == in.s1 && CONVERT((res.s0 < res.s1), DATA_TYPE_SELECT));
    res.s0     = select(res.s1, res.s0, CONVERT(idx_sel.s0, int));

    return res.s0 + x_elem;
#endif // WIDTH < 16
}
#endif // defined(PREV_OUTPUT)
#endif // defined(ARG_MAX)

/** This kernel performs parallel reduction given an operation on x-axis.
 *
 * @note In case the results of previous stages are passed the flag PREV_OUTPUT has to be passed using -DPREV_OUTPUT
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data type of the output must be passed at compile time using -DDATA_TYPE_OUTPUT: e.g. -DDATA_TYPE_OUTPUT=uint
 * @note The arg_max flag must be passed at compile time using -DARG_MAX if we want to compute the ArgMax
 * @note The arg_min flag must be passed at compile time using -DARG_MIN if we want to compute the ArgMin
 *
 * @param[in] src_ptr                                   Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32
 * @param[in] src_stride_x                              Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                                src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                              Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                                src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes         The offset of the first element in the source tensor
 * @param[in] prev_res_ptr                              (Optional) Pointer to previous results tensor. Supported data types: U32/S32
 * @param[in] prev_res_stride_x                         (Optional) Stride of the output tensor in X dimension (in bytes)
 * @param[in] prev_res_step_x                           (Optional) prev_res_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] prev_res_stride_y                         (Optional) Stride of the output tensor in Y dimension (in bytes)
 * @param[in] prev_res_step_y                           (Optional) prev_res_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] prev_res_offset_first_element_in_bytes    (Optional) The offset of the first element in the previous results tensor
 * @param[in] partial_res_ptr                           The local buffer to hold partial result values. Supported data types: U32/S32
 * @param[in] partial_res_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] partial_res_step_x                        partial_res_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] partial_res_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] partial_res_step_y                        partial_res_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] partial_res_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] local_results                             Local buffer for storing the partial result
 */
__kernel void arg_min_max_x(
    IMAGE_DECLARATION(src),
#if defined(PREV_OUTPUT)
    IMAGE_DECLARATION(prev_res),
#endif // defined(PREV_OUTPUT)
    IMAGE_DECLARATION(partial_res),
    __local DATA_TYPE_OUTPUT *local_results)
{
#if defined(PREV_OUTPUT)
    Image src      = CONVERT_TO_IMAGE_STRUCT_NO_STEP(src);
    Image prev_res = CONVERT_TO_IMAGE_STRUCT(prev_res);
#else  // !defined(PREV_OUTPUT)
    Image src                      = CONVERT_TO_IMAGE_STRUCT(src);
#endif // defined(PREV_OUTPUT)
    Image partial_res = CONVERT_TO_IMAGE_STRUCT(partial_res);

    unsigned int lsize = get_local_size(0);
    unsigned int lid   = get_local_id(0);

    const uint     x_idx                 = get_global_id(0);
    const uint     y_idx                 = get_global_id(1);
    const __global DATA_TYPE *src_in_row = (const __global DATA_TYPE *)(src_ptr + src_offset_first_element_in_bytes + y_idx * src_step_y);

    for(unsigned int y = 0; y < get_local_size(1); ++y)
    {
#if defined(ARG_MAX)
#if defined(PREV_OUTPUT)
        local_results[lid] = arg_idx_max_prev_out(src_in_row, (__global DATA_TYPE_OUTPUT *)offset(&prev_res, 0, y), x_idx);
#else  // !defined(PREV_OUTPUT)
        local_results[lid] = arg_idx_max((__global DATA_TYPE *)offset(&src, 0, y), x_idx);
#endif // defined(PREV_OUTPUT)
#else  // defined(ARG_MIN)
#if defined(PREV_OUTPUT)
        local_results[lid]         = arg_idx_min_prev_out(src_in_row, (__global DATA_TYPE_OUTPUT *)offset(&prev_res, 0, y), x_idx);
#else  // !defined(PREV_OUTPUT)
        local_results[lid] = arg_idx_min((__global DATA_TYPE *)offset(&src, 0, y), x_idx);
#endif // defined(PREV_OUTPUT)
#endif // defined(ARG_MAX) || defined(ARG_MIN)

        barrier(CLK_LOCAL_MEM_FENCE);

        // Looking for the next highest power of 2 (maximum value of lsize is 8)
        unsigned int middle = lsize - 1;
        middle |= middle >> 1;
        middle |= middle >> 2;
        middle += 1;
        // Perform parallel reduction
        for(unsigned int i = middle; i > 0; i >>= 1)
        {
            if(lid < i && lid + i < lsize)
            {
                DATA_TYPE tmp0 = *(src_in_row + local_results[lid]);
                DATA_TYPE tmp1 = *(src_in_row + local_results[lid + i]);
#if defined(ARG_MAX)
                local_results[lid] = select(
                                         local_results[lid],
                                         local_results[lid + i],
                                         ((tmp0 == tmp1) && (local_results[lid + i] < local_results[lid])) || (tmp0 < tmp1));
#else  // defined(ARG_MIN)
                local_results[lid] = select(
                                         local_results[lid],
                                         local_results[lid + i],
                                         ((tmp0 == tmp1) && (local_results[lid + i] < local_results[lid])) || (tmp0 > tmp1));
#endif // defined(ARG_MAX) || defined(ARG_MIN)
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(lid == 0)
        {
            ((__global DATA_TYPE_OUTPUT *)offset(&partial_res, get_group_id(0), y))[0] = local_results[0];
        }
    }
}
#endif // defined(WIDTH)

#if defined(HEIGHT)
/** This kernel performs reduction on y-axis.
 *
 * @note The input data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data type of the output must be passed at compile time using -DDATA_TYPE_OUTPUT: e.g. -DDATA_TYPE_OUTPUT=uint
 * @note The data type of the select results must be passed at compile time using -DDATA_TYPE_SELECT: e.g. -DDATA_TYPE_SELECT=int
 * @note The height size must be passed at compile time using -DHEIGHT e.g. -DHEIGHT=128
 *
 * @param[in] src_ptr                              Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32
 * @param[in] src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: U32/S32
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void arg_min_max_y(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(output))
{
    Image src    = CONVERT_TO_IMAGE_STRUCT(src);
    Image output = CONVERT_TO_IMAGE_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    res = CONVERT(vload16(0, (__global DATA_TYPE *)offset(&src, 0, 0)), VEC_DATA_TYPE(DATA_TYPE, 16));

    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16)
    indx = 0;
    for(unsigned int y = 1; y < HEIGHT; ++y)
    {
        VEC_DATA_TYPE(DATA_TYPE, 16)
        in = CONVERT(vload16(0, (__global DATA_TYPE *)offset(&src, 0, y)), VEC_DATA_TYPE(DATA_TYPE, 16));

        VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16)
        cond_conv = CONVERT(CONDITION_TO_USE(in, res), VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16));
        indx      = select(indx, y, cond_conv);
        res       = select(res, in, CONDITION_TO_USE(in, res));
    }

    // Store result
    vstore16(indx, 0, (__global DATA_TYPE_OUTPUT *)output.ptr);
}
#endif // defined(HEIGHT)

#if defined(DEPTH)
/** This kernel performs reduction on z-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data type of the select results must be passed at compile time using -DDATA_TYPE_SELECT: e.g. -DDATA_TYPE_SELECT=int
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
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    res = CONVERT(vload16(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0)), VEC_DATA_TYPE(DATA_TYPE, 16));

    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16)
    indx = 0;
    for(DATA_TYPE_OUTPUT z = 1; z < DEPTH; ++z)
    {
        VEC_DATA_TYPE(DATA_TYPE, 16)
        in = CONVERT(vload16(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, z)), VEC_DATA_TYPE(DATA_TYPE, 16));

        VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16)
        cond_conv = CONVERT(CONDITION_TO_USE(in, res), VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16));
        indx      = select(indx, z, cond_conv);
        res       = select(res, in, CONDITION_TO_USE(in, res));
    }

    // Store result
    vstore16(indx, 0, (__global DATA_TYPE_OUTPUT *)output.ptr);
}
#endif /* defined(DEPTH) */

#if defined(BATCH) && defined(DEPTH)
/** This kernel performs reduction on w-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data type of the select results must be passed at compile time using -DDATA_TYPE_SELECT: e.g. -DDATA_TYPE_SELECT=int
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
    Tensor4D input  = CONVERT_TO_TENSOR4D_STRUCT(input, DEPTH);
    Tensor4D output = CONVERT_TO_TENSOR4D_STRUCT(output, DEPTH);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    res = CONVERT(vload16(0, (__global DATA_TYPE *)tensor4D_offset(&input, 0, 0, 0, 0)), VEC_DATA_TYPE(DATA_TYPE, 16));

    VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16)
    indx = 0;
    for(DATA_TYPE_OUTPUT w = 1; w < BATCH; ++w)
    {
        VEC_DATA_TYPE(DATA_TYPE, 16)
        in = CONVERT(vload16(0, (__global DATA_TYPE *)tensor4D_offset(&input, 0, 0, 0, w)), VEC_DATA_TYPE(DATA_TYPE, 16));

        VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16)
        cond_conv = CONVERT(CONDITION_TO_USE(in, res), VEC_DATA_TYPE(DATA_TYPE_OUTPUT, 16));
        indx      = select(indx, w, cond_conv);
        res       = select(res, in, CONDITION_TO_USE(in, res));
    }

    // Store result
    vstore16(indx, 0, (__global DATA_TYPE_OUTPUT *)output.ptr);
}
#endif /* defined(BATCH) && defined(DEPTH) */
#endif /* defined(DATA_TYPE_OUTPUT) && defined(DATA_TYPE_SELECT) */