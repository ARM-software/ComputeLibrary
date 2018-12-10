/*
 * Copyright (c) 2016-2019 ARM Limited.
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

/** Calculate square sum of a vector
 *
 * @param[in] input Pointer to the first pixel.
 *
 * @return square sum of vector.
 */
inline DATA_TYPE square_sum(__global const DATA_TYPE *input)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, input);

    in *= in;

    in.s01234567 += in.s89ABCDEF;
    in.s0123 += in.s4567;
    in.s01 += in.s23;

    return (in.s0 + in.s1);
}

/** Calculate sum of a vector
 *
 * @param[in] input Pointer to the first pixel.
 *
 * @return sum of vector.
 */
inline DATA_TYPE sum(__global const DATA_TYPE *input)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, input);

    in.s01234567 += in.s89ABCDEF;
    in.s0123 += in.s4567;
    in.s01 += in.s23;

    return (in.s0 + in.s1);
}

/** Calculate product of a vector
 *
 * @param[in] input Pointer to the first pixel.
 *
 * @return product of vector.
 */
inline DATA_TYPE product(__global const DATA_TYPE *input)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, input);

    in.s01234567 *= in.s89ABCDEF;
    in.s0123 *= in.s4567;
    in.s01 *= in.s23;

    return (in.s0 * in.s1);
}
#if defined(OPERATION)
/** This kernel performs parallel reduction given an operation on x-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The operation we want to perform must be passed at compile time using -DOPERATION e.g. -DOPERATION=square_sum
 * @note The mean flag must be passed at compile time using -DMEAN if we want to compute the mean value
 * @note The product flag must be passed at compile time using -DPROD if we want to compute the product, otherwise sum will be used
 * @note The width size must be passed at compile time using -DWIDTH e.g. -DWIDTH=128 if we want to compute the mean value
 *
 * @param[in] src_ptr                                   Pointer to the source tensor. Supported data types: F16/F32
 * @param[in] src_stride_x                              Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                                src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                              Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                                src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes         The offset of the first element in the source tensor
 * @param[in] partial_res_ptr                           The local buffer to hold partial result values. Supported data types: same as @p src_ptr
 * @param[in] partial_res_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] partial_res_step_x                        partial_res_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] partial_res_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] partial_res_step_y                        partial_res_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] partial_res_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] local_results                             Local buffer for storing the partial result
 */
__kernel void reduction_operation_x(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(partial_res),
    __local DATA_TYPE *local_results)
{
    Image src         = CONVERT_TO_IMAGE_STRUCT(src);
    Image partial_res = CONVERT_TO_IMAGE_STRUCT(partial_res);

    unsigned int lsize = get_local_size(0);
    unsigned int lid   = get_local_id(0);

    for(unsigned int y = 0; y < get_local_size(1); ++y)
    {
        local_results[lid] = OPERATION((__global DATA_TYPE *)offset(&src, 0, y));
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform parallel reduction
        for(unsigned int i = lsize >> 1; i > 0; i >>= 1)
        {
            if(lid < i)
            {
#if defined(PROD)
                local_results[lid] *= local_results[lid + i];
#else  //!defined(PROD)
                local_results[lid] += local_results[lid + i];
#endif //defined(PROD)
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(lid == 0)
        {
#if defined(MEAN) && defined(WIDTH)
            if(y == get_local_size(1) - 1)
            {
                local_results[0] /= WIDTH;
            }
#endif /* defined(MEAN) && defined(WIDTH) */
            ((__global DATA_TYPE *)offset(&partial_res, get_group_id(0), y))[0] = local_results[0];
        }
    }
}
#endif // defined(OPERATION)

#if defined(WIDTH)
/** This kernel performs reduction on x-axis. (Non parallel)
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width size must be passed at compile time using -DWIDTH e.g. -DWIDTH=128
 * @note The product flag must be passed at compile time using -DPROD if we want to compute the product, otherwise sum will be used
 * @note In case of ARG_MIN and ARG_MAX the condition data type must be passed at compile time using -DCOND_DATA_TYPE e.g. -DCOND_DATA_TYPE=short
 *
 * @param[in] src_ptr                              Pointer to the source tensor. Supported data types: F16/F32 and QASYMM8 for operation MEAN
 * @param[in] src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p src_ptt
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void reduction_operation_non_parallel_x(
    VECTOR_DECLARATION(src),
    VECTOR_DECLARATION(output))
{
    Vector src    = CONVERT_TO_VECTOR_STRUCT(src);
    Vector output = CONVERT_TO_VECTOR_STRUCT(output);

    DATA_TYPE_PROMOTED res = *((__global DATA_TYPE *)vector_offset(&src, 0));

#if defined(ARG_MAX) || defined(ARG_MIN)
    uint indx = 0;
#endif // defined(ARG_MAX) || defined(ARG_MIN)

    for(unsigned int x = 1; x < WIDTH; ++x)
    {
        DATA_TYPE_PROMOTED in = *((__global DATA_TYPE *)vector_offset(&src, x));
#if defined(ARG_MAX)
        indx = select(indx, x, isgreater(in, res));
        res  = select(res, in, CONVERT(isgreater(in, res), COND_DATA_TYPE));
#elif defined(ARG_MIN)
        indx = select(indx, x, isless(in, res));
        res  = select(res, in, CONVERT(isless(in, res), COND_DATA_TYPE));
#else  // !(defined(ARG_MAX) || defined(ARG_MIN))
        res += in;
#endif // defined(ARG_MAX) || defined(ARG_MIN)
    }

    // Store result
#if defined(ARG_MAX) || defined(ARG_MIN)
    *((__global uint *)output.ptr) = indx;
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#if defined(MEAN)
    res /= WIDTH;
#endif // defined(MEAN)
    *((__global uchar *)output.ptr) = convert_uchar(res);
#endif // defined(ARG_MAX) || defined(ARG_MIN)
}
#endif /* defined(WIDTH) */

#if defined(HEIGHT)
/** This kernel performs reduction on y-axis.
 *
 * @note The input data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The height size must be passed at compile time using -DHEIGHT e.g. -DHEIGHT=128
 *
 * @param[in] src_ptr                              Pointer to the source tensor. Supported data types: QASYMM8/F16/F32
 * @param[in] src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p src_ptt
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void reduction_operation_y(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(output))
{
    Image src    = CONVERT_TO_IMAGE_STRUCT(src);
    Image output = CONVERT_TO_IMAGE_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16)
    res = CONVERT(vload16(0, (__global DATA_TYPE *)offset(&src, 0, 0)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16));

#if defined(SUM_SQUARE)
    res *= res;
#endif // defined(SUM_SQUARE)

#if defined(ARG_MAX) || defined(ARG_MIN)
    uint16 indx = 0;
#endif // defined(ARG_MAX) || defined(ARG_MIN)

    for(unsigned int y = 1; y < HEIGHT; ++y)
    {
        VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16)
        in = CONVERT(vload16(0, (__global DATA_TYPE *)offset(&src, 0, y)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16));
#if defined(ARG_MAX)
        uint16 cond_conv = CONVERT(isgreater(in, res), uint16);
        indx             = select(indx, y, cond_conv);
        res              = select(res, in, isgreater(in, res));
#elif defined(ARG_MIN)
        uint16  cond_conv           = CONVERT(isless(in, res), uint16);
        indx                        = select(indx, y, cond_conv);
        res                         = select(res, in, isless(in, res));
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#if defined(SUM_SQUARE)
        in *= in;
#endif // defined(SUM_SQUARE)
#if defined(PROD)
        res *= in;
#else  //!defined(PROD)
        res += in;
#endif //defined(PROD)
#endif // defined(ARG_MAX) || defined(ARG_MIN)
    }

    // Store result
#if defined(ARG_MAX) || defined(ARG_MIN)
    vstore16(indx, 0, (__global uint *)output.ptr);
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#if defined(MEAN)
    res /= HEIGHT;
#endif // defined(MEAN)
    vstore16(CONVERT(res, VEC_DATA_TYPE(DATA_TYPE, 16)), 0, (__global DATA_TYPE *)output.ptr);
#endif // defined(ARG_MAX) || defined(ARG_MIN)
}
#endif /* defined(HEIGHT) */

#if defined(DEPTH)
/** This kernel performs reduction on z-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The depth size must be passed at compile time using -DDEPTH e.g. -DDEPTH=128
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p input_ptt
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_stride_z                      Stride of the output tensor in Z dimension (in bytes)
 * @param[in] output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void reduction_operation_z(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16)
    res = CONVERT(vload16(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, 0)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16));

#if defined(SUM_SQUARE)
    res *= res;
#endif // defined(SUM_SQUARE)

#if defined(ARG_MAX) || defined(ARG_MIN)
    uint16 indx = 0;
#endif // defined(ARG_MAX) || defined(ARG_MIN)

    for(unsigned int z = 1; z < DEPTH; ++z)
    {
        VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16)
        in = CONVERT(vload16(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 0, z)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16));

#if defined(ARG_MAX)
        uint16 cond_conv = CONVERT(isgreater(in, res), uint16);
        indx             = select(indx, z, cond_conv);
        res              = select(res, in, isgreater(in, res));
#elif defined(ARG_MIN)
        uint16 cond_conv = CONVERT(isless(in, res), uint16);
        indx             = select(indx, z, cond_conv);
        res              = select(res, in, isless(in, res));
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#if defined(SUM_SQUARE)
        in *= in;
#endif // defined(SUM_SQUARE)
#if defined(PROD)
        res *= in;
#else  //!defined(PROD)
        res += in;
#endif //defined(PROD)
#endif // defined(ARG_MAX) || defined(ARG_MIN)
    }

    // Store result
#if defined(ARG_MAX) || defined(ARG_MIN)
    vstore16(indx, 0, (__global uint *)output.ptr);
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#if defined(MEAN)
    res /= DEPTH;
#endif // defined(MEAN)
    vstore16(CONVERT(res, VEC_DATA_TYPE(DATA_TYPE, 16)), 0, (__global DATA_TYPE *)output.ptr);
#endif // defined(ARG_MAX) || defined(ARG_MIN)
}
#endif /* defined(DEPTH) */

#if defined(BATCH) && defined(DEPTH)
/** This kernel performs reduction on w-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The batch size must be passed at compile time using -DBATCH e.g. -DBATCH=128
 * @note The depth size must be passed at compile time using -DBATCH e.g. -DDEPTH=128
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] input_stride_w                       Stride of the source tensor in W dimension (in bytes)
 * @param[in] input_step_w                         input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p input_ptt
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
__kernel void reduction_operation_w(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output))
{
    Tensor4D input  = CONVERT_TO_TENSOR4D_STRUCT(input, DEPTH);
    Tensor4D output = CONVERT_TO_TENSOR4D_STRUCT(output, DEPTH);

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16)
    res = CONVERT(vload16(0, (__global DATA_TYPE *)tensor4D_offset(&input, 0, 0, 0, 0)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16));

#if defined(SUM_SQUARE)
    res *= res;
#endif // defined(SUM_SQUARE)

#if defined(ARG_MAX) || defined(ARG_MIN)
    uint16 indx = 0;
#endif // defined(ARG_MAX) || defined(ARG_MIN)

    for(unsigned int w = 1; w < BATCH; ++w)
    {
        VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16)
        in = CONVERT(vload16(0, (__global DATA_TYPE *)tensor4D_offset(&input, 0, 0, 0, w)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 16));

#if defined(ARG_MAX)
        uint16 cond_conv = CONVERT(isgreater(in, res), uint16);
        indx             = select(indx, w, cond_conv);
        res              = select(res, in, isgreater(in, res));
#elif defined(ARG_MIN)
        uint16 cond_conv = CONVERT(isless(in, res), uint16);
        indx             = select(indx, w, cond_conv);
        res              = select(res, in, isless(in, res));
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#if defined(SUM_SQUARE)
        in *= in;
#endif // defined(SUM_SQUARE)
#if defined(PROD)
        res *= in;
#else  //!defined(PROD)
        res += in;
#endif //defined(PROD)
#endif // defined(ARG_MAX) || defined(ARG_MIN)
    }

    // Store result
#if defined(ARG_MAX) || defined(ARG_MIN)
    vstore16(indx, 0, (__global uint *)output.ptr);
#else // !(defined(ARG_MAX) || defined(ARG_MIN))
#if defined(MEAN)
    res /= BATCH;
#endif // defined(MEAN)
    vstore16(CONVERT(res, VEC_DATA_TYPE(DATA_TYPE, 16)), 0, (__global DATA_TYPE *)output.ptr);
#endif // defined(ARG_MAX) || defined(ARG_MIN)
}
#endif /* defined(BATCH) && defined(DEPTH) */
