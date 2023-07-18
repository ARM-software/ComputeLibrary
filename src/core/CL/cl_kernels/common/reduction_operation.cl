/*
 * Copyright (c) 2016-2021, 2023 Arm Limited.
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
#include "helpers_asymm.h"

#if defined(FLOAT_DATA_TYPE)
#define ISGREATER(x, y) (SELECT_VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE))(isgreater(x, y))
#define ISLESS(x, y) (SELECT_VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE))(isless(x, y))
#define ISGREATER_SCALAR(x, y) (SELECT_DATA_TYPE(DATA_TYPE_PROMOTED))(isgreater(x, y))
#define ISLESS_SCALAR(x, y) (SELECT_DATA_TYPE(DATA_TYPE_PROMOTED))(isless(x, y))
#else // !FLOAT_DATA_TYPE
#if defined(WIDTH)
#define ISGREATER(x, y) (x > y) ? 1 : 0
#define ISLESS(x, y) (x < y) ? 1 : 0
#define ISGREATER_SCALAR ISGREATER
#define ISLESS_SCALAR ISLESS
#else // !defined(WIDTH)
#define ISGREATER(x, y) select((VEC_DATA_TYPE(int, VEC_SIZE))0, (VEC_DATA_TYPE(int, VEC_SIZE)) - 1, x > y)
#define ISLESS(x, y) select((VEC_DATA_TYPE(int, VEC_SIZE))0, (VEC_DATA_TYPE(int, VEC_SIZE)) - 1, x < y)
#endif // defined(WIDTH)
#endif // defined(FLOAT_DATA_TYPE)

#if defined(WIDTH)
#if defined(OPERATION)

#define sum(in0, in1, size) (in0 + SUM_REDUCE(in1, size))
#define square_sum(in0, in1, size) (in0 + SUM_REDUCE((in1 * in1), size))
#define product(in0, in1, size) (in0 * PROD_REDUCE(in1, size))
#define min_(in0, in1, size) (min(in0, MIN_REDUCE(in1, size)))
#define max_(in0, in1, size) (max(in0, MAX_REDUCE(in1, size)))

/** This kernel performs parallel reduction given an operation on x-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The operation we want to perform must be passed at compile time using -DOPERATION e.g. -DOPERATION=square_sum
 * @note The mean flag must be passed at compile time using -DMEAN if we want to compute the mean value
 * @note The product flag must be passed at compile time using -DPROD if we want to compute the product, otherwise sum will be used
 * @note The width size must be passed at compile time using -DWIDTH e.g. -DWIDTH=128 if we want to compute the mean value
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input
 * @param[in] output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void reduction_operation_x(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + y * input_stride_y + z * input_stride_z;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + y * output_stride_y + z * output_stride_z;

#if !defined(MIN) && !defined(MAX)
#if defined(PROD)
    DATA_TYPE res = (DATA_TYPE)1;
#else  // defined(PROD)
    DATA_TYPE res = (DATA_TYPE)0;
#endif // defined(PROD)
#else  // #if !defined(MIN) && !defined(MAX)
    DATA_TYPE res = *((__global DATA_TYPE *)input_addr);
#endif // #if defined(MIN) || defined(MAX)
    int x = 0;

    for(; x <= (WIDTH - VEC_SIZE); x += VEC_SIZE)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        vals = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + x * sizeof(DATA_TYPE)));
        res  = OPERATION(res, vals, VEC_SIZE);
    }

#if(WIDTH % VEC_SIZE)
    _Pragma("unroll") for(; x < WIDTH; ++x)
    {
        DATA_TYPE val = *((__global DATA_TYPE *)(input_addr + x * sizeof(DATA_TYPE)));
        res           = OPERATION(res, val, 1);
    }
#endif // (WIDTH % VEC_SIZE)

#if defined(MEAN)
    res /= WIDTH;
#endif // defined(MEAN)
    *((__global DATA_TYPE *)output_addr) = res;
}
#endif // defined(OPERATION)
/** This kernel performs reduction on x-axis. (Non parallel)
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The width size must be passed at compile time using -DWIDTH e.g. -DWIDTH=128
 * @note The product flag must be passed at compile time using -DPROD if we want to compute the product, otherwise sum will be used
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: S32/F16/F32 and QASYMM8/QASYMM8_SIGNED for operation MEAN
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p input_ptr
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void reduction_operation_non_parallel_x(
    VECTOR_DECLARATION(input),
    VECTOR_DECLARATION(output))
{
    Vector input  = CONVERT_TO_VECTOR_STRUCT(input);
    Vector output = CONVERT_TO_VECTOR_STRUCT(output);

    DATA_TYPE_PROMOTED res = CONVERT(*((__global DATA_TYPE *)vector_offset(&input, 0)), DATA_TYPE_PROMOTED);

    // Convert input into F32 in order to perform quantized multiplication
#if defined(PROD) && defined(OFFSET) && defined(SCALE)
    float res_f = DEQUANTIZE(res, OFFSET, SCALE, DATA_TYPE_PROMOTED, 1);
#endif // defined(PROD) && defined(OFFSET) && defined(SCALE)

    for(unsigned int x = 1; x < WIDTH; ++x)
    {
        DATA_TYPE_PROMOTED in = CONVERT(*((__global DATA_TYPE *)vector_offset(&input, x)), DATA_TYPE_PROMOTED);
#if defined(MIN)
        res = select(res, in, ISLESS_SCALAR(in, res));
#elif defined(MAX)
        res = select(res, in, ISGREATER_SCALAR(in, res));
#elif defined(PROD)
#if defined(OFFSET) && defined(SCALE)
        res_f *= DEQUANTIZE(in, OFFSET, SCALE, DATA_TYPE_PROMOTED, 1);
#else  // !(defined(OFFSET) && defined(SCALE))
        res *= in;
#endif //  defined(OFFSET) && defined(SCALE)
#else  // defined(SUM))
        res += in;
#endif // defined(MAX) || defined(MIN) || defined(PROD)
    }

    // Store result
#if defined(MEAN)
    res /= WIDTH;
#endif // defined(MEAN)

    // Subtract the offsets in case of quantized SUM
#if defined(SUM) && defined(OFFSET) && defined(SCALE)
    res -= (WIDTH - 1) * OFFSET;
#endif // defined(OFFSET) && defined(OFFSET) && defined(SCALE)

    // Re-quantize
#if defined(PROD) && defined(OFFSET) && defined(SCALE)
    res = QUANTIZE(res_f, OFFSET, SCALE, DATA_TYPE_PROMOTED, 1);
#endif // defined(PROD) && defined(OFFSET) && defined(SCALE)

    *((__global DATA_TYPE *)output.ptr) = CONVERT_SAT(res, DATA_TYPE);
}
#endif // defined(WIDTH)

#if defined(HEIGHT)
/** This kernel performs reduction on y-axis.
 *
 * @note The input data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The height size must be passed at compile time using -DHEIGHT e.g. -DHEIGHT=128
 *
 * @param[in] input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/S32/F16/F32
 * @param[in] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p input_ptr
 * @param[in] output_stride_x                      Stride of the output tensor in X dimension (in bytes)
 * @param[in] output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_stride_y                      Stride of the output tensor in Y dimension (in bytes)
 * @param[in] output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] output_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void reduction_operation_y(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    int x = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    int y = get_global_id(1);

    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * input_stride_y;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * output_stride_y;

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE)
    res = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE));

    // Convert input into F32 in order to perform quantized multiplication
#if defined(PROD) && defined(OFFSET) && defined(SCALE)
    VEC_DATA_TYPE(float, VEC_SIZE)
    res_f = DEQUANTIZE(res, OFFSET, SCALE, DATA_TYPE_PROMOTED, VEC_SIZE);
#endif // defined(PROD) && defined(OFFSET) && defined(SCALE)

#if defined(SUM_SQUARE)
    res *= res;
#endif // defined(SUM_SQUARE)

    for(unsigned int y = 1; y < HEIGHT; ++y)
    {
        VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE)
        in = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + y * input_stride_y)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE));
#if defined(MIN)
        res = select(res, in, ISLESS(in, res));
#elif defined(MAX)
        res = select(res, in, ISGREATER(in, res));
#else // !(defined(MAX) || defined(MIN))
#if defined(SUM_SQUARE)
        in *= in;
#endif // defined(SUM_SQUARE)
#if defined(PROD)

#if defined(OFFSET) && defined(SCALE)
        res_f *= DEQUANTIZE(in, OFFSET, SCALE, DATA_TYPE_PROMOTED, VEC_SIZE);
#else  // !(defined(OFFSET) && defined(SCALE))
        res *= in;
#endif //  defined(OFFSET) && defined(SCALE)

#else  // !defined(PROD)
        res += in;
#endif // defined(PROD)
#endif // defined(MAX) || defined(MIN)
    }

#if defined(MEAN)
    res /= HEIGHT;
#endif // defined(MEAN)

    // Subtract the offsets in case of quantized SUM
#if defined(SUM) && defined(OFFSET) && defined(SCALE)
    res -= (HEIGHT - 1) * OFFSET;
#endif // defined(OFFSET) && defined(OFFSET) && defined(SCALE)

    // Re-quantize
#if defined(PROD) && defined(OFFSET) && defined(SCALE)
    res = QUANTIZE(res_f, OFFSET, SCALE, DATA_TYPE_PROMOTED, VEC_SIZE);
#endif // defined(PROD) && defined(OFFSET) && defined(SCALE)

    // Store result
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    res0 = CONVERT_SAT(res, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));
    STORE_VECTOR_SELECT(res, DATA_TYPE, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif // defined(HEIGHT)

#if defined(DEPTH)
/** This kernel performs reduction on z-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
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
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p input_ptr
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
    int x = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * input_stride_y + z * input_stride_z;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * output_stride_y + z * output_stride_z;

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE)
    res = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE));

    // Convert input into F32 in order to perform quantized multiplication
#if defined(PROD) && defined(OFFSET) && defined(SCALE)
    VEC_DATA_TYPE(float, VEC_SIZE)
    res_f = DEQUANTIZE(res, OFFSET, SCALE, DATA_TYPE_PROMOTED, VEC_SIZE);
#endif // defined(PROD) && defined(OFFSET) && defined(SCALE)

#if defined(SUM_SQUARE)
    res *= res;
#endif // defined(SUM_SQUARE)

    for(unsigned int z = 1; z < DEPTH; ++z)
    {
        VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE)
        in = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + z * input_stride_z)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE));

#if defined(MIN)
        res = select(res, in, ISLESS(in, res));
#elif defined(MAX)
        res = select(res, in, ISGREATER(in, res));
#else // !(defined(MAX) || defined(MIN))
#if defined(SUM_SQUARE)
        in *= in;
#endif // defined(SUM_SQUARE)
#if defined(PROD)

#if defined(OFFSET) && defined(SCALE)
        res_f *= DEQUANTIZE(in, OFFSET, SCALE, DATA_TYPE_PROMOTED, VEC_SIZE);
#else  // !(defined(OFFSET) && defined(SCALE))
        res *= in;
#endif //  defined(OFFSET) && defined(SCALE)

#else  // !defined(PROD)
        res += in;
#endif // defined(PROD)
#endif // defined(MAX) || defined(MIN)
    }

#if defined(MEAN)
    res /= DEPTH;
#endif // defined(MEAN)

    // Subtract the offsets in case of quantized SUM
#if defined(SUM) && defined(OFFSET) && defined(SCALE)
    res -= (DEPTH - 1) * OFFSET;
#endif // defined(OFFSET) && defined(OFFSET) && defined(SCALE)

    // Re-quantize
#if defined(PROD) && defined(OFFSET) && defined(SCALE)
    res = QUANTIZE(res_f, OFFSET, SCALE, DATA_TYPE_PROMOTED, VEC_SIZE);
#endif // defined(PROD) && defined(OFFSET) && defined(SCALE)

    // Store result
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    res0 = CONVERT_SAT(res, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));

    STORE_VECTOR_SELECT(res, DATA_TYPE, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif /* defined(DEPTH) */

#if defined(BATCH) && defined(DEPTH)
/** This kernel performs reduction on w-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
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
 * @param[in] output_ptr                           The local buffer to hold sumed values. Supported data types: same as @p input_ptr
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
    int x = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * input_stride_y + (z % DEPTH) * input_stride_z + (z / DEPTH) * input_stride_w;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * output_stride_y + (z % DEPTH) * output_stride_z + (z / DEPTH) * output_stride_z;

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE)
    res = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE));

    // Convert input into F32 in order to perform quantized multiplication
#if defined(PROD) && defined(OFFSET) && defined(SCALE)
    VEC_DATA_TYPE(float, VEC_SIZE)
    res_f = DEQUANTIZE(res, OFFSET, SCALE, DATA_TYPE_PROMOTED, VEC_SIZE);
#endif // defined(PROD) && defined(OFFSET) && defined(SCALE)

#if defined(SUM_SQUARE)
    res *= res;
#endif // defined(SUM_SQUARE)

    for(unsigned int w = 1; w < BATCH; ++w)
    {
        VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE)
        in = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + w * input_stride_w)), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, VEC_SIZE));

#if defined(MIN)
        res = select(res, in, ISLESS(in, res));
#elif defined(MAX)
        res = select(res, in, ISGREATER(in, res));
#else // !(defined(MAX) || defined(MIN))
#if defined(SUM_SQUARE)
        in *= in;
#endif // defined(SUM_SQUARE)
#if defined(PROD)

#if defined(OFFSET) && defined(SCALE)
        res_f *= DEQUANTIZE(in, OFFSET, SCALE, DATA_TYPE_PROMOTED, VEC_SIZE);
#else  // !(defined(OFFSET) && defined(SCALE))
        res *= in;
#endif //  defined(OFFSET) && defined(SCALE)

#else  // !defined(PROD)
        res += in;
#endif //defined(PROD)
#endif // defined(MAX) || defined(MIN)
    }

#if defined(MEAN)
    res /= BATCH;
#endif // defined(MEAN)

    // Subtract the offsets in case of quantized SUM
#if defined(SUM) && defined(OFFSET) && defined(SCALE)
    res -= (BATCH - 1) * OFFSET;
#endif // defined(OFFSET) && defined(OFFSET) && defined(SCALE)

    // Re-quantize
#if defined(PROD) && defined(OFFSET) && defined(SCALE)
    res = QUANTIZE(res_f, OFFSET, SCALE, DATA_TYPE_PROMOTED, VEC_SIZE);
#endif // defined(PROD) && defined(OFFSET) && defined(SCALE)

    // Store result
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    res0 = CONVERT_SAT(res, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));
    STORE_VECTOR_SELECT(res, DATA_TYPE, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif /* defined(BATCH) && defined(DEPTH) */
