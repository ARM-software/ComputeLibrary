/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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

#define MIN_VALUE_float -FLT_MAX
#define MIN_VALUE_half  -HALF_MAX
#define MIN_VALUE_char  CHAR_MIN
#define MIN_VALUE_uchar 0

#define MIN_VALUE_TYPE_STR(data_type) MIN_VALUE_##data_type
#define MIN_VALUE_TYPE(data_type) MIN_VALUE_TYPE_STR(data_type)
#define MIN_VALUE MIN_VALUE_TYPE(DATA_TYPE)

#ifdef SOFTMAX_X

/** 3-pass softmax in the x dimension.
 *
 * List of preprocessors:
 *   - DATA_TYPE: the input/output data type.
 *   - TMP_DATA_TYPE: the data type used for computing and temporary tensor storage.
 *     If DATA_TYPE is quantized, TMP_DATA_TYPE is floating-point, otherwise TMP_DATA_TYPE is the same as DATA_TYPE.
 *   - IS_LOG (optional): indicating whether this is log softmax.
 *   - LENGTH: the number of elements in softmax axis in the input/output tensors.
 *   - BETA: the beta coefficient.
 *   - IS_QUANTIZED (optional): indicating whether the input/output data type is quantized data.
 *   - VEC_SIZE: the size of the vector.
 *
 * Additional preprocessors in case IS_QUANTIZED is present:
 *   - SRC_SCALE and SRC_OFFSET: the quantization information of the source tensor.
 *   - DST_SCALE and DST_OFFSET: the quantization information of the destination tensor.
 *
 * @param[in] src_ptr                  Pointer to the source tensor.
 * @param[in] src_stride_0             Stride in bytes of the source tensor in the dimension corresponding to global ID 0.
 * @param[in] src_stride_1             Stride in bytes of the source tensor in the dimension corresponding to global ID 1.
 * @param[in] src_stride_2             Stride in bytes of the source tensor in the dimension corresponding to global ID 2.
 * @param[in] src_offset_first_element Offset of the first element in the source tensor.
 * @param[in] dst_ptr                  Pointer to the destination tensor.
 * @param[in] dst_stride_0             Stride in bytes of the destination tensor in the dimension corresponding to global ID 0.
 * @param[in] dst_stride_1             Stride in bytes of the destination tensor in the dimension corresponding to global ID 1.
 * @param[in] dst_stride_2             Stride in bytes of the destination tensor in the dimension corresponding to global ID 2.
 * @param[in] dst_offset_first_element Offset of the first element in the destination tensor.
 * @param[in] tmp_ptr                  Pointer to the temporary tensor.
 * @param[in] tmp_stride_0             Stride in bytes of the temporary tensor in the dimension corresponding to global ID 0.
 * @param[in] tmp_stride_1             Stride in bytes of the temporary tensor in the dimension corresponding to global ID 1.
 * @param[in] tmp_stride_2             Stride in bytes of the temporary tensor in the dimension corresponding to global ID 2.
 * @param[in] tmp_offset_first_element Offset of the first element in the temporary tensor.
 */
__kernel void softmax_x(
    __global uchar *src_ptr,
    uint src_stride_0,
    uint src_stride_1,
    uint src_stride_2,
    uint src_offset_first_element,

    __global uchar *dst_ptr,
    uint dst_stride_0,
    uint dst_stride_1,
    uint dst_stride_2,
    uint dst_offset_first_element

#ifdef IS_QUANTIZED
    ,
    __global uchar *tmp_ptr,
    uint tmp_stride_0,
    uint tmp_stride_1,
    uint tmp_stride_2,
    uint tmp_offset_first_element
#endif // IS_QUANTIZED
)
{
    const int dim_0 = get_global_id(0);
    const int dim_1 = get_global_id(1);
    const int dim_2 = get_global_id(2);

    src_ptr += src_offset_first_element + dim_2 * src_stride_2 + dim_1 * src_stride_1 + dim_0 * src_stride_0;
    dst_ptr += dst_offset_first_element + dim_2 * dst_stride_2 + dim_1 * dst_stride_1 + dim_0 * dst_stride_0;

#ifdef IS_QUANTIZED
    tmp_ptr += tmp_offset_first_element + dim_2 * tmp_stride_2 + dim_1 * tmp_stride_1 + dim_0 * tmp_stride_0;
#else // IS_QUANTIZED
    __global uchar *tmp_ptr = dst_ptr;
#endif // IS_QUANTIZED

    // Calculate max value.
    DATA_TYPE max_value = MIN_VALUE;
    int i = 0;

    for (i = 0; i < LENGTH - VEC_SIZE; i += VEC_SIZE)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_ptr + i * sizeof(DATA_TYPE)));

        max_value = max(max_value, MAX_REDUCE(data, VEC_SIZE));
    }

    for (; i < LENGTH; ++i)
    {
        DATA_TYPE data = *(__global DATA_TYPE *)(src_ptr + i * sizeof(DATA_TYPE));

        max_value = max(max_value, data);
    }

    // Regularize the data.
    TMP_DATA_TYPE sum_value = 0;

#ifdef IS_QUANTIZED
    TMP_DATA_TYPE max_value_f = (CONVERT(max_value, TMP_DATA_TYPE) - SRC_OFFSET) * SRC_SCALE;
    TMP_DATA_TYPE regularize_offset = -SRC_OFFSET * SRC_SCALE * (TMP_DATA_TYPE)BETA - max_value_f * (TMP_DATA_TYPE)BETA;
# define REGULARIZE(x) ((x) * SRC_SCALE * (TMP_DATA_TYPE)BETA + regularize_offset)
#else // IS_QUANTIZED
# define REGULARIZE(x) (((x) - max_value) * (TMP_DATA_TYPE)BETA)
#endif // IS_QUANTIZED

    for (i = 0; i < LENGTH - VEC_SIZE; i += VEC_SIZE)
    {
        VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE) data = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_ptr + i * sizeof(DATA_TYPE))), VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE));

        data = REGULARIZE(data);

#ifdef IS_LOG
        sum_value += SUM_REDUCE(exp(data), VEC_SIZE);
#else // IS_LOG
        data = exp(data);
        sum_value += SUM_REDUCE(data, VEC_SIZE);
#endif // IS_LOG

        VSTORE(VEC_SIZE)(data, 0, (__global TMP_DATA_TYPE *)(tmp_ptr + i * sizeof(TMP_DATA_TYPE)));
    }

    for (; i < LENGTH; ++i)
    {
        TMP_DATA_TYPE data = CONVERT(*(__global DATA_TYPE *)(src_ptr + i * sizeof(DATA_TYPE)), TMP_DATA_TYPE);

        data = REGULARIZE(data);

#ifdef IS_LOG
        sum_value += exp(data);
#else // IS_LOG
        data = exp(data);
        sum_value += data;
#endif // IS_LOG

        *(__global TMP_DATA_TYPE *)(tmp_ptr + i * sizeof(TMP_DATA_TYPE)) = data;
    }

#undef REGULARIZE

    // Normalize the data.
#ifdef IS_QUANTIZED
# if IS_LOG
    TMP_DATA_TYPE norm_offset = -log(sum_value) + DST_OFFSET;
#  define NORMALIZE(SIZE, x) CONVERT_SAT_ROUND((x) / DST_SCALE + norm_offset, VEC_DATA_TYPE(DATA_TYPE, SIZE), rte)
# else // IS_LOG
    TMP_DATA_TYPE norm_div = sum_value * DST_SCALE;
#  define NORMALIZE(SIZE, x) CONVERT_SAT(add_sat(CONVERT_SAT_ROUND((x) / norm_div, VEC_DATA_TYPE(int, SIZE), rte), DST_OFFSET), VEC_DATA_TYPE(DATA_TYPE, SIZE))
#  endif // IS_LOG
#else // IS_QUANTIZED
# if IS_LOG
#  define NORMALIZE(SIZE, x) ((x) - log(sum_value))
# else // IS_LOG
#  define NORMALIZE(SIZE, x) ((x) / sum_value)
# endif // IS_LOG
#endif // IS_QUANTIZED

    for (i = 0; i < LENGTH - VEC_SIZE; i += VEC_SIZE)
    {
        VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE) data = VLOAD(VEC_SIZE)(0, (__global TMP_DATA_TYPE *)(tmp_ptr + i * sizeof(TMP_DATA_TYPE)));

        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) result = NORMALIZE(VEC_SIZE, data);

        VSTORE(VEC_SIZE)(result, 0, (__global DATA_TYPE *)(dst_ptr + i * sizeof(DATA_TYPE)));
    }

    for (; i < LENGTH; ++i)
    {
        TMP_DATA_TYPE data = *(__global TMP_DATA_TYPE *)(tmp_ptr + i * sizeof(TMP_DATA_TYPE));

        DATA_TYPE result = NORMALIZE(1, data);

        *(__global DATA_TYPE *)(dst_ptr + i * sizeof(DATA_TYPE)) = result;
    }

#undef NORMALIZE
}

#endif // SOFTMAX_X

#ifdef SOFTMAX_NON_X

/** 3-pass softmax in any dimension higher than the x dimension.
 *
 * List of preprocessors:
 *   - DATA_TYPE: the input/output data type.
 *   - TMP_DATA_TYPE: the data type used for computing and temporary tensor storage.
 *     If DATA_TYPE is quantized, TMP_DATA_TYPE is floating-point, otherwise TMP_DATA_TYPE is the same as DATA_TYPE.
 *   - IS_LOG (optional): indicating whether this is log softmax.
 *   - LENGTH: the number of elements in softmax axis in the input/output tensors.
 *   - BETA: the beta coefficient.
 *   - IS_QUANTIZED (optional): indicating whether the input/output data type is quantized data.
 *   - VEC_SIZE: the size of the vector.
 *   - VEC_SIZE_LEFTOVER: the size of the leftover part.
 *
 * Additional preprocessors in case IS_QUANTIZED is present:
 *   - SRC_SCALE and SRC_OFFSET: the quantization information of the source tensor.
 *   - DST_SCALE and DST_OFFSET: the quantization information of the destination tensor.
 *
 * @param[in] src_ptr                  Pointer to the source tensor.
 * @param[in] src_stride_0             Stride in bytes of the source tensor in the dimension corresponding to global ID 0.
 * @param[in] src_stride_1             Stride in bytes of the source tensor in the dimension corresponding to global ID 1.
 * @param[in] src_stride_2             Stride in bytes of the source tensor in the dimension corresponding to global ID 2.
 * @param[in] src_offset_first_element Offset of the first element in the source tensor.
 * @param[in] dst_ptr                  Pointer to the destination tensor.
 * @param[in] dst_stride_0             Stride in bytes of the destination tensor in the dimension corresponding to global ID 0.
 * @param[in] dst_stride_1             Stride in bytes of the destination tensor in the dimension corresponding to global ID 1.
 * @param[in] dst_stride_2             Stride in bytes of the destination tensor in the dimension corresponding to global ID 2.
 * @param[in] dst_offset_first_element Offset of the first element in the destination tensor.
 * @param[in] tmp_ptr                  Pointer to the temporary tensor.
 * @param[in] tmp_stride_0             Stride in bytes of the temporary tensor in the dimension corresponding to global ID 0.
 * @param[in] tmp_stride_1             Stride in bytes of the temporary tensor in the dimension corresponding to global ID 1.
 * @param[in] tmp_stride_2             Stride in bytes of the temporary tensor in the dimension corresponding to global ID 2.
 * @param[in] tmp_offset_first_element Offset of the first element in the temporary tensor.
 */
__kernel void softmax_non_x(
    __global uchar *src_ptr,
    uint src_stride_0,
    uint src_stride_1,
    uint src_stride_2,
    uint src_offset_first_element,

    __global uchar *dst_ptr,
    uint dst_stride_0,
    uint dst_stride_1,
    uint dst_stride_2,
    uint dst_offset_first_element,

    __global uchar *tmp_ptr,
    uint tmp_stride_0,
    uint tmp_stride_1,
    uint tmp_stride_2,
    uint tmp_offset_first_element,

    uint src_stride_axis,
    uint dst_stride_axis
)
{
    const int dim_0 = max((int)get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE, 0);
    const int dim_1 = get_global_id(1);
    const int dim_2 = get_global_id(2);

    src_ptr += src_offset_first_element + dim_2 * src_stride_2 + dim_1 * src_stride_1 + dim_0 * src_stride_0;
    dst_ptr += dst_offset_first_element + dim_2 * dst_stride_2 + dim_1 * dst_stride_1 + dim_0 * dst_stride_0;
    tmp_ptr += tmp_offset_first_element + dim_2 * tmp_stride_2 + dim_1 * tmp_stride_1 + dim_0 * tmp_stride_0;

    // In case of processing quantized data, i.e. DATA_TYPE is smaller than TMP_DATA_TYPE:
    //
    // In the first pass (finding max), the quantized data is copied from the input tensor to the temporary tensor.
    // Dequantization is not needed to find the max value and since dequantization widens the data, we defer it
    // to the second pass pass to reduce memory bandwidth of the first pass.
    //
    // In the second pass, it reads the quantized data from the temporary tensor and writes the dequantized data
    // back to the temporary tensor.
    //
    // To avoid dequantized data overwritting the unprocessed quantized data in the temporary tensor,
    // this extra offset is introduced to store the quantized data at the end of the temporary tensor.
    //
    // Note: Another approach is to perform the second pass in reverse order, but for unexplanable reason
    // it doesn't work in some devices.
    uint tmp_extra_offset = LENGTH * VEC_SIZE * (sizeof(TMP_DATA_TYPE) - sizeof(DATA_TYPE));

    // Calculate max value and store the input data to the temporary tensor in suitable format.
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) max_value = MIN_VALUE;
    int i = 0;

    for (i = 0; i < LENGTH; ++i)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_ptr + i * src_stride_axis));

        max_value = max(max_value, data);

        VSTORE(VEC_SIZE)(data, 0, (__global DATA_TYPE *)(tmp_ptr + tmp_extra_offset + i * VEC_SIZE * sizeof(DATA_TYPE)));
    }

    // Regularize the data.
    VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE) sum_value = 0;

#ifdef IS_QUANTIZED
    VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE) max_value_f = (CONVERT(max_value, VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE)) - SRC_OFFSET) * SRC_SCALE;
    VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE) regularize_offset = -SRC_OFFSET * SRC_SCALE * (TMP_DATA_TYPE)BETA - max_value_f * (TMP_DATA_TYPE)BETA;
# define REGULARIZE(x) ((x) * SRC_SCALE * (TMP_DATA_TYPE)BETA + regularize_offset)
#else // IS_QUANTIZED
# define REGULARIZE(x) (((x) - max_value) * (TMP_DATA_TYPE)BETA)
#endif // IS_QUANTIZED

    for (i = 0; i < LENGTH; ++i)
    {
        VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE) data = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(tmp_ptr + tmp_extra_offset + i * VEC_SIZE * sizeof(DATA_TYPE))), VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE));

        data = REGULARIZE(data);

#ifdef IS_LOG
        sum_value += exp(data);
#else // IS_LOG
        data = exp(data);
        sum_value += data;
#endif // IS_LOG

        VSTORE(VEC_SIZE)(data, 0, (__global TMP_DATA_TYPE *)(tmp_ptr + i * VEC_SIZE * sizeof(TMP_DATA_TYPE)));
    }

#undef REGULARIZE

    // Normalize the data.
#ifdef IS_QUANTIZED
# if IS_LOG
    VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE) norm_offset = -log(sum_value) + DST_OFFSET;
#  define NORMALIZE(x) CONVERT_SAT_ROUND((x) / DST_SCALE + norm_offset, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE), rte)
# else // IS_LOG
    VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE) norm_div = sum_value * DST_SCALE;
#  define NORMALIZE(x) CONVERT_SAT(add_sat(CONVERT_SAT_ROUND((x) / norm_div, VEC_DATA_TYPE(int, VEC_SIZE), rte), DST_OFFSET), VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))
#  endif // IS_LOG
#else // IS_QUANTIZED
# if IS_LOG
#  define NORMALIZE(x) ((x) - log(sum_value))
# else // IS_LOG
#  define NORMALIZE(x) ((x) / sum_value)
# endif // IS_LOG
#endif // IS_QUANTIZED

    for (i = 0; i < LENGTH; ++i)
    {
        VEC_DATA_TYPE(TMP_DATA_TYPE, VEC_SIZE) data = VLOAD(VEC_SIZE)(0, (__global TMP_DATA_TYPE *)(tmp_ptr + i * VEC_SIZE * sizeof(TMP_DATA_TYPE)));

        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) result0 = NORMALIZE(data);

        STORE_VECTOR_SELECT(result, DATA_TYPE, dst_ptr + i * dst_stride_axis, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
    }

#undef NORMALIZE
}

#endif // SOFTMAX_NON_X

#undef MIN_VALUE
#undef MIN_VALUE_TYPE
#undef MIN_VALUE_TYPE_STR

#undef MIN_VALUE_float
#undef MIN_VALUE_half
#undef MIN_VALUE_char
#undef MIN_VALUE_uchar
