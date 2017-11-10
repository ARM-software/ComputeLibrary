/*
 * Copyright (c) 2017 ARM Limited.
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

layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;

#include "helpers_cs.h"

#if defined(DATA_TYPE_FP16)
precision mediump float;
#endif // DATA_TYPE_FP16

// Common definitions
#define MAX_OP(x, y) max((x), (y))
#define ADD_OP(x, y) ((x) + (y))
#define SUB_OP(x, y) ((x) - (y))
#define DIV_OP(x, y) ((x) / (y))
#define EXP_OP(x) exp((x))

const float float_min = -1.0 / 0.0;
const vec4  vec4_min  = vec4(float_min);

#ifdef SOFTMAX_LAYER_MAX

/** Identifies the maximum value across the 1st dimension.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note In case the input is not multiple of 4 NON_MULTIPLE_OF_4 must be passed.
 *
 * @param[in]  src_ptr   Pointer to the source tensor slice. Supported data types: F16/F32
 * @param[in]  src_attrs The attributes of the source tensor
 * @param[out] dst_ptr   Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination tensor
 * @param[in]  width     Input image width
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
    uint               width;
};

#if defined(DATA_TYPE_FP32)

TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    ImageIterator src_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    // Initialize local maximum
    vec4 max_val = vec4_min;

    // Calculate max of row
    uint width2 = width >> 2;
    for(int i = 0; i < int(width2); i++)
    {
        vec4 data = VLOAD4(vec4, src_ptr, IMAGE_OFFSET(src_iter, i << 2, 0));
        max_val   = MAX_OP(data, max_val);
    }

#ifdef NON_MULTIPLE_OF_4
    // Handle non multiple of 4
    for(int i = int(width2 << 2); i < int(width); i++)
    {
        float data = LOAD(src_ptr, IMAGE_OFFSET(src_iter, i, 0));
        max_val.x  = MAX_OP(data, max_val.x);
    }
#endif /* NON_MULTIPLE_OF_4 */

    // Perform max reduction
    max_val.xy = MAX_OP(max_val.xy, max_val.zw);
    max_val.x  = MAX_OP(max_val.x, max_val.y);

    // Store result
    STORE_CURRENT_ITEM(dst_ptr, dst_iter, max_val.x);
}
#elif defined(DATA_TYPE_FP16)

TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    ImageIterator src_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);

    // Initialize local maximum
    vec4 max_val = vec4_min;

    // Calculate max of row
    uint width2 = width >> 2;
    for(int i = 0; i < int(width2); i++)
    {
        vec4 data = VLOAD2_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, i << 2, 0));
        max_val   = MAX_OP(data, max_val);
    }

#ifdef NON_MULTIPLE_OF_4
    // Handle non multiple of 4
    for(int i = int(width2 << 2); i < int(width); i = i + 2)
    {
        vec2 data = LOAD_UNPACK2_HALF(src_ptr, IMAGE_OFFSET(src_iter, i, 0));
        max_val.x = MAX_OP(data.x, max_val.x);
        if((i + 1) < int(width))
        {
            max_val.x = MAX_OP(data.y, max_val.x);
        }
    }
#endif /* NON_MULTIPLE_OF_4 */

    // Perform max reduction
    max_val.xy = MAX_OP(max_val.xy, max_val.zw);
    max_val.x  = MAX_OP(max_val.x, max_val.y);

    STORE_PACK2_CURRENT_ITEM_HALF(dst_ptr, dst_iter, max_val.xy);
}
#else  // DATA_TYPE_FP32
#error Data type not supported
#endif // DATA_TYPE_FP32
#elif defined(SOFTMAX_LAYER_SHIFT_EXP_SUM)

/** Shifts the values of the input tensor by the max calculated in softmax_layer_max kernel,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note In case the input is not multiple of 4 NON_MULTIPLE_OF_4 must be passed.
 *
 * @param[in]  src_ptr   Pointer to the source tensor slice. Supported data types: F16/F32
 * @param[in]  src_attrs The attributes of the source tensor
 * @param[in]  max_ptr   Pointer to the max values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  max_attrs The attributes of the max values tensor
 * @param[out] dst_ptr   Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination tensor
 * @param[out] sum_ptr   Pointer to the sum values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  sum_attrs The attributes of the sum values tensor
 * @param[in]  width     Input image width
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes max_attrs;
    Tensor3DAttributes dst_attrs;
    Tensor3DAttributes sum_attrs;
    uint               width;
};
#if defined(DATA_TYPE_FP32)

TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, maxBuffer, float, max_ptr, max_shift, 2, readonly);
TENSOR_DECLARATION(3, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);
TENSOR_DECLARATION(4, sumBuffer, float, sum_ptr, sum_shift, 2, writeonly);

void main(void)
{
    ImageIterator src_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);
    ImageIterator max_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(max_attrs, max_shift);
    ImageIterator sum_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(sum_attrs, sum_shift);

    // Load max value of 1D logits vector (row)
    vec4 max_val = vec4(LOAD_CURRENT_ITEM(max_ptr, max_iter));

    // Set sum vector
    vec4 sum1D = vec4(0);

    // Shift values, exp and sum
    uint width2 = width >> 2;
    for(int i = 0; i < int(width2); i++)
    {
        vec4 data = VLOAD4(vec4, src_ptr, IMAGE_OFFSET(src_iter, i << 2, 0));
        data      = SUB_OP(data, max_val);
        data      = EXP_OP(data);
        VSTORE4(dst_ptr, IMAGE_OFFSET(dst_iter, i << 2, 0), data);
        sum1D = ADD_OP(sum1D, data);
    }

#ifdef NON_MULTIPLE_OF_4
    // Handle non multiple of 4
    for(int i = int(width2 << 2); i < int(width); i++)
    {
        float data = LOAD(src_ptr, IMAGE_OFFSET(src_iter, i, 0));
        data       = SUB_OP(data, max_val.x);
        data       = EXP_OP(data);
        STORE(dst_ptr, IMAGE_OFFSET(dst_iter, i, 0), data);
        sum1D.x = ADD_OP(sum1D.x, data);
    }
#endif /* NON_MULTIPLE_OF_4 */

    // Perform min/max reduction
    sum1D.xy = ADD_OP(sum1D.xy, sum1D.zw);
    sum1D.x  = ADD_OP(sum1D.x, sum1D.y);

    // Calculate and store result
    STORE_CURRENT_ITEM(sum_ptr, sum_iter, sum1D.x);
}
#elif defined(DATA_TYPE_FP16)

TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, maxBuffer, uint, max_ptr, max_shift, 2, readonly);
TENSOR_DECLARATION(3, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);
TENSOR_DECLARATION(4, sumBuffer, uint, sum_ptr, sum_shift, 2, writeonly);

void main(void)
{
    ImageIterator src_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);
    ImageIterator max_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(max_attrs, max_shift);
    ImageIterator sum_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(sum_attrs, sum_shift);

    // Load max value of 1D logits vector (row)
    vec2 datamaxinit = LOAD_UNPACK2_CURRENT_ITEM_HALF(max_ptr, max_iter);
    vec4 max_val     = vec4(datamaxinit.x);

    // Set sum vector
    vec4 sum1D = vec4(0.f);

    // Shift values, exp and sum
    uint width2 = width >> 2;
    for(int i = 0; i < int(width2); i++)
    {
        vec4 data = VLOAD2_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, i << 2, 0));
        data      = SUB_OP(data, max_val);
        data      = EXP_OP(data);
        VSTORE2_PACK4_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, i << 2, 0), data);
        sum1D = ADD_OP(sum1D, data);
    }

#ifdef NON_MULTIPLE_OF_4
    // Handle non multiple of 4
    for(int i = int(width2 << 2); i < int(width); i = i + 2)
    {
        float data;
        vec2  datamiddle = LOAD_UNPACK2_HALF(src_ptr, IMAGE_OFFSET(src_iter, i, 0));
        data             = SUB_OP(datamiddle.x, max_val.x);
        data             = EXP_OP(data);
        vec2 datares;
        if((i + 1) < int(width))
        {
            float data2;
            data2   = SUB_OP(datamiddle.y, max_val.x);
            data2   = EXP_OP(data2);
            datares = vec2(data, data2);
            data    = ADD_OP(data2, data);
        }
        else
        {
            datares = vec2(data, 0.f);
        }

        STORE_PACK2_HALF(dst_ptr, IMAGE_OFFSET(dst_iter, i, 0), datares);

        sum1D.x = ADD_OP(sum1D.x, data);
    }
#endif /* NON_MULTIPLE_OF_4 */

    // Perform min/max reduction
    sum1D.xy = ADD_OP(sum1D.xy, sum1D.zw);
    sum1D.x  = ADD_OP(sum1D.x, sum1D.y);

    // Calculate and store result
    STORE_PACK2_CURRENT_ITEM_HALF(sum_ptr, sum_iter, sum1D.xy);
}
#else  // DATA_TYPE_FP32
#error Data type not supported
#endif // DATA_TYPE_FP32
#elif defined(SOFTMAX_LAYER_NORM)

/** Divides all the values of the input tensor by the sum calculated from softmax_layer_shift_exp_sum kernel.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 *
 * @param[in]  src_ptr   Pointer to the source tensor slice. Supported data types: F16/F32
 * @param[in]  src_attrs The attributes of the source tensor
 * @param[in]  sum_ptr   Pointer to the sum values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  sum_attrs The attributes of the sum values tensor
 * @param[out] dst_ptr   Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination tensor
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes sum_attrs;
    Tensor3DAttributes dst_attrs;
};
#if defined(DATA_TYPE_FP32)
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, sumBuffer, float, sum_ptr, sum_shift, 2, readonly);
TENSOR_DECLARATION(3, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);
void main(void)
{
    ImageIterator src_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);
    ImageIterator sum_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR_NO_STEP(sum_attrs, sum_shift);

    // Load max value of 1D logits vector (row)
    vec4 sum_val = vec4(LOAD(sum_ptr, IMAGE_OFFSET(sum_iter, 0, gl_GlobalInvocationID.y)));
    vec4 data    = VLOAD4_CURRENT_ITEM(vec4, src_ptr, src_iter);
    VSTORE4_CURRENT_ITEM(dst_ptr, dst_iter, DIV_OP(data, sum_val));
}
#elif defined(DATA_TYPE_FP16)
TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, sumBuffer, uint, sum_ptr, sum_shift, 2, readonly);
TENSOR_DECLARATION(3, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);
void main(void)
{
    ImageIterator src_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    ImageIterator dst_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(dst_attrs, dst_shift);
    ImageIterator sum_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR_NO_STEP(sum_attrs, sum_shift);

    // Load max value of 1D logits vector (row)
    vec4 sum_val = vec4(LOAD_UNPACK2_HALF(sum_ptr, IMAGE_OFFSET(sum_iter, 0, gl_GlobalInvocationID.y)).x);
    vec4 data    = VLOAD2_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    VSTORE2_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, DIV_OP(data, sum_val));
}
#else // DATA_TYPE_FP32
#error Data type not supported
#endif // DATA_TYPE_FP32
#endif // SOFTMAX_LAYER_MAX
