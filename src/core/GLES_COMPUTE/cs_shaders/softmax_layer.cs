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

#include "helpers.h"

#define MAX_OP(x, y) max((x), (y))
#define ADD_OP(x, y) ((x) + (y))
#define SUB_OP(x, y) ((x) - (y))
#define DIV_OP(x, y) ((x) / (y))
#define EXP_OP(x) exp((x))

#if defined(DATA_TYPE_FP32)
const float MINVAL   = -1.0 / 0.0;
vec4        type_min = CONVERT(MINVAL, vec4);

#define LOAD16(name, offset)            \
    vec4(LOAD4(name, offset),           \
         LOAD4(name, offset + uint(1)), \
         LOAD4(name, offset + uint(2)), \
         LOAD4(name, offset + uint(3)))

#define STORE16(name, offset, value)         \
    STORE4(name, offset, value.x);           \
    STORE4(name, offset + uint(1), value.y); \
    STORE4(name, offset + uint(2), value.z); \
    STORE4(name, offset + uint(3), value.w)

#ifdef SOFTMAX_LAYER_MAX
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);
#elif defined(SOFTMAX_LAYER_SHIFT_EXP_SUM)
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(max, 2, float, readonly);
BUFFER_DECLARATION(dst, 3, float, writeonly);
BUFFER_DECLARATION(sum, 4, float, writeonly);
#elif defined(SOFTMAX_LAYER_NORM)
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(sum, 2, float, readonly);
BUFFER_DECLARATION(dst, 3, float, writeonly);
#endif // SOFTMAX_LAYER_MAX

layout(std140) uniform shader_params
{
#ifdef SOFTMAX_LAYER_MAX
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(dst);
    uint width;
#elif defined(SOFTMAX_LAYER_SHIFT_EXP_SUM)
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(max);
    TENSOR3D_PARAM_DECLARATION(dst);
    TENSOR3D_PARAM_DECLARATION(sum);
    uint width;
#elif defined(SOFTMAX_LAYER_NORM)
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(sum);
    TENSOR3D_PARAM_DECLARATION(dst);
#endif // SOFTMAX_LAYER_MAX
};

#ifdef SOFTMAX_LAYER_MAX
/** Identifies the maximum value across the 1st dimension.
 *
 * @note Datatype must be given as a preprocessor argument using "#define DATA_TYPE_FP32"
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  width                             Input image width
 */
void main(void)
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);

    // Initialize local maximum
    vec4 max_val = CONVERT(type_min, vec4);

    // Calculate max of row
    uint width2 = width >> 2;
    for(int i = 0; i < int(width2); i++)
    {
        vec4 data = LOAD16(src, offset(src, i << 2, 0));
        max_val   = MAX_OP(data, max_val);
    }

#ifdef NON_MULTIPLE_OF_4
    // Handle non multiple of 4
    for(int i = int(width2 << 2); i < int(width); i++)
    {
        float data = LOAD4(src, offset(src, i, 0));
        max_val.x  = MAX_OP(data, max_val.x);
    }
#endif /* NON_MULTIPLE_OF_4 */

    // Perform max reduction
    max_val.xy = MAX_OP(max_val.xy, max_val.zw);
    max_val.x  = MAX_OP(max_val.x, max_val.y);

    // Store result
    STORE4(dst, CURRENT_OFFSET(dst), max_val.x);
}
#elif defined(SOFTMAX_LAYER_SHIFT_EXP_SUM) // SOFTMAX_LAYER_MAX
/** Shifts the values of the input tensor by the max calculated in softmax_layer_max kernel,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note Datatype must be given as a preprocessor argument using "#define DATA_TYPE_FP32"
 *
 * @note In case the input is not multiple of 4 NON_MULTIPLE_OF_4 must be passed.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: F32
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
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[out] sum_ptr                           Pointer to the sum values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  sum_stride_x                      Stride of the sum values tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the sum values tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_stride_z                      Stride of the sum values tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the sum values tensor
 * @param[in]  width                             Input image width
 */
void main(void)
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image max = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(max);
    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

    // Load max value of 1D logits vector (row)
    vec4 max_val = CONVERT(LOAD4(max, CURRENT_OFFSET(max)), vec4);

    // Set sum vector
    vec4 sum1D = CONVERT(0, vec4);

    // Shift values, exp and sum
    uint width2 = width >> 2;
    for(int i = 0; i < int(width2); i++)
    {
        vec4 data = LOAD16(src, offset(src, i << 2, 0));
        data      = SUB_OP(data, max_val);
        data      = EXP_OP(data);
        STORE16(dst, offset(dst, i << 2, 0), data);
        sum1D = ADD_OP(sum1D, data);
    }

#ifdef NON_MULTIPLE_OF_4
    // Handle non multiple of 4
    for(int i = int(width2 << 2); i < int(width); i++)
    {
        float data;
        data = LOAD4(src, offset(src, i, 0));
        data = SUB_OP(data, max_val.x);
        data = EXP_OP(data);
        STORE4(dst, offset(dst, i, 0), data);
        sum1D.x = ADD_OP(sum1D.x, data);
    }
#endif                            /* NON_MULTIPLE_OF_4 */

    // Perform min/max reduction
    sum1D.xy = ADD_OP(sum1D.xy, sum1D.zw);
    sum1D.x  = ADD_OP(sum1D.x, sum1D.y);

    // Calculate and store result
    STORE4(sum, CURRENT_OFFSET(sum), sum1D.x);
}
#elif defined(SOFTMAX_LAYER_NORM) // SOFTMAX_LAYER_MAX
/** Divides all the values of the input tensor by the sum calculated from softmax_layer_shift_exp_sum kernel.
 *
 * @note Datatype must be given as a preprocessor argument using "#define DATA_TYPE_FP32"
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: F32
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
void main(void)
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(sum);

    // Load max value of 1D logits vector (row)
    vec4 sum_val = CONVERT(LOAD4(sum, offset(sum, 0, int(gl_GlobalInvocationID.y))), vec4);
    vec4 data    = LOAD16(src, CURRENT_OFFSET(src));
    STORE16(dst, CURRENT_OFFSET(dst), DIV_OP(data, sum_val));
}
#endif                            // SOFTMAX_LAYER_MAX

#elif defined(DATA_TYPE_FP16)
precision mediump float;

const float MINVAL1   = -1.0 / 0.0;
vec4        type_min1 = CONVERT(MINVAL1, vec4);

#define GC_LOAD4_IMAGE(r, name, x, y)  \
    load_and_unpack(r.xy, name, x, y); \
    load_and_unpack(r.zw, name, (x + 2), y)

#define GC_STORE4_IMAGE(r, name, x, y)                         \
    GC_STORE1_2D_OFFSET(uint(packHalf2x16(r.xy)), name, x, y); \
    GC_STORE1_2D_OFFSET(uint(packHalf2x16(r.zw)), name, (x + 2), y)

#ifdef SOFTMAX_LAYER_MAX
BUFFER_DECLARATION(src, 1, uint, readonly);
BUFFER_DECLARATION(dst, 2, uint, writeonly);
#elif defined(SOFTMAX_LAYER_SHIFT_EXP_SUM)
BUFFER_DECLARATION(src, 1, uint, readonly);
BUFFER_DECLARATION(max, 2, uint, readonly);
BUFFER_DECLARATION(dst, 3, uint, writeonly);
BUFFER_DECLARATION(sum, 4, uint, writeonly);
#elif defined(SOFTMAX_LAYER_NORM)
BUFFER_DECLARATION(src, 1, uint, readonly);
BUFFER_DECLARATION(sum, 2, uint, readonly);
BUFFER_DECLARATION(dst, 3, uint, writeonly);
#endif // SOFTMAX_LAYER_MAX

layout(std140) uniform shader_params
{
#ifdef SOFTMAX_LAYER_MAX
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(dst);
    uint width;
#elif defined(SOFTMAX_LAYER_SHIFT_EXP_SUM)
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(max);
    TENSOR3D_PARAM_DECLARATION(dst);
    TENSOR3D_PARAM_DECLARATION(sum);
    uint width;
#elif defined(SOFTMAX_LAYER_NORM)
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(sum);
    TENSOR3D_PARAM_DECLARATION(dst);
#endif // SOFTMAX_LAYER_MAX
};

#define load_and_unpack(rs, names, xs, ys)           \
    do                                               \
    {                                                \
        uint packed_s;                               \
        GC_LOAD1_2D_OFFSET(packed_s, names, xs, ys); \
        rs = vec2(unpackHalf2x16(packed_s));         \
    } while(false)

#ifdef SOFTMAX_LAYER_MAX
/** Identifies the maximum value across the 1st dimension.
 *
 * @note Datatype must be given as a preprocessor argument using "#define DATA_TYPE_FP16"
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  width                             Input image width
 */
void main(void)
{
    Image src = GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);

    // Initialize local maximum
    vec4 max_val1 = CONVERT(type_min1, vec4);

    // Calculate max of row
    uint width2 = width >> 2;
    for(int i = 0; i < int(width2); i++)
    {
        vec4 data1;
        GC_LOAD4_IMAGE(data1, src, (i << 2), 0);
        max_val1 = MAX_OP(data1, max_val1);
    }

#ifdef NON_MULTIPLE_OF_4
    // Handle non multiple of 4
    for(int i = int(width2 << 2); i < int(width); i = i + 2)
    {
        vec2 data;
        load_and_unpack(data, src, i, 0);
        max_val1.x = MAX_OP(data.x, max_val1.x);
        if((i + 1) < int(width))
        {
            max_val1.x = MAX_OP(data.y, max_val1.x);
        }
    }
#endif                                     /* NON_MULTIPLE_OF_4 */

    // Perform max reduction
    max_val1.xy = MAX_OP(max_val1.xy, max_val1.zw);
    max_val1.x  = MAX_OP(max_val1.x, max_val1.y);
    vec2 res1   = vec2(max_val1.x, 0.f);
    uint res;
    res = uint(packHalf2x16(res1));

    // Store result
    GC_STORE1_2D_OFFSET(res, dst, 0, 0);
}
#elif defined(SOFTMAX_LAYER_SHIFT_EXP_SUM) // SOFTMAX_LAYER_MAX
/** Shifts the values of the input tensor by the max calculated in softmax_layer_max kernel,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note Datatype must be given as a preprocessor argument using "#define DATA_TYPE_FP16"
 *
 * @note In case the input is not multiple of 4 NON_MULTIPLE_OF_4 must be passed.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: F16
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
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[out] sum_ptr                           Pointer to the sum values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  sum_stride_x                      Stride of the sum values tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the sum values tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_stride_z                      Stride of the sum values tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the sum values tensor
 * @param[in]  width                             Input image width
 */
void main(void)
{
    Image src = GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image max = GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT(max);
    Image sum = GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

    // Load max value of 1D logits vector (row)
    vec2 datamaxinit;
    load_and_unpack(datamaxinit, max, 0, 0);
    vec4 max_val = CONVERT(datamaxinit.x, vec4);

    // Set sum vector
    vec4 sum1D1 = CONVERT(0.f, vec4);

    // Shift values, exp and sum
    uint width2 = width >> 2;
    for(int i = 0; i < int(width2); i++)
    {
        vec4 data;
        GC_LOAD4_IMAGE(data, src, (i << 2), 0);
        data = SUB_OP(data, max_val);
        data = EXP_OP(data);
        GC_STORE4_IMAGE(data, dst, (i << 2), 0);
        sum1D1 = ADD_OP(sum1D1, data);
    }

#ifdef NON_MULTIPLE_OF_4
    // Handle non multiple of 4
    for(int i = int(width2 << 2); i < int(width); i = i + 2)
    {
        vec2  datamiddle;
        float data1;
        load_and_unpack(datamiddle, src, i, 0);
        data1 = SUB_OP(datamiddle.x, max_val.x);
        data1 = EXP_OP(data1);
        vec2 datares1;
        if((i + 1) < int(width))
        {
            float data2;
            data2    = SUB_OP(datamiddle.y, max_val.x);
            data2    = EXP_OP(data2);
            datares1 = vec2(data1, data2);
            data1    = ADD_OP(data2, data1);
        }
        else
        {
            datares1 = vec2(data1, 0.f);
        }
        uint datares;
        datares = uint(packHalf2x16(datares1));
        GC_STORE1_2D_OFFSET(datares, dst, i, 0);
        sum1D1.x = ADD_OP(sum1D1.x, data1);
    }
#endif                            /* NON_MULTIPLE_OF_4 */

    // Perform min/max reduction
    sum1D1.xy = ADD_OP(sum1D1.xy, sum1D1.zw);
    sum1D1.x  = ADD_OP(sum1D1.x, sum1D1.y);
    vec2 res1 = vec2(sum1D1.x, 0.f);
    uint res;
    res = uint(packHalf2x16(res1));
    // Calculate and store result
    GC_STORE1_2D_OFFSET(res, sum, 0, 0);
}
#elif defined(SOFTMAX_LAYER_NORM) // SOFTMAX_LAYER_MAX
/** Divides all the values of the input tensor by the sum calculated from softmax_layer_shift_exp_sum kernel.
 *
 * @note Datatype must be given as a preprocessor argument using "#define DATA_TYPE_FP16"
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: F16
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
void main(void)
{
    Image src = GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image sum = GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(sum);

    // Load max value of 1D logits vector (row)
    vec2 sum1;
    load_and_unpack(sum1, sum, 0, int(gl_GlobalInvocationID.y));
    vec4 sum_val1 = CONVERT(sum1.x, vec4);

    vec4 data1;
    GC_LOAD4_IMAGE(data1, src, 0, 0);
    vec4 res = DIV_OP(data1, sum_val1);
    GC_STORE4_IMAGE(res, dst, 0, 0);
}
#endif                            // SOFTMAX_LAYER_MAX
#endif                            // DATA_TYPE_FP32