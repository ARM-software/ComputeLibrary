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

#if defined(DATA_TYPE_FP32)

float calculate_max(const int, Tensor3D, const int, const int, const int, const int, const int, const int);
float calculate_avg(const int, Tensor3D, const int, const int, const int, const int, const int, const int);

BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);

layout(std140) uniform shader_params
{
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(dst);
};

#define LOAD8(r, name, offset) \
    r.x = LOAD4(name, offset); \
    r.y = LOAD4(name, offset + uint(1))

#define LOAD16(r, name, offset)          \
    r.x = LOAD4(name, offset);           \
    r.y = LOAD4(name, offset + uint(1)); \
    r.z = LOAD4(name, offset + uint(2)); \
    r.w = LOAD4(name, offset + uint(3))

#define STORE16(name, offset, r)         \
    STORE4(name, offset, r.x);           \
    STORE4(name, offset + uint(1), r.y); \
    STORE4(name, offset + uint(2), r.z); \
    STORE4(name, offset + uint(3), r.w)

#if defined(POOL_AVG) || defined(POOL_L2)
#define POOL_OP(res, a, b) ((res) = (a) + (b))
#define POOL_OP_float(res, a, b) (res = a + b)
#define POOL_OP_vec2(res, a, b) ((res) = (a) + (b))
#else /* defined(POOL_AVG) || defined(POOL_L2) */
#define POOL_OP(res, a, b)        \
    (res) = (a);                  \
    if(isnan(a.x) || (a.x < b.x)) \
    {                             \
        res.x = b.x;              \
    }                             \
    if(isnan(a.y) || (a.y < b.y)) \
    {                             \
        res.y = b.y;              \
    }                             \
    if(isnan(a.z) || (a.z < b.z)) \
    {                             \
        res.z = b.z;              \
    }                             \
    if(isnan(a.w) || (a.w < b.w)) \
    {                             \
        res.w = b.w;              \
    }
#define POOL_OP_float(res, a, b) \
    (res) = (a);                 \
    if(isnan(a) || (a < b))      \
    {                            \
        res = b;                 \
    }
#define POOL_OP_vec2(res, a, b)   \
    (res) = (a);                  \
    if(isnan(a.x) || (a.x < b.x)) \
    {                             \
        res.x = b.x;              \
    }                             \
    if(isnan(a.y) || (a.y < b.y)) \
    {                             \
        res.y = b.y;              \
    }
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
#define POW2_OP(x, vec_size) ((x) * (x))
#else /* defined(POOL_L2) */
#define POW2_OP(x, vec_size) (x)
#endif /* defined(POOL_L2) */

#define DIV_OP(x, y) (x * (1.f / y))
#define SQRT_OP(x) sqrt((x))

#if defined(POOL_SIZE)
// Set the initial value for the pooling operation accordingly with the data type
#if defined(POOL_AVG) || defined(POOL_L2)
#define INITIAL_VALUE 0.0f
#else /* defined(POOL_AVG) || defined(POOL_L2) */
#define INITIAL_VALUE -3.402823466385289e+38
#endif // POOL_AVG
#endif //POOL_SIZE

#define POOLING3x3_STRIDE1(res, input, output)                                                                     \
    vec4 data00;                                                                                                   \
    vec2 data01;                                                                                                   \
    vec4 data10;                                                                                                   \
    vec2 data11;                                                                                                   \
    vec4 data20;                                                                                                   \
    vec2 data21;                                                                                                   \
    LOAD16(data00, input, tensor3D_offset(input, 0, 0, 0));                                                        \
    LOAD8(data01, input, tensor3D_offset(input, 0, 0, 0) + uint(4));                                               \
    LOAD16(data10, input, tensor3D_offset(input, 0, 1, 0));                                                        \
    LOAD8(data11, input, tensor3D_offset(input, 0, 1, 0) + uint(4));                                               \
    LOAD16(data20, input, tensor3D_offset(input, 0, 2, 0));                                                        \
    LOAD8(data21, input, tensor3D_offset(input, 0, 2, 0) + uint(4));                                               \
    data00 = POW2_OP(data00, 4);                                                                                   \
    data01 = POW2_OP(data01, 2);                                                                                   \
    data10 = POW2_OP(data10, 4);                                                                                   \
    data11 = POW2_OP(data11, 2);                                                                                   \
    data20 = POW2_OP(data20, 4);                                                                                   \
    data21 = POW2_OP(data21, 2);                                                                                   \
    \
    vec4 values000;                                                                                                \
    vec4 values001;                                                                                                \
    vec4 values010;                                                                                                \
    vec4 values100;                                                                                                \
    vec4 values101;                                                                                                \
    vec4 values11;                                                                                                 \
    vec4 values200;                                                                                                \
    vec4 values201;                                                                                                \
    vec4 values21;                                                                                                 \
    values000.xyzw = data00.xyzy;                                                                                  \
    values001.xyzw = data00.zwzw;                                                                                  \
    values010.x    = data01.x;                                                                                     \
    values010.y    = data00.w;                                                                                     \
    values010.zw   = data01.xy;                                                                                    \
    values100.xyzw = data10.xyzy;                                                                                  \
    values101.xyzw = data10.zwzw;                                                                                  \
    values11.x     = data11.x;                                                                                     \
    values11.y     = data10.w;                                                                                     \
    values11.zw    = data11.xy;                                                                                    \
    values200.xyzw = data20.xyzy;                                                                                  \
    values201.xyzw = data20.zwzw;                                                                                  \
    values21.x     = data21.x;                                                                                     \
    values21.y     = data20.w;                                                                                     \
    values21.zw    = data21.xy;                                                                                    \
    POOL_OP(values000.xyzw, values000.xyzw, values100.xyzw);                                                       \
    POOL_OP(values001.xyzw, values001.xyzw, values101.xyzw);                                                       \
    POOL_OP(values010.xyzw, values010.xyzw, values11.xyzw);                                                        \
    POOL_OP(values000.xyzw, values000.xyzw, values200.xyzw);                                                       \
    POOL_OP(values001.xyzw, values001.xyzw, values201.xyzw);                                                       \
    POOL_OP(values010.xyzw, values010.xyzw, values21.xyzw);                                                        \
    POOL_OP(res.xyzw, vec4(values000.xw, values001.z, values010.y), vec4(values000.y, values001.xw, values010.z)); \
    POOL_OP(res.xyzw, res.xyzw, vec4(values000.z, values001.y, values010.xw))

#define POOLING3x3_STRIDE2(res, input, output)                                                                     \
    vec4  data000;                                                                                                 \
    vec4  data001;                                                                                                 \
    float data010;                                                                                                 \
    vec4  data100;                                                                                                 \
    vec4  data101;                                                                                                 \
    float data11;                                                                                                  \
    vec4  data200;                                                                                                 \
    vec4  data201;                                                                                                 \
    float data21;                                                                                                  \
    LOAD16(data000, input, tensor3D_offset(input, 0, 0, 0));                                                       \
    LOAD16(data001, input, tensor3D_offset(input, 0, 0, 0) + uint(4));                                             \
    data010 = LOAD4(input, tensor3D_offset(input, 0, 0, 0) + uint(8));                                             \
    LOAD16(data100, input, tensor3D_offset(input, 0, 1, 0));                                                       \
    LOAD16(data101, input, tensor3D_offset(input, 0, 1, 0) + uint(4));                                             \
    data11 = LOAD4(input, tensor3D_offset(input, 0, 1, 0) + uint(8));                                              \
    LOAD16(data200, input, tensor3D_offset(input, 0, 2, 0));                                                       \
    LOAD16(data201, input, tensor3D_offset(input, 0, 2, 0) + uint(4));                                             \
    data21  = LOAD4(input, tensor3D_offset(input, 0, 2, 0) + uint(8));                                             \
    data000 = POW2_OP(data000, 4);                                                                                 \
    data001 = POW2_OP(data001, 4);                                                                                 \
    data010 = POW2_OP(data010, 1);                                                                                 \
    data100 = POW2_OP(data100, 4);                                                                                 \
    data101 = POW2_OP(data101, 4);                                                                                 \
    data11  = POW2_OP(data11, 1);                                                                                  \
    data200 = POW2_OP(data200, 4);                                                                                 \
    data201 = POW2_OP(data201, 4);                                                                                 \
    data21  = POW2_OP(data21, 1);                                                                                  \
    \
    vec4 values000;                                                                                                \
    vec4 values001;                                                                                                \
    vec4 values010;                                                                                                \
    vec4 values100;                                                                                                \
    vec4 values101;                                                                                                \
    vec4 values11;                                                                                                 \
    vec4 values200;                                                                                                \
    vec4 values201;                                                                                                \
    vec4 values21;                                                                                                 \
    values000.xyzw = data000.xyzz;                                                                                 \
    values001.xyzw = vec4(data000.w, data001.xxy);                                                                 \
    values010.xyzw = vec4(data001.zzw, data010);                                                                   \
    values100.xyzw = data100.xyzz;                                                                                 \
    values101.xyzw = vec4(data100.w, data101.xxy);                                                                 \
    values11.xyzw  = vec4(data101.zzw, data11);                                                                    \
    values200.xyzw = data200.xyzz;                                                                                 \
    values201.xyzw = vec4(data200.w, data201.xxy);                                                                 \
    values21.xyzw  = vec4(data201.zzw, data21);                                                                    \
    POOL_OP(values000.xyzw, values000.xyzw, values100.xyzw);                                                       \
    POOL_OP(values001.xyzw, values001.xyzw, values101.xyzw);                                                       \
    POOL_OP(values010.xyzw, values010.xyzw, values11.xyzw);                                                        \
    POOL_OP(values000.xyzw, values000.xyzw, values200.xyzw);                                                       \
    POOL_OP(values001.xyzw, values001.xyzw, values201.xyzw);                                                       \
    POOL_OP(values010.xyzw, values010.xyzw, values21.xyzw);                                                        \
    POOL_OP(res.xyzw, vec4(values000.xw, values001.z, values010.y), vec4(values000.y, values001.xw, values010.z)); \
    POOL_OP(res.xyzw, res.xyzw, vec4(values000.z, values001.y, values010.xw))

#define POOLING3x3_STRIDE3(res, input, output)                                                         \
    vec4 data000;                                                                                      \
    vec4 data001;                                                                                      \
    vec4 data010;                                                                                      \
    vec4 data100;                                                                                      \
    vec4 data101;                                                                                      \
    vec4 data11;                                                                                       \
    vec4 data200;                                                                                      \
    vec4 data201;                                                                                      \
    vec4 data21;                                                                                       \
    LOAD16(data000, input, tensor3D_offset(input, 0, 0, 0));                                           \
    LOAD16(data001, input, tensor3D_offset(input, 0, 0, 0) + uint(4));                                 \
    LOAD16(data010, input, tensor3D_offset(input, 0, 0, 0) + uint(8));                                 \
    LOAD16(data100, input, tensor3D_offset(input, 0, 1, 0));                                           \
    LOAD16(data101, input, tensor3D_offset(input, 0, 1, 0) + uint(4));                                 \
    LOAD16(data11, input, tensor3D_offset(input, 0, 1, 0) + uint(8));                                  \
    LOAD16(data200, input, tensor3D_offset(input, 0, 2, 0));                                           \
    LOAD16(data201, input, tensor3D_offset(input, 0, 2, 0) + uint(4));                                 \
    LOAD16(data21, input, tensor3D_offset(input, 0, 2, 0) + uint(8));                                  \
    data000 = POW2_OP(data000, 4);                                                                     \
    data001 = POW2_OP(data001, 4);                                                                     \
    data010 = POW2_OP(data010, 4);                                                                     \
    data100 = POW2_OP(data100, 4);                                                                     \
    data101 = POW2_OP(data101, 4);                                                                     \
    data11  = POW2_OP(data11, 4);                                                                      \
    data200 = POW2_OP(data200, 4);                                                                     \
    data201 = POW2_OP(data201, 4);                                                                     \
    data21  = POW2_OP(data21, 4);                                                                      \
    \
    POOL_OP(data000.xyzw, data000.xyzw, data100.xyzw);                                                 \
    POOL_OP(data001.xyzw, data001.xyzw, data101.xyzw);                                                 \
    POOL_OP(data010.xyzw, data010.xyzw, data11.xyzw);                                                  \
    POOL_OP(data000.xyzw, data000.xyzw, data200.xyzw);                                                 \
    POOL_OP(data001.xyzw, data001.xyzw, data201.xyzw);                                                 \
    POOL_OP(data010.xyzw, data010.xyzw, data21.xyzw);                                                  \
    POOL_OP(res.xyzw, vec4(data000.xw, data001.z, data010.y), vec4(data000.y, data001.xw, data010.z)); \
    POOL_OP(res.xyzw, res.xyzw, vec4(data000.z, data001.y, data010.xw))

float calculate_max(const int pool_size, Tensor3D src, const int upper_bound_w, const int upper_bound_h, const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int start_x = int(gl_GlobalInvocationID.x) * stride_x - pad_x;
    int start_y = int(gl_GlobalInvocationID.y) * stride_y - pad_y;
    int end_x   = int(min(start_x + pool_size, upper_bound_w));
    int end_y   = int(min(start_y + pool_size, upper_bound_h));

    float data_max;
    data_max = LOAD4(src, tensor3D_offset(src, 0, 0, 0));

    for(int i = 0; (start_y + i) < end_y; ++i)
    {
        for(int j = 0; (start_x + j) < end_x; ++j)
        {
            float data = LOAD4(src, tensor3D_offset(src, j, i, 0));
            POOL_OP_float(data_max, data_max, data);
        }
    }

    return data_max;
}

float calculate_avg(const int pool_size, Tensor3D src, const int upper_bound_w, const int upper_bound_h, const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int start_x = int(gl_GlobalInvocationID.x) * stride_x - pad_x;
    int start_y = int(gl_GlobalInvocationID.y) * stride_y - pad_y;
    int end_x   = int(min(start_x + pool_size, upper_bound_w));
    int end_y   = int(min(start_y + pool_size, upper_bound_h));

    float data_total = 0.0f;
    for(int i = 0; (start_x + i) < end_x; i++)
    {
        for(int j = 0; (start_y + j) < end_y; ++j)
        {
            float data = LOAD4(src, tensor3D_offset(src, i, j, 0));
            if(isnan(data))
            {
                data = 0.0f;
            }
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data = POW2_OP(data, 1);
#endif /* defined(POOL_L2) */
            data_total = data_total + data;
        }
    }

#if defined(EXCLUDE_PADDING)
    start_x = max(0, start_x);
    start_y = max(0, start_y);
#endif /* defined(EXCLUDE_PADDING) */

    return data_total / float((end_y - start_y) * (end_x - start_x));
}

#ifdef POOLING_LAYER_2
/** Performs a pooling function of pool size equal to 2.
 *
 * @note Supported data types are F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    //Load and calculate data
    float res;
#if defined(POOL_AVG) || defined(POOL_L2)
    res = calculate_avg(2, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#else  /*POOL_AVG*/
    res = calculate_max(2, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    STORE4(dst, CURRENT_OFFSET(dst), res);
}

#elif defined(POOLING_LAYER_3)
/** Performs a pooling function of pool size equal to 3.
 *
 * @note Supported data types are F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    //Load and calculate data
    float res;
#if defined(POOL_AVG) || defined(POOL_L2)
    res = calculate_avg(3, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#else  /*POOL_AVG*/
    res = calculate_max(3, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    STORE4(dst, CURRENT_OFFSET(dst), res);
}

#elif defined(POOLING_LAYER_3_OPTIMIZED)
/** Performs an optimized pooling function of pool size equal to 3 when the stride_x is less equal than 3
 *
 * @note Supported data types are F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    vec4 res;
    // Perform pooling 3x3 for 4 output elements
#if STRIDE_X == 1
    POOLING3x3_STRIDE1(res, src, dst);
#elif STRIDE_X == 2
    POOLING3x3_STRIDE2(res, src, dst);
#elif STRIDE_X == 3
    POOLING3x3_STRIDE3(res, src, dst);
#endif /*STRIDE_X == 1*/

    // Divide by pool region in case of average pooling
#if defined(POOL_AVG) || defined(POOL_L2)
    ivec4 start_x = ((ivec4(int(gl_GlobalInvocationID.x) * 4) + ivec4(0, 1, 2, 3)) * (ivec4(STRIDE_X))) - (ivec4(PAD_X));
    int   start_y = int(gl_GlobalInvocationID.y) * STRIDE_Y - PAD_Y;
    ivec4 end_x   = min((start_x + (ivec4(3))), (ivec4(MAX_WIDTH)));
    int   end_y   = min((start_y + 3), MAX_HEIGHT);
#if defined(EXCLUDE_PADDING)
    start_x       = max(ivec4(0), start_x);
    start_y       = max(0, start_y);
#endif /* defined(EXCLUDE_PADDING) */
    res *= (vec4((1.f)) / vec4((ivec4(end_y - start_y)) * (end_x - start_x)));
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    STORE16(dst, CURRENT_OFFSET(dst), res);
}

#elif defined(POOLING_LAYER_7)
/** Performs a pooling function of pool size equal to 7.
 *
 * @note Supported data types are F32;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    //Load and calculate data
    float res;
#if defined(POOL_AVG) || defined(POOL_L2)
    res = calculate_avg(7, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#else  /*POOL_AVG*/
    res = calculate_max(7, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    STORE4(dst, CURRENT_OFFSET(dst), res);
}

#elif defined(POOLING_LAYER_N)
/** Performs a pooling function of pool size equal to N
 *
 * @note Supported data types are F32;
 * @note Pool size must be passed using POOL_SIZE e.g. POOL_SIZE=13;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    vec4 vdata0;
    vdata0 = vec4(INITIAL_VALUE);
    vec4 vdata1;
    vdata1 = vec4(INITIAL_VALUE);
    float sdata;
    sdata = float(INITIAL_VALUE);

    for(int y = 0; y < int(POOL_SIZE); y++)
    {
        int x = 0;
        for(; x <= (int(POOL_SIZE) - 8); x += 8)
        {
            vec4 data2;
            vec4 data3;
            LOAD16(data2, src, tensor3D_offset(src, x, y, 0));
            LOAD16(data3, src, tensor3D_offset(src, x, y, 0) + uint(4));

#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data2 *= data2;
            data3 *= data3;
#endif /* defined(POOL_L2) */

            POOL_OP(vdata0, vdata0, data2);
            POOL_OP(vdata1, vdata1, data3);
        }

        // Leftover
        for(; x < int(POOL_SIZE); ++x)
        {
            float data4 = LOAD4(src, tensor3D_offset(src, x, y, 0));
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data4 *= data4;
#endif /* defined(POOL_L2) */
            POOL_OP_float(sdata, sdata, data4);
        }
    }

    //Reduce result
    vec4 reduce4;
    POOL_OP(reduce4, vdata0.xyzw, vdata1.xyzw);
    vec2 reduce2;
    POOL_OP_vec2(reduce2, reduce4.xy, reduce4.zw);
    float res;
    POOL_OP_float(res, reduce2.x, reduce2.y);
    POOL_OP_float(res, res, sdata);

#if defined(POOL_AVG) || defined(POOL_L2)
    {
        // Divide by pool region in case of average pooling
        int start_x = int(gl_GlobalInvocationID.x) * STRIDE_X - PAD_X;
        int start_y = int(gl_GlobalInvocationID.y) * STRIDE_Y - PAD_Y;
        int end_x   = int(min(start_x + POOL_SIZE, MAX_WIDTH));
        int end_y   = int(min(start_y + POOL_SIZE, MAX_HEIGHT));
#if defined(EXCLUDE_PADDING)
        start_x     = max(0, start_x);
        start_y     = max(0, start_y);
#endif /* defined(EXCLUDE_PADDING) */
        float res1  = float((end_y - start_y) * (end_x - start_x));
        res         = DIV_OP(res, res1);
    }
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    STORE4(dst, CURRENT_OFFSET(dst), res);
}
#endif /* POOLING_LAYER_2 */

#elif defined(DATA_TYPE_FP16)

precision mediump float;

vec2 load_and_unpack(Tensor3D, uint);
vec2 calculate_max(const int, Tensor3D, const int, const int, const int, const int, const int, const int);
vec2 calculate_avg(const int, Tensor3D, const int, const int, const int, const int, const int, const int);

BUFFER_DECLARATION(src, 1, uint, readonly);
BUFFER_DECLARATION(dst, 2, uint, writeonly);

layout(std140) uniform shader_params
{
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(dst);
};

#define LOAD2_fp16(r, name, offset) \
    r.xy = load_and_unpack(name, offset)

#define LOAD4_fp16(r, name, offset)       \
    r.xy = load_and_unpack(name, offset); \
    r.zw = load_and_unpack(name, offset + uint(1))

#define STORE4_fp16(name, offset, r)             \
    uint datastore1;                             \
    uint datastore2;                             \
    datastore1 = uint(packHalf2x16(r.xy));       \
    datastore2 = uint(packHalf2x16(r.zw));       \
    STORE1(name, offset << uint(1), datastore1); \
    STORE1(name, (offset << uint(1)) + uint(1), datastore2)

#if defined(POOL_AVG) || defined(POOL_L2)
#define POOL_OP(res, a, b) ((res) = (a) + (b))
#define POOL_OP_float(res, a, b) (res = a + b)
#define POOL_OP_vec2(res, a, b) ((res) = (a) + (b))
#else /* defined(POOL_AVG) || defined(POOL_L2) */
#define POOL_OP(res, a, b)        \
    (res) = (a);                  \
    if(isnan(a.x) || (a.x < b.x)) \
    {                             \
        res.x = b.x;              \
    }                             \
    if(isnan(a.y) || (a.y < b.y)) \
    {                             \
        res.y = b.y;              \
    }                             \
    if(isnan(a.z) || (a.z < b.z)) \
    {                             \
        res.z = b.z;              \
    }                             \
    if(isnan(a.w) || (a.w < b.w)) \
    {                             \
        res.w = b.w;              \
    }
#define POOL_OP_float(res, a, b) \
    (res) = (a);                 \
    if(isnan(a) || (a < b))      \
    {                            \
        res = b;                 \
    }
#define POOL_OP_vec2(res, a, b)   \
    (res) = (a);                  \
    if(isnan(a.x) || (a.x < b.x)) \
    {                             \
        res.x = b.x;              \
    }                             \
    if(isnan(a.y) || (a.y < b.y)) \
    {                             \
        res.y = b.y;              \
    }
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
#define POW2_OP(x, vec_size) ((x) * (x))
#else /* defined(POOL_L2) */
#define POW2_OP(x, vec_size) (x)
#endif /* defined(POOL_L2) */

#define DIV_OP(x, y) (x * (1.f / y))
#define SQRT_OP(x) sqrt((x))

#if defined(POOL_SIZE)
// Set the initial value for the pooling operation accordingly with the data type
#if defined(POOL_AVG) || defined(POOL_L2)
#define INITIAL_VALUE 0.0f
#else /* defined(POOL_AVG) || defined(POOL_L2) */
#define INITIAL_VALUE -65504.0f
#endif //POOL_AVG
#endif //POOL_SIZE

#define POOLING3x3_STRIDE1_fp16(res, input, output)                                                                \
    vec4 data00;                                                                                                   \
    vec2 data01;                                                                                                   \
    vec4 data10;                                                                                                   \
    vec2 data11;                                                                                                   \
    vec4 data20;                                                                                                   \
    vec2 data21;                                                                                                   \
    LOAD4_fp16(data00, input, (tensor3D_offset_fp16(input, 0, 0, 0) >> uint(2)));                                  \
    LOAD2_fp16(data01, input, (tensor3D_offset_fp16(input, 0, 0, 0) >> uint(2)) + uint(2));                        \
    LOAD4_fp16(data10, input, (tensor3D_offset_fp16(input, 0, 1, 0) >> uint(2)));                                  \
    LOAD2_fp16(data11, input, (tensor3D_offset_fp16(input, 0, 1, 0) >> uint(2)) + uint(2));                        \
    LOAD4_fp16(data20, input, (tensor3D_offset_fp16(input, 0, 2, 0) >> uint(2)));                                  \
    LOAD2_fp16(data21, input, (tensor3D_offset_fp16(input, 0, 2, 0) >> uint(2)) + uint(2));                        \
    data00 = POW2_OP(data00, 4);                                                                                   \
    data01 = POW2_OP(data01, 2);                                                                                   \
    data10 = POW2_OP(data10, 4);                                                                                   \
    data11 = POW2_OP(data11, 2);                                                                                   \
    data20 = POW2_OP(data20, 4);                                                                                   \
    data21 = POW2_OP(data21, 2);                                                                                   \
    \
    vec4 values000;                                                                                                \
    vec4 values001;                                                                                                \
    vec4 values010;                                                                                                \
    vec4 values100;                                                                                                \
    vec4 values101;                                                                                                \
    vec4 values11;                                                                                                 \
    vec4 values200;                                                                                                \
    vec4 values201;                                                                                                \
    vec4 values21;                                                                                                 \
    values000.xyzw = data00.xyzy;                                                                                  \
    values001.xyzw = data00.zwzw;                                                                                  \
    values010.x    = data01.x;                                                                                     \
    values010.y    = data00.w;                                                                                     \
    values010.zw   = data01.xy;                                                                                    \
    values100.xyzw = data10.xyzy;                                                                                  \
    values101.xyzw = data10.zwzw;                                                                                  \
    values11.x     = data11.x;                                                                                     \
    values11.y     = data10.w;                                                                                     \
    values11.zw    = data11.xy;                                                                                    \
    values200.xyzw = data20.xyzy;                                                                                  \
    values201.xyzw = data20.zwzw;                                                                                  \
    values21.x     = data21.x;                                                                                     \
    values21.y     = data20.w;                                                                                     \
    values21.zw    = data21.xy;                                                                                    \
    POOL_OP(values000.xyzw, values000.xyzw, values100.xyzw);                                                       \
    POOL_OP(values001.xyzw, values001.xyzw, values101.xyzw);                                                       \
    POOL_OP(values010.xyzw, values010.xyzw, values11.xyzw);                                                        \
    POOL_OP(values000.xyzw, values000.xyzw, values200.xyzw);                                                       \
    POOL_OP(values001.xyzw, values001.xyzw, values201.xyzw);                                                       \
    POOL_OP(values010.xyzw, values010.xyzw, values21.xyzw);                                                        \
    POOL_OP(res.xyzw, vec4(values000.xw, values001.z, values010.y), vec4(values000.y, values001.xw, values010.z)); \
    POOL_OP(res.xyzw, res.xyzw, vec4(values000.z, values001.y, values010.xw))

#define POOLING3x3_STRIDE2_fp16(res, input, output)                                                                \
    vec4  data000;                                                                                                 \
    vec4  data001;                                                                                                 \
    float data010;                                                                                                 \
    vec4  data100;                                                                                                 \
    vec4  data101;                                                                                                 \
    float data11;                                                                                                  \
    vec4  data200;                                                                                                 \
    vec4  data201;                                                                                                 \
    float data21;                                                                                                  \
    vec2  datamiddle0;                                                                                             \
    vec2  datamiddle1;                                                                                             \
    vec2  datamiddle2;                                                                                             \
    LOAD4_fp16(data000, input, (tensor3D_offset_fp16(input, 0, 0, 0) >> uint(2)));                                 \
    LOAD4_fp16(data001, input, (tensor3D_offset_fp16(input, 0, 0, 0) >> uint(2)) + uint(2));                       \
    datamiddle0 = load_and_unpack(input, (tensor3D_offset_fp16(input, 0, 0, 0) >> uint(2)) + uint(4));             \
    data010     = datamiddle0.x;                                                                                   \
    LOAD4_fp16(data100, input, (tensor3D_offset_fp16(input, 0, 1, 0) >> uint(2)));                                 \
    LOAD4_fp16(data101, input, (tensor3D_offset_fp16(input, 0, 1, 0) >> uint(2)) + uint(2));                       \
    datamiddle1 = load_and_unpack(input, (tensor3D_offset_fp16(input, 0, 1, 0) >> uint(2)) + uint(4));             \
    data11      = datamiddle1.x;                                                                                   \
    LOAD4_fp16(data200, input, (tensor3D_offset_fp16(input, 0, 2, 0) >> uint(2)));                                 \
    LOAD4_fp16(data201, input, (tensor3D_offset_fp16(input, 0, 2, 0) >> uint(2)) + uint(2));                       \
    datamiddle2 = load_and_unpack(input, (tensor3D_offset_fp16(input, 0, 2, 0) >> uint(2)) + uint(4));             \
    data21      = datamiddle2.x;                                                                                   \
    data000     = POW2_OP(data000, 4);                                                                             \
    data001     = POW2_OP(data001, 4);                                                                             \
    data010     = POW2_OP(data010, 1);                                                                             \
    data100     = POW2_OP(data100, 4);                                                                             \
    data101     = POW2_OP(data101, 4);                                                                             \
    data11      = POW2_OP(data11, 1);                                                                              \
    data200     = POW2_OP(data200, 4);                                                                             \
    data201     = POW2_OP(data201, 4);                                                                             \
    data21      = POW2_OP(data21, 1);                                                                              \
    \
    vec4 values000;                                                                                                \
    vec4 values001;                                                                                                \
    vec4 values010;                                                                                                \
    vec4 values100;                                                                                                \
    vec4 values101;                                                                                                \
    vec4 values11;                                                                                                 \
    vec4 values200;                                                                                                \
    vec4 values201;                                                                                                \
    vec4 values21;                                                                                                 \
    values000.xyzw = data000.xyzz;                                                                                 \
    values001.xyzw = vec4(data000.w, data001.xxy);                                                                 \
    values010.xyzw = vec4(data001.zzw, data010);                                                                   \
    values100.xyzw = data100.xyzz;                                                                                 \
    values101.xyzw = vec4(data100.w, data101.xxy);                                                                 \
    values11.xyzw  = vec4(data101.zzw, data11);                                                                    \
    values200.xyzw = data200.xyzz;                                                                                 \
    values201.xyzw = vec4(data200.w, data201.xxy);                                                                 \
    values21.xyzw  = vec4(data201.zzw, data21);                                                                    \
    POOL_OP(values000.xyzw, values000.xyzw, values100.xyzw);                                                       \
    POOL_OP(values001.xyzw, values001.xyzw, values101.xyzw);                                                       \
    POOL_OP(values010.xyzw, values010.xyzw, values11.xyzw);                                                        \
    POOL_OP(values000.xyzw, values000.xyzw, values200.xyzw);                                                       \
    POOL_OP(values001.xyzw, values001.xyzw, values201.xyzw);                                                       \
    POOL_OP(values010.xyzw, values010.xyzw, values21.xyzw);                                                        \
    POOL_OP(res.xyzw, vec4(values000.xw, values001.z, values010.y), vec4(values000.y, values001.xw, values010.z)); \
    POOL_OP(res.xyzw, res.xyzw, vec4(values000.z, values001.y, values010.xw))

#define POOLING3x3_STRIDE3_fp16(res, input, output)                                                    \
    vec4 data000;                                                                                      \
    vec4 data001;                                                                                      \
    vec4 data010;                                                                                      \
    vec4 data100;                                                                                      \
    vec4 data101;                                                                                      \
    vec4 data11;                                                                                       \
    vec4 data200;                                                                                      \
    vec4 data201;                                                                                      \
    vec4 data21;                                                                                       \
    LOAD4_fp16(data000, input, (tensor3D_offset_fp16(input, 0, 0, 0) >> uint(2)));                     \
    LOAD4_fp16(data001, input, (tensor3D_offset_fp16(input, 0, 0, 0) >> uint(2)) + uint(2));           \
    LOAD4_fp16(data010, input, (tensor3D_offset_fp16(input, 0, 0, 0) >> uint(2)) + uint(4));           \
    LOAD4_fp16(data100, input, (tensor3D_offset_fp16(input, 0, 1, 0) >> uint(2)));                     \
    LOAD4_fp16(data101, input, (tensor3D_offset_fp16(input, 0, 1, 0) >> uint(2)) + uint(2));           \
    LOAD4_fp16(data11, input, (tensor3D_offset_fp16(input, 0, 1, 0) >> uint(2)) + uint(4));            \
    LOAD4_fp16(data200, input, (tensor3D_offset_fp16(input, 0, 2, 0) >> uint(2)));                     \
    LOAD4_fp16(data201, input, (tensor3D_offset_fp16(input, 0, 2, 0) >> uint(2)) + uint(2));           \
    LOAD4_fp16(data21, input, (tensor3D_offset_fp16(input, 0, 2, 0) >> uint(2)) + uint(4));            \
    data000 = POW2_OP(data000, 4);                                                                     \
    data001 = POW2_OP(data001, 4);                                                                     \
    data010 = POW2_OP(data010, 4);                                                                     \
    data100 = POW2_OP(data100, 4);                                                                     \
    data101 = POW2_OP(data101, 4);                                                                     \
    data11  = POW2_OP(data11, 4);                                                                      \
    data200 = POW2_OP(data200, 4);                                                                     \
    data201 = POW2_OP(data201, 4);                                                                     \
    data21  = POW2_OP(data21, 4);                                                                      \
    \
    POOL_OP(data000.xyzw, data000.xyzw, data100.xyzw);                                                 \
    POOL_OP(data001.xyzw, data001.xyzw, data101.xyzw);                                                 \
    POOL_OP(data010.xyzw, data010.xyzw, data11.xyzw);                                                  \
    POOL_OP(data000.xyzw, data000.xyzw, data200.xyzw);                                                 \
    POOL_OP(data001.xyzw, data001.xyzw, data201.xyzw);                                                 \
    POOL_OP(data010.xyzw, data010.xyzw, data21.xyzw);                                                  \
    POOL_OP(res.xyzw, vec4(data000.xw, data001.z, data010.y), vec4(data000.y, data001.xw, data010.z)); \
    POOL_OP(res.xyzw, res.xyzw, vec4(data000.z, data001.y, data010.xw))

vec2 load_and_unpack(Tensor3D src, uint offset)
{
    uint packed_s;
    vec2 s;
    LOAD1(packed_s, src, offset);

    s = vec2(unpackHalf2x16(packed_s));
    return s;
}

vec2 calculate_max(const int pool_size, Tensor3D src, const int upper_bound_w, const int upper_bound_h, const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int start_x1 = int(gl_GlobalInvocationID.x) * stride_x - pad_x;
    int start_y1 = int(gl_GlobalInvocationID.y) * stride_y - pad_y;
    int end_x1   = int(min(start_x1 + pool_size, upper_bound_w));
    int end_y1   = int(min(start_y1 + pool_size, upper_bound_h));

    int start_x2 = start_x1 + stride_x;
    int start_y2 = start_y1;
    int end_x2   = int(min(start_x2 + pool_size, upper_bound_w));
    int end_y2   = int(min(start_y2 + pool_size, upper_bound_h));

    //Initialize maximum
    vec2 data_max = vec2(0);

    //Load and Set initial maximum1
    vec2 data_init1 = load_and_unpack(src, tensor3D_offset_fp16(src, 0, 0, 0) >> uint(2));
    data_max.x      = data_init1.x;

    //Load and Set initial maximum2
    if(end_x1 < upper_bound_w)
    {
        if((stride_x % 2) == 0)
        {
            vec2 data_init2 = load_and_unpack(src, tensor3D_offset_fp16(src, stride_x, 0, 0) >> uint(2));
            data_max.y      = data_init2.x;
        }
        else
        {
            vec2 data_init2 = load_and_unpack(src, tensor3D_offset_fp16(src, stride_x - 1, 0, 0) >> uint(2));
            data_max.y      = data_init2.y;
        }
    }

    for(int i = 0; (start_y1 + i) < end_y1; i++)
        for(int j = 0; (start_x1 + j) < end_x1; j = j + 2)
        {
            //Calculate maximum1
            if((start_x1 + j + 1) < end_x1)
            {
                vec2  data1 = load_and_unpack(src, tensor3D_offset_fp16(src, j, i, 0) >> uint(2));
                float data_mr1;
                POOL_OP_float(data_mr1, data1.x, data1.y);
                POOL_OP_float(data_max.x, data_max.x, data_mr1);
            }
            else
            {
                vec2 data1 = load_and_unpack(src, tensor3D_offset_fp16(src, j, i, 0) >> uint(2));
                POOL_OP_float(data_max.x, data_max.x, data1.x);
            }

            //Calculate maximum2
            if((start_x2 + j) < end_x2 && end_x1 < upper_bound_w)
            {
                if((stride_x % 2) == 0)
                {
                    vec2 data2 = load_and_unpack(src, (tensor3D_offset_fp16(src, (j + stride_x), i, 0) >> uint(2)));

                    if((start_x2 + j + 1) < end_x2)
                    {
                        float data_mr2;
                        POOL_OP_float(data_mr2, data2.x, data2.y);
                        POOL_OP_float(data_max.y, data_max.y, data_mr2);
                    }
                    else
                    {
                        POOL_OP_float(data_max.y, data_max.y, data2.x);
                    }
                }
                else
                {
                    vec2 data2 = load_and_unpack(src, (tensor3D_offset_fp16(src, (j + stride_x - 1), i, 0) >> uint(2)));
                    vec2 data3 = load_and_unpack(src, (tensor3D_offset_fp16(src, (j + stride_x + 1), i, 0) >> uint(2)));
                    if((start_x2 + j + 1) < end_x2)
                    {
                        float data_mr2;
                        POOL_OP_float(data_mr2, data3.x, data2.y);
                        POOL_OP_float(data_max.y, data_max.y, data_mr2);
                    }
                    else
                    {
                        POOL_OP_float(data_max.y, data_max.y, data2.y);
                    }
                }
            }
        }
    return data_max;
}

vec2 calculate_avg(const int pool_size, Tensor3D src, const int upper_bound_w, const int upper_bound_h, const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int start_x1 = (2 * int(gl_GlobalInvocationID.x)) * stride_x - pad_x;
    int start_y1 = int(gl_GlobalInvocationID.y) * stride_y - pad_y;
    int end_x1   = int(min(start_x1 + pool_size, upper_bound_w));
    int end_y1   = int(min(start_y1 + pool_size, upper_bound_h));

    int start_x2 = start_x1 + stride_x;
    int start_y2 = start_y1;
    int end_x2   = int(min(start_x2 + pool_size, upper_bound_w));
    int end_y2   = int(min(start_y2 + pool_size, upper_bound_h));

    //Initialize sum
    float data_total1 = float(0);
    float data_total2 = float(0);
    for(int i = 0; (start_y1 + i) < end_y1; i++)
        for(int j = 0; (start_x1 + j) < end_x1; j = j + 2)
        {
            vec2 data1 = load_and_unpack(src, tensor3D_offset_fp16(src, j, i, 0) >> uint(2));
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data1 = POW2_OP(data1, 2);
#endif /* defined(POOL_L2) */
            //Calculate sum1
            if((start_x1 + j + 1) < end_x1)
            {
                data_total1 = data_total1 + data1.x + data1.y;
            }
            else
            {
                data_total1 = data_total1 + data1.x;
            }

            //Calculate sum2
            if((start_x2 + j) < end_x2 && end_x1 <= upper_bound_w)
            {
                if((stride_x % 2) == 0)
                {
                    vec2 data2 = load_and_unpack(src, (tensor3D_offset_fp16(src, (j + stride_x), i, 0) >> uint(2)));
#if defined(POOL_L2)
                    // Raise to power of 2 for L2 Pooling
                    data2 = POW2_OP(data2, 2);
#endif /* defined(POOL_L2) */
                    if((start_x2 + j + 1) < end_x2)
                    {
                        data_total2 = data_total2 + data2.x + data2.y;
                    }
                    else
                    {
                        data_total2 = data_total2 + data2.x;
                    }
                }
                else
                {
                    vec2 data2 = load_and_unpack(src, (tensor3D_offset_fp16(src, (j + stride_x - 1), i, 0) >> uint(2)));
                    vec2 data3 = load_and_unpack(src, (tensor3D_offset_fp16(src, (j + stride_x + 1), i, 0) >> uint(2)));
#if defined(POOL_L2)
                    // Raise to power of 2 for L2 Pooling
                    data2 = POW2_OP(data2, 2);
                    data3 = POW2_OP(data3, 2);
#endif /* defined(POOL_L2) */
                    if((start_x2 + j + 1) < end_x2)
                    {
                        data_total2 = data_total2 + data3.x + data2.y;
                    }
                    else
                    {
                        data_total2 = data_total2 + data2.y;
                    }
                }
            }
        }
#if defined(EXCLUDE_PADDING)
    start_x1 = max(0, start_x1);
    start_y1 = max(0, start_y1);
    start_x2 = max(0, start_x2);
    start_y2 = max(0, start_y2);
#endif /* defined(EXCLUDE_PADDING) */

    //Calculate average
    vec2 data_avg;
    data_avg.x = data_total1 / float((end_y1 - start_y1) * (end_x1 - start_x1));
    data_avg.y = data_total2 / float((end_y2 - start_y2) * (end_x2 - start_x2));

    return data_avg;
}

#ifdef POOLING_LAYER_2
/** Performs a pooling function of pool size equal to 2.
 *
 * @note Supported data types are F16;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT_FP16(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

    //Load and calculate data
    vec2 data;
    uint res;
#if defined(POOL_AVG) || defined(POOL_L2)
    data = calculate_avg(2, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#else  /*POOL_AVG*/
    data = calculate_max(2, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    data = SQRT_OP(data);
#endif /* defined(POOL_L2) */

    res = uint(packHalf2x16(data));

    // Store result
    STORE1(dst, CURRENT_OFFSET(dst) >> uint(2), res);
}

#elif defined(POOLING_LAYER_3)
/** Performs a pooling function of pool size equal to 3.
 *
 * @note Supported data types are F16;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT_FP16(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

    //Load and calculate data
    vec2 data;
    uint res;
#if defined(POOL_AVG) || defined(POOL_L2)
    data = calculate_avg(3, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#else  /*POOL_AVG*/
    data = calculate_max(3, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    data = SQRT_OP(data);
#endif /* defined(POOL_L2) */

    res = uint(packHalf2x16(data));

    // Store result
    STORE1(dst, CURRENT_OFFSET(dst) >> uint(2), res);
}

#elif defined(POOLING_LAYER_3_OPTIMIZED)
/** Performs an optimized pooling function of pool size equal to 3 when the stride_x is less equal than 3
 *
 * @note Supported data types are F16;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT_FP16(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

    vec4 res;
    // Perform pooling 3x3 for 4 output elements
#if STRIDE_X == 1
    POOLING3x3_STRIDE1_fp16(res, src, dst);
#elif STRIDE_X == 2
    POOLING3x3_STRIDE2_fp16(res, src, dst);
#elif STRIDE_X == 3
    POOLING3x3_STRIDE3_fp16(res, src, dst);
#endif /*STRIDE_X == 1*/

    // Divide by pool region in case of average pooling
#if defined(POOL_AVG) || defined(POOL_L2)
    ivec4 start_x = ((ivec4(int(gl_GlobalInvocationID.x) * 4) + ivec4(0, 1, 2, 3)) * (ivec4(STRIDE_X))) - (ivec4(PAD_X));
    int   start_y = int(gl_GlobalInvocationID.y) * STRIDE_Y - PAD_Y;
    ivec4 end_x   = min((start_x + (ivec4(3))), (ivec4(MAX_WIDTH)));
    int   end_y   = min((start_y + 3), MAX_HEIGHT);
#if defined(EXCLUDE_PADDING)
    start_x       = max(ivec4(0), start_x);
    start_y       = max(0, start_y);
#endif /* defined(EXCLUDE_PADDING) */
    res *= (vec4((1.f)) / vec4((ivec4(end_y - start_y)) * (end_x - start_x)));
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    STORE4_fp16(dst, CURRENT_OFFSET(dst) >> uint(3), res);
}

#elif defined(POOLING_LAYER_7)
/** Performs a pooling function of pool size equal to 7.
 *
 * @note Supported data types are F16;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT_FP16(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

    //Load and calculate data
    vec2 data;
    uint res;
#if defined(POOL_AVG) || defined(POOL_L2)
    data = calculate_avg(7, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#else  /*POOL_AVG*/
    data = calculate_max(7, src, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    data = SQRT_OP(data);
#endif /* defined(POOL_L2) */

    res = uint(packHalf2x16(data));

    // Store result
    STORE1(dst, CURRENT_OFFSET(dst) >> uint(2), res);
}

#elif defined(POOLING_LAYER_N)
/** Performs a pooling function of pool size equal to N
 *
 * @note Supported data types are F16;
 * @note Pool size must be passed using POOL_SIZE e.g. POOL_SIZE=13;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    // Get pixels pointer
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT_FP16(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

    vec4 vdata00;
    vdata00 = vec4(INITIAL_VALUE);
    vec4 vdata01;
    vdata01 = vec4(INITIAL_VALUE);
    vec4 vdata10;
    vdata10 = vec4(INITIAL_VALUE);
    vec4 vdata11;
    vdata11 = vec4(INITIAL_VALUE);
    vec2 sdata;
    sdata = vec2(INITIAL_VALUE);

    for(int y = 0; y < int(POOL_SIZE); y++)
    {
        int x = 0;
        for(; x <= (int(POOL_SIZE) - 8); x += 8)
        {
            vec4 data2;
            vec4 data3;
            LOAD4_fp16(data2, src, (tensor3D_offset_fp16(src, x, y, 0) >> uint(2)));
            LOAD4_fp16(data3, src, (tensor3D_offset_fp16(src, x, y, 0) >> uint(2)) + uint(2));

#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data2 *= data2;
            data3 *= data3;
#endif /* defined(POOL_L2) */

            POOL_OP(vdata00, vdata00, data2);
            POOL_OP(vdata10, vdata10, data3);
        }

        // Leftover
        for(; x < int(POOL_SIZE); x = x + 2)
        {
            vec2 data4middle;
            data4middle = load_and_unpack(src, (tensor3D_offset_fp16(src, x, y, 0) >> uint(2)));
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            data4middle *= data4middle;
#endif /* defined(POOL_L2) */
            if((x + 1) >= int(POOL_SIZE))
            {
                POOL_OP_float(sdata.x, sdata.x, data4middle.x);
            }
            else
            {
                float data4;
                POOL_OP_float(data4, data4middle.x, data4middle.y);
                POOL_OP_float(sdata.x, sdata.x, data4);
            }
        }
    }

    for(int y = 0; y < int(POOL_SIZE); y++)
    {
        if((STRIDE_X % 2) == 0)
        {
            int x1 = STRIDE_X;
            for(; x1 <= (int(POOL_SIZE + STRIDE_X) - 8); x1 += 8)
            {
                vec4 data2;
                vec4 data3;
                LOAD4_fp16(data2, src, (tensor3D_offset_fp16(src, x1, y, 0) >> uint(2)));
                LOAD4_fp16(data3, src, (tensor3D_offset_fp16(src, x1, y, 0) >> uint(2)) + uint(2));

#if defined(POOL_L2)
                // Raise to power of 2 for L2 Pooling
                data2 *= data2;
                data3 *= data3;
#endif /* defined(POOL_L2) */

                POOL_OP(vdata01, vdata01, data2);
                POOL_OP(vdata11, vdata11, data3);
            }

            // Leftover
            for(; x1 < int(POOL_SIZE + STRIDE_X); x1 = x1 + 2)
            {
                vec2 data4middle;
                data4middle = load_and_unpack(src, (tensor3D_offset_fp16(src, x1, y, 0) >> uint(2)));
#if defined(POOL_L2)
                // Raise to power of 2 for L2 Pooling
                data4middle *= data4middle;
#endif /* defined(POOL_L2) */
                if((x1 + 1) >= int(POOL_SIZE + STRIDE_X))
                {
                    POOL_OP_float(sdata.y, sdata.y, data4middle.x);
                }
                else
                {
                    float data4;
                    POOL_OP_float(data4, data4middle.x, data4middle.y);
                    POOL_OP_float(sdata.y, sdata.y, data4);
                }
            }
        }
        else
        {
            vec2 dataorigin2;
            dataorigin2 = load_and_unpack(src, (tensor3D_offset_fp16(src, (STRIDE_X - 1), y, 0) >> uint(2)));
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            dataorigin2.y *= dataorigin2.y;
#endif /* defined(POOL_L2) */
            POOL_OP_float(sdata.y, sdata.y, dataorigin2.y);

            int x1 = STRIDE_X + 1;
            for(; x1 <= (int(POOL_SIZE + STRIDE_X) - 8); x1 += 8)
            {
                vec4 data2;
                vec4 data3;
                LOAD4_fp16(data2, src, (tensor3D_offset_fp16(src, x1, y, 0) >> uint(2)));
                LOAD4_fp16(data3, src, (tensor3D_offset_fp16(src, x1, y, 0) >> uint(2)) + uint(2));

#if defined(POOL_L2)
                // Raise to power of 2 for L2 Pooling
                data2 *= data2;
                data3 *= data3;
#endif /* defined(POOL_L2) */

                POOL_OP(vdata01, vdata01, data2);
                POOL_OP(vdata11, vdata11, data3);
            }

            // Leftover
            for(; x1 < int(POOL_SIZE + STRIDE_X); x1 = x1 + 2)
            {
                vec2 data4middle;
                data4middle = load_and_unpack(src, (tensor3D_offset_fp16(src, x1, y, 0) >> uint(2)));
#if defined(POOL_L2)
                // Raise to power of 2 for L2 Pooling
                data4middle *= data4middle;
#endif /* defined(POOL_L2) */
                if((x1 + 1) >= int(POOL_SIZE + STRIDE_X))
                {
                    POOL_OP_float(sdata.y, sdata.y, data4middle.x);
                }
                else
                {
                    float data4;
                    POOL_OP_float(data4, data4middle.x, data4middle.y);
                    POOL_OP_float(sdata.y, sdata.y, data4);
                }
            }
        }
    }

    //Reduce result
    vec4 reduce40;
    POOL_OP(reduce40, vdata00.xyzw, vdata10.xyzw);
    vec2 reduce20;
    POOL_OP_vec2(reduce20, reduce40.xy, reduce40.zw);
    vec4 reduce41;
    POOL_OP(reduce41, vdata01.xyzw, vdata11.xyzw);
    vec2 reduce21;
    POOL_OP_vec2(reduce21, reduce41.xy, reduce41.zw);
    vec2 data;
    POOL_OP_float(data.x, reduce20.x, reduce20.y);
    POOL_OP_float(data.x, data.x, sdata.x);
    POOL_OP_float(data.y, reduce21.x, reduce21.y);
    POOL_OP_float(data.y, data.y, sdata.y);

#if defined(POOL_AVG) || defined(POOL_L2)
    {
        // Divide by pool region in case of average pooling
        int start_x1 = (2 * int(gl_GlobalInvocationID.x)) * STRIDE_X - PAD_X;
        int start_y1 = int(gl_GlobalInvocationID.y) * STRIDE_Y - PAD_Y;
        int end_x1   = int(min(start_x1 + POOL_SIZE, MAX_WIDTH));
        int end_y1   = int(min(start_y1 + POOL_SIZE, MAX_HEIGHT));
        int start_x2 = start_x1 + STRIDE_X;
        int start_y2 = start_y1;
        int end_x2   = int(min(start_x2 + POOL_SIZE, MAX_WIDTH));
        int end_y2   = int(min(start_y2 + POOL_SIZE, MAX_HEIGHT));
#if defined(EXCLUDE_PADDING)
        start_x1     = max(0, start_x1);
        start_y1     = max(0, start_y1);
        start_x2     = max(0, start_x2);
        start_y2     = max(0, start_y2);
#endif /* defined(EXCLUDE_PADDING) */
        vec2 res1;
        res1.x = float((end_y1 - start_y1) * (end_x1 - start_x1));
        res1.y = float((end_y2 - start_y2) * (end_x2 - start_x2));
        data.x = DIV_OP(data.x, res1.x);
        data.y = DIV_OP(data.y, res1.y);
    }
#endif /* defined(POOL_AVG) || defined(POOL_L2) */

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    data = SQRT_OP(data);
#endif /* defined(POOL_L2) */
    uint res;
    res = uint(packHalf2x16(data));

    // Store result
    STORE1(dst, CURRENT_OFFSET(dst) >> uint(2), res);
}
#endif /*POOLING_LAYER_2*/
#endif /*DATA_TYPE_FP32 */
