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

/** Performs a pooling function
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note The pool size must be passed at compile time using "#define POOLING_LAYER_n". e.g. "#define POOLING_LAYER_2"
 *       n must be one of these: 2, 3, 7, N
 *       Pool size must be passed using POOL_SIZE if POOLING_LAYER_N is defined. e.g. POOL_SIZE=13;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr   Pointer to the source image. Supported data types: F32/F16
 * @param[in]  src_attrs The attributes of the source image
 * @param[out] dst_ptr   Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  src_attrs The attributes of the destination image
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
};

// Common definitions
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

#if defined(DATA_TYPE_FP32)

float calculate_max(const int, Tensor3DIterator, const int, const int, const int, const int, const int, const int);
float calculate_avg(const int, Tensor3DIterator, const int, const int, const int, const int, const int, const int);

TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

#if defined(POOL_SIZE)
// Set the initial value for the pooling operation accordingly with the data type
#if defined(POOL_AVG) || defined(POOL_L2)
#define INITIAL_VALUE 0.0f
#else /* defined(POOL_AVG) || defined(POOL_L2) */
#define INITIAL_VALUE -3.402823466385289e+38
#endif // POOL_AVG
#endif //POOL_SIZE

float calculate_max(const int pool_size, Tensor3DIterator src_iter, const int upper_bound_w, const int upper_bound_h, const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int start_x = int(gl_GlobalInvocationID.x) * stride_x - pad_x;
    int start_y = int(gl_GlobalInvocationID.y) * stride_y - pad_y;
    int end_x   = int(min(start_x + pool_size, upper_bound_w));
    int end_y   = int(min(start_y + pool_size, upper_bound_h));

    float data_max;
    data_max = LOAD_CURRENT_ITEM(src_ptr, src_iter);

    for(int i = 0; (start_y + i) < end_y; ++i)
    {
        for(int j = 0; (start_x + j) < end_x; ++j)
        {
            float data = LOAD(src_ptr, TENSOR3D_OFFSET(src_iter, j, i, 0));
            POOL_OP_float(data_max, data_max, data);
        }
    }

    return data_max;
}

float calculate_avg(const int pool_size, Tensor3DIterator src_iter, const int upper_bound_w, const int upper_bound_h, const int pad_x, const int pad_y, const int stride_x, const int stride_y)
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
            float data = LOAD(src_ptr, TENSOR3D_OFFSET(src_iter, i, j, 0));
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

#if defined(POOLING_LAYER_2) || defined(POOLING_LAYER_3) || defined(POOLING_LAYER_7)

#if defined(POOLING_LAYER_2)
#define POOL_SIZE 2
#elif defined(POOLING_LAYER_3)
#define POOL_SIZE 3
#elif defined(POOLING_LAYER_7)
#define POOL_SIZE 7
#else // POOLING_LAYER_n
#error Please define POOLING_LAYER_N instead.
#endif // POOLING_LAYER_n

void main(void)
{
    // Get pixels pointer
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    //Load and calculate data
    float res;
#if defined(POOL_AVG) || defined(POOL_L2)
    res = calculate_avg(POOL_SIZE, src_iter, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#else  /*POOL_AVG*/
    res = calculate_max(POOL_SIZE, src_iter, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    res = SQRT_OP(res);
#endif /* defined(POOL_L2) */

    // Store result
    STORE_CURRENT_ITEM(dst_ptr, dst_iter, res);
}

#elif defined(POOLING_LAYER_3_OPTIMIZED)

#define POOLING3x3_STRIDE1(res, input_ptr, input_iter)                                                             \
    vec4 data00 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0));                                   \
    vec2 data01 = VLOAD2(vec2, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(4));                         \
    vec4 data10 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0));                                   \
    vec2 data11 = VLOAD2(vec2, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(4));                         \
    vec4 data20 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0));                                   \
    vec2 data21 = VLOAD2(vec2, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(4));                         \
    data00      = POW2_OP(data00, 4);                                                                              \
    data01      = POW2_OP(data01, 2);                                                                              \
    data10      = POW2_OP(data10, 4);                                                                              \
    data11      = POW2_OP(data11, 2);                                                                              \
    data20      = POW2_OP(data20, 4);                                                                              \
    data21      = POW2_OP(data21, 2);                                                                              \
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

#define POOLING3x3_STRIDE2(res, input_ptr, input_iter)                                                             \
    vec4  data000 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0));                                 \
    vec4  data001 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(4));                       \
    float data010 = LOAD(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(8));                               \
    vec4  data100 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0));                                 \
    vec4  data101 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(4));                       \
    float data11  = LOAD(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(8));                               \
    vec4  data200 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0));                                 \
    vec4  data201 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(4));                       \
    float data21  = LOAD(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(8));                               \
    data000       = POW2_OP(data000, 4);                                                                           \
    data001       = POW2_OP(data001, 4);                                                                           \
    data010       = POW2_OP(data010, 1);                                                                           \
    data100       = POW2_OP(data100, 4);                                                                           \
    data101       = POW2_OP(data101, 4);                                                                           \
    data11        = POW2_OP(data11, 1);                                                                            \
    data200       = POW2_OP(data200, 4);                                                                           \
    data201       = POW2_OP(data201, 4);                                                                           \
    data21        = POW2_OP(data21, 1);                                                                            \
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

#define POOLING3x3_STRIDE3(res, input_ptr, input_iter)                                                 \
    vec4 data000 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0));                      \
    vec4 data001 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(4));            \
    vec4 data010 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(8));            \
    vec4 data100 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0));                      \
    vec4 data101 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(4));            \
    vec4 data11  = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(8));            \
    vec4 data200 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0));                      \
    vec4 data201 = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(4));            \
    vec4 data21  = VLOAD4(vec4, input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(8));            \
    data000      = POW2_OP(data000, 4);                                                                \
    data001      = POW2_OP(data001, 4);                                                                \
    data010      = POW2_OP(data010, 4);                                                                \
    data100      = POW2_OP(data100, 4);                                                                \
    data101      = POW2_OP(data101, 4);                                                                \
    data11       = POW2_OP(data11, 4);                                                                 \
    data200      = POW2_OP(data200, 4);                                                                \
    data201      = POW2_OP(data201, 4);                                                                \
    data21       = POW2_OP(data21, 4);                                                                 \
    \
    POOL_OP(data000.xyzw, data000.xyzw, data100.xyzw);                                                 \
    POOL_OP(data001.xyzw, data001.xyzw, data101.xyzw);                                                 \
    POOL_OP(data010.xyzw, data010.xyzw, data11.xyzw);                                                  \
    POOL_OP(data000.xyzw, data000.xyzw, data200.xyzw);                                                 \
    POOL_OP(data001.xyzw, data001.xyzw, data201.xyzw);                                                 \
    POOL_OP(data010.xyzw, data010.xyzw, data21.xyzw);                                                  \
    POOL_OP(res.xyzw, vec4(data000.xw, data001.z, data010.y), vec4(data000.y, data001.xw, data010.z)); \
    POOL_OP(res.xyzw, res.xyzw, vec4(data000.z, data001.y, data010.xw))

void main(void)
{
    // Get pixels pointer
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    vec4 res;
    // Perform pooling 3x3 for 4 output elements
#if STRIDE_X == 1
    POOLING3x3_STRIDE1(res, src_ptr, src_iter);
#elif STRIDE_X == 2
    POOLING3x3_STRIDE2(res, src_ptr, src_iter);
#elif STRIDE_X == 3
    POOLING3x3_STRIDE3(res, src_ptr, src_iter);
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

    VSTORE4_CURRENT_ITEM(dst_ptr, dst_iter, res);
}

#elif defined(POOLING_LAYER_N)

void main(void)
{
    // Get pixels pointer
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    vec4  vdata0 = vec4(INITIAL_VALUE);
    vec4  vdata1 = vec4(INITIAL_VALUE);
    float sdata  = float(INITIAL_VALUE);

    for(int y = 0; y < int(POOL_SIZE); y++)
    {
        int x = 0;
        for(; x <= (int(POOL_SIZE) - 8); x += 8)
        {
            vec4 data2 = VLOAD4(vec4, src_ptr, TENSOR3D_OFFSET(src_iter, x, y, 0));
            vec4 data3 = VLOAD4(vec4, src_ptr, TENSOR3D_OFFSET(src_iter, x, y, 0) + uint(4));

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
            float data4 = LOAD(src_ptr, TENSOR3D_OFFSET(src_iter, x, y, 0));
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
    STORE_CURRENT_ITEM(dst_ptr, dst_iter, res);
}
#endif // POOLING_LAYER_N

#elif defined(DATA_TYPE_FP16)

vec2 calculate_max(const int, Tensor3DIterator, const int, const int, const int, const int, const int, const int);
vec2 calculate_avg(const int, Tensor3DIterator, const int, const int, const int, const int, const int, const int);

TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);

#if defined(POOL_SIZE)
// Set the initial value for the pooling operation accordingly with the data type
#if defined(POOL_AVG) || defined(POOL_L2)
#define INITIAL_VALUE 0.0f
#else /* defined(POOL_AVG) || defined(POOL_L2) */
#define INITIAL_VALUE -65504.0f
#endif //POOL_AVG
#endif //POOL_SIZE

vec2 calculate_max(const int pool_size, Tensor3DIterator src_iter, const int upper_bound_w, const int upper_bound_h, const int pad_x, const int pad_y, const int stride_x, const int stride_y)
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
    vec2 data_init1 = LOAD_UNPACK2_CURRENT_ITEM_HALF(src_ptr, src_iter);
    data_max.x      = data_init1.x;

    //Load and Set initial maximum2
    if(end_x1 < upper_bound_w)
    {
        if((stride_x % 2) == 0)
        {
            vec2 data_init2 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, stride_x, 0, 0));
            data_max.y      = data_init2.x;
        }
        else
        {
            vec2 data_init2 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, stride_x - 1, 0, 0));
            data_max.y      = data_init2.y;
        }
    }

    for(int i = 0; (start_y1 + i) < end_y1; i++)
        for(int j = 0; (start_x1 + j) < end_x1; j = j + 2)
        {
            //Calculate maximum1
            if((start_x1 + j + 1) < end_x1)
            {
                vec2  data1 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, j, i, 0));
                float data_mr1;
                POOL_OP_float(data_mr1, data1.x, data1.y);
                POOL_OP_float(data_max.x, data_max.x, data_mr1);
            }
            else
            {
                vec2 data1 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, j, i, 0));
                POOL_OP_float(data_max.x, data_max.x, data1.x);
            }

            //Calculate maximum2
            if((start_x2 + j) < end_x2 && end_x1 < upper_bound_w)
            {
                if((stride_x % 2) == 0)
                {
                    vec2 data2 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, (j + stride_x), i, 0));

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
                    vec2 data2 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, (j + stride_x - 1), i, 0));
                    vec2 data3 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, (j + stride_x + 1), i, 0));
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

vec2 calculate_avg(const int pool_size, Tensor3DIterator src_iter, const int upper_bound_w, const int upper_bound_h, const int pad_x, const int pad_y, const int stride_x, const int stride_y)
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
            vec2 data1 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, j, i, 0));
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
                    vec2 data2 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, (j + stride_x), i, 0));
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
                    vec2 data2 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, (j + stride_x - 1), i, 0));
                    vec2 data3 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, (j + stride_x + 1), i, 0));
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

#if defined(POOLING_LAYER_2) || defined(POOLING_LAYER_3) || defined(POOLING_LAYER_7)

#if defined(POOLING_LAYER_2)
#define POOL_SIZE 2
#elif defined(POOLING_LAYER_3)
#define POOL_SIZE 3
#elif defined(POOLING_LAYER_7)
#define POOL_SIZE 7
#else // POOLING_LAYER_n
#error Please define POOLING_LAYER_N instead.
#endif // POOLING_LAYER_n

void main(void)
{
    // Get pixels pointer
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    //Load and calculate data
    vec2 data;
#if defined(POOL_AVG) || defined(POOL_L2)
    data = calculate_avg(POOL_SIZE, src_iter, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#else  /*POOL_AVG*/
    data = calculate_max(POOL_SIZE, src_iter, MAX_WIDTH, MAX_HEIGHT, PAD_X, PAD_Y, STRIDE_X, STRIDE_Y);
#endif /*POOL_AVG*/

#if defined(POOL_L2)
    // Take square root of the result in L2 pooling
    data = SQRT_OP(data);
#endif /* defined(POOL_L2) */

    // Store result
    STORE_PACK2_CURRENT_ITEM_HALF(dst_ptr, dst_iter, data);
}

#elif defined(POOLING_LAYER_3_OPTIMIZED)

#define POOLING3x3_STRIDE1_fp16(res, input_ptr, input_iter)                                                        \
    vec4 data00 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0));                            \
    vec2 data01 = LOAD_UNPACK2_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(2));                    \
    vec4 data10 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0));                            \
    vec2 data11 = LOAD_UNPACK2_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(2));                    \
    vec4 data20 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0));                            \
    vec2 data21 = LOAD_UNPACK2_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(2));                    \
    data00      = POW2_OP(data00, 4);                                                                              \
    data01      = POW2_OP(data01, 2);                                                                              \
    data10      = POW2_OP(data10, 4);                                                                              \
    data11      = POW2_OP(data11, 2);                                                                              \
    data20      = POW2_OP(data20, 4);                                                                              \
    data21      = POW2_OP(data21, 2);                                                                              \
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

#define POOLING3x3_STRIDE2_fp16(res, input_ptr, input_iter)                                                        \
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
    data000     = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0));                            \
    data001     = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(2));                  \
    datamiddle0 = LOAD_UNPACK2_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(4));                    \
    data010     = datamiddle0.x;                                                                                   \
    data100     = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0));                            \
    data101     = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(2));                  \
    datamiddle1 = LOAD_UNPACK2_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(4));                    \
    data11      = datamiddle1.x;                                                                                   \
    data200     = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0));                            \
    data201     = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(2));                  \
    datamiddle2 = LOAD_UNPACK2_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(4));                    \
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

#define POOLING3x3_STRIDE3_fp16(res, input_ptr, input_iter)                                            \
    vec4 data000 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0));               \
    vec4 data001 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(2));     \
    vec4 data010 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 0, 0) + uint(4));     \
    vec4 data100 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0));               \
    vec4 data101 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(2));     \
    vec4 data11  = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 1, 0) + uint(4));     \
    vec4 data200 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0));               \
    vec4 data201 = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(2));     \
    vec4 data21  = VLOAD2_UNPACK4_HALF(input_ptr, TENSOR3D_OFFSET(input_iter, 0, 2, 0) + uint(4));     \
    data000      = POW2_OP(data000, 4);                                                                \
    data001      = POW2_OP(data001, 4);                                                                \
    data010      = POW2_OP(data010, 4);                                                                \
    data100      = POW2_OP(data100, 4);                                                                \
    data101      = POW2_OP(data101, 4);                                                                \
    data11       = POW2_OP(data11, 4);                                                                 \
    data200      = POW2_OP(data200, 4);                                                                \
    data201      = POW2_OP(data201, 4);                                                                \
    data21       = POW2_OP(data21, 4);                                                                 \
    \
    POOL_OP(data000.xyzw, data000.xyzw, data100.xyzw);                                                 \
    POOL_OP(data001.xyzw, data001.xyzw, data101.xyzw);                                                 \
    POOL_OP(data010.xyzw, data010.xyzw, data11.xyzw);                                                  \
    POOL_OP(data000.xyzw, data000.xyzw, data200.xyzw);                                                 \
    POOL_OP(data001.xyzw, data001.xyzw, data201.xyzw);                                                 \
    POOL_OP(data010.xyzw, data010.xyzw, data21.xyzw);                                                  \
    POOL_OP(res.xyzw, vec4(data000.xw, data001.z, data010.y), vec4(data000.y, data001.xw, data010.z)); \
    POOL_OP(res.xyzw, res.xyzw, vec4(data000.z, data001.y, data010.xw))

void main(void)
{
    // Get pixels pointer
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    vec4 res;
    // Perform pooling 3x3 for 4 output elements
#if STRIDE_X == 1
    POOLING3x3_STRIDE1_fp16(res, src_ptr, src_iter);
#elif STRIDE_X == 2
    POOLING3x3_STRIDE2_fp16(res, src_ptr, src_iter);
#elif STRIDE_X == 3
    POOLING3x3_STRIDE3_fp16(res, src_ptr, src_iter);
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

    VSTORE2_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, res);
}

#elif defined(POOLING_LAYER_N)

void main(void)
{
    // Get pixels pointer
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    vec4 vdata00 = vec4(INITIAL_VALUE);
    vec4 vdata01 = vec4(INITIAL_VALUE);
    vec4 vdata10 = vec4(INITIAL_VALUE);
    vec4 vdata11 = vec4(INITIAL_VALUE);
    vec2 sdata   = vec2(INITIAL_VALUE);

    for(int y = 0; y < int(POOL_SIZE); y++)
    {
        int x = 0;
        for(; x <= (int(POOL_SIZE) - 8); x += 8)
        {
            vec4 data2 = VLOAD2_UNPACK4_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x, y, 0));
            vec4 data3 = VLOAD2_UNPACK4_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x, y, 0) + uint(2));

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
            vec2 data4middle = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x, y, 0));
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
                vec4 data2 = VLOAD2_UNPACK4_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x1, y, 0));
                vec4 data3 = VLOAD2_UNPACK4_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x1, y, 0) + uint(2));

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
                data4middle = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x1, y, 0));
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
            dataorigin2 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, (STRIDE_X - 1), y, 0));
#if defined(POOL_L2)
            // Raise to power of 2 for L2 Pooling
            dataorigin2.y *= dataorigin2.y;
#endif /* defined(POOL_L2) */
            POOL_OP_float(sdata.y, sdata.y, dataorigin2.y);

            int x1 = STRIDE_X + 1;
            for(; x1 <= (int(POOL_SIZE + STRIDE_X) - 8); x1 += 8)
            {
                vec4 data2 = VLOAD2_UNPACK4_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x1, y, 0));
                vec4 data3 = VLOAD2_UNPACK4_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x1, y, 0) + uint(2));

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
                vec2 data4middle = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x1, y, 0));
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

    // Store result
    STORE_PACK2_CURRENT_ITEM_HALF(dst_ptr, dst_iter, data);
}
#endif // POOLING_LAYER_N

#else // DATA_TYPE_FP32
#error Data type not supported
#endif // DATA_TYPE_FP32
