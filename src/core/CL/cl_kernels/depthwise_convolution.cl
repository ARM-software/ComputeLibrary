/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#if defined(DEPTH_MULTIPLIER)
#if defined(CONV_STRIDE_X)

#if CONV_STRIDE_X == 1
#define convolution1x3 convolution1x3_stride_1
#elif CONV_STRIDE_X == 2
#define convolution1x3 convolution1x3_stride_2
#elif CONV_STRIDE_X == 3
#define convolution1x3 convolution1x3_stride_3
#else /* CONV_STRIDE_X */
#error "Stride not supported"
#endif /* CONV_STRIDE_X */

/** Compute a 1D horizontal convolution of size 3 and stride 1 for floating point type.
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a float2 containing 2 convoluted values.
 */
inline float2 convolution1x3_stride_1(__global const uchar *left_pixel,
                                      const float           left_coeff,
                                      const float           middle_coeff,
                                      const float           right_coeff)
{
    float4 temp = vload4(0, (__global float *)left_pixel);

    float2 left   = CONVERT(temp.s01, float2);
    float2 middle = CONVERT(temp.s12, float2);
    float2 right  = CONVERT(temp.s23, float2);

    return left * (float2)left_coeff + middle * (float2)middle_coeff + right * (float2)right_coeff;
}

/** Compute a 1D horizontal convolution of size 3 and stride 2 for floating point type.
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a float2 containing 2 convoluted values.
 */
inline float2 convolution1x3_stride_2(__global const uchar *left_pixel,
                                      const float           left_coeff,
                                      const float           middle_coeff,
                                      const float           right_coeff)
{
    float4 temp0 = vload4(0, (__global float *)left_pixel);
    float  temp1 = *((__global float *)(left_pixel + 4 * sizeof(float)));

    float2 left   = CONVERT(temp0.s02, float2);
    float2 middle = CONVERT(temp0.s13, float2);
    float2 right  = CONVERT((float2)(temp0.s2, temp1), float2);

    return left * (float2)left_coeff + middle * (float2)middle_coeff + right * (float2)right_coeff;
}

/** Compute a 1D horizontal convolution of size 3 and stride 3 for floating point type.
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a float2 containing 2 convoluted values.
 */
inline float2 convolution1x3_stride_3(__global const uchar *left_pixel,
                                      const float           left_coeff,
                                      const float           middle_coeff,
                                      const float           right_coeff)
{
    float4 temp0 = vload4(0, (__global float *)left_pixel);
    float2 temp1 = vload2(0, (__global float *)(left_pixel + 4 * sizeof(float)));

    float2 left   = CONVERT(temp0.s03, float2);
    float2 middle = CONVERT((float2)(temp0.s1, temp1.s0), float2);
    float2 right  = CONVERT((float2)(temp0.s2, temp1.s1), float2);

    return left * (float2)left_coeff + middle * (float2)middle_coeff + right * (float2)right_coeff;
}

/** Apply a 3x3 convolution matrix to a single channel F32 input image and return the result.
 *
 * Convolution matrix layout:
 *
 * [ mat0, mat1, mat2 ]\n
 * [ mat3, mat4, mat5 ]\n
 * [ mat6, mat7, mat8 ]\n
 *
 * @param[in] src  A pointer to source Image structure
 * @param[in] mat0 Coefficient from the convolution matrix
 * @param[in] mat1 Coefficient from the convolution matrix
 * @param[in] mat2 Coefficient from the convolution matrix
 * @param[in] mat3 Coefficient from the convolution matrix
 * @param[in] mat4 Coefficient from the convolution matrix
 * @param[in] mat5 Coefficient from the convolution matrix
 * @param[in] mat6 Coefficient from the convolution matrix
 * @param[in] mat0 Coefficient from the convolution matrix
 * @param[in] mat7 Coefficient from the convolution matrix
 * @param[in] mat8 Coefficient from the convolution matrix
 *
 * @return a float2 containing 2 convoluted values.
 */
inline float2 convolution3x3(
    Image      *src,
    const float mat0, const float mat1, const float mat2,
    const float mat3, const float mat4, const float mat5,
    const float mat6, const float mat7, const float mat8)
{
    float2 pixels;

    pixels = convolution1x3(offset(src, 0, 0), mat0, mat1, mat2);
    pixels += convolution1x3(offset(src, 0, 1), mat3, mat4, mat5);
    pixels += convolution1x3(offset(src, 0, 2), mat6, mat7, mat8);

    return pixels;
}

/** This OpenCL kernel computes the depthwise convolution 3x3
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: F32
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: F32
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: F32
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the biases vector
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: F16/F32
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */
__kernel void depthwise_convolution_3x3(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT(weights);
#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif //defined(HAS_BIAS)

    src.ptr -= (get_global_id(2) - get_global_id(2) / DEPTH_MULTIPLIER) * src_step_z;

    uchar3 offset          = (uchar3)(0, 1, 2) * (uchar3)weights_stride_y;
    float3 weights_values0 = vload3(0, (__global float *)(weights.ptr + offset.s0));
    float3 weights_values1 = vload3(0, (__global float *)(weights.ptr + offset.s1));
    float3 weights_values2 = vload3(0, (__global float *)(weights.ptr + offset.s2));

    float2 pixels = convolution3x3(&src, weights_values0.s0, weights_values0.s1, weights_values0.s2,
                                   weights_values1.s0, weights_values1.s1, weights_values1.s2,
                                   weights_values2.s0, weights_values2.s1, weights_values2.s2);
#if defined(HAS_BIAS)
    pixels += (float2)(*((__global float *)(biases.ptr + get_global_id(2) * biases_stride_x)));
#endif //defined(HAS_BIAS)

    vstore2(pixels, 0, (__global float *)dst.ptr);
}
#endif //defined(CONV_STRIDE_X)

#define CONVOLUTION1x3_BIFROST2X1_STRIDE1(acc, src0, weights_row0) \
    ({                                                             \
        acc.s0 = fma(src0.s0, weights_row0.s0, acc.s0);            \
        acc.s0 = fma(src0.s1, weights_row0.s1, acc.s0);            \
        acc.s0 = fma(src0.s2, weights_row0.s2, acc.s0);            \
        acc.s1 = fma(src0.s1, weights_row0.s0, acc.s1);            \
        acc.s1 = fma(src0.s2, weights_row0.s1, acc.s1);            \
        acc.s1 = fma(src0.s3, weights_row0.s2, acc.s1);            \
    })

#define CONVOLUTION1x3_BIFROST4X1_STRIDE1(acc, src0, weights_row0) \
    ({                                                             \
        acc.s0 = fma(src0.s0, weights_row0.s0, acc.s0);            \
        acc.s0 = fma(src0.s1, weights_row0.s1, acc.s0);            \
        acc.s0 = fma(src0.s2, weights_row0.s2, acc.s0);            \
        acc.s1 = fma(src0.s1, weights_row0.s0, acc.s1);            \
        acc.s1 = fma(src0.s2, weights_row0.s1, acc.s1);            \
        acc.s1 = fma(src0.s3, weights_row0.s2, acc.s1);            \
        acc.s2 = fma(src0.s2, weights_row0.s0, acc.s2);            \
        acc.s2 = fma(src0.s3, weights_row0.s1, acc.s2);            \
        acc.s2 = fma(src0.s4, weights_row0.s2, acc.s2);            \
        acc.s3 = fma(src0.s3, weights_row0.s0, acc.s3);            \
        acc.s3 = fma(src0.s4, weights_row0.s1, acc.s3);            \
        acc.s3 = fma(src0.s5, weights_row0.s2, acc.s3);            \
    })

#define CONVOLUTION1x3_BIFROST2X1_STRIDE2(acc, src0, src1, weights_row0) \
    ({                                                                   \
        acc.s0 = fma(src0.s0, weights_row0.s0, acc.s0);                  \
        acc.s0 = fma(src0.s1, weights_row0.s1, acc.s0);                  \
        acc.s0 = fma(src0.s2, weights_row0.s2, acc.s0);                  \
        acc.s1 = fma(src0.s2, weights_row0.s0, acc.s1);                  \
        acc.s1 = fma(src0.s3, weights_row0.s1, acc.s1);                  \
        acc.s1 = fma(src1.s0, weights_row0.s2, acc.s1);                  \
    })

#define CONVOLUTION1x3_BIFROST4X1_STRIDE2(acc, src0, src1, weights_row0) \
    ({                                                                   \
        acc.s0 = fma(src0.s0, weights_row0.s0, acc.s0);                  \
        acc.s0 = fma(src0.s1, weights_row0.s1, acc.s0);                  \
        acc.s0 = fma(src0.s2, weights_row0.s2, acc.s0);                  \
        acc.s1 = fma(src0.s2, weights_row0.s0, acc.s1);                  \
        acc.s1 = fma(src0.s3, weights_row0.s1, acc.s1);                  \
        acc.s1 = fma(src0.s4, weights_row0.s2, acc.s1);                  \
        acc.s2 = fma(src0.s4, weights_row0.s0, acc.s2);                  \
        acc.s2 = fma(src0.s5, weights_row0.s1, acc.s2);                  \
        acc.s2 = fma(src0.s6, weights_row0.s2, acc.s2);                  \
        acc.s3 = fma(src0.s6, weights_row0.s0, acc.s3);                  \
        acc.s3 = fma(src0.s7, weights_row0.s1, acc.s3);                  \
        acc.s3 = fma(src1.s0, weights_row0.s2, acc.s3);                  \
    })

/** This OpenCL kernel is optimized for Bifrost architectures and computes the depthwise convolution 3x3 when both
 * stride_x and stride_y are equal to 1
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: F32
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: F32
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: F32
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the biases vector
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: F32
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */
__kernel void depthwise_convolution_3x3_stridex1_stridey1_bifrost_f32(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT(weights);

    float2 pixels0 = 0.0f;
    float2 pixels1 = 0.0f;
    float2 pixels2 = 0.0f;
    float2 pixels3 = 0.0f;

    __global uchar *weights_addr = (__global uchar *)weights.ptr;
    __global uchar *src_addr     = src.ptr - (get_global_id(2) - get_global_id(2) / DEPTH_MULTIPLIER) * src_step_z;

    // Load the weights
    float3 weights_row0 = vload3(0, (__global float *)(weights_addr + 0 * weights_stride_y));
    float3 weights_row1 = vload3(0, (__global float *)(weights_addr + 1 * weights_stride_y));
    float3 weights_row2 = vload3(0, (__global float *)(weights_addr + 2 * weights_stride_y));

    // Note: Since each work-item computes 4x2 elements, we need to load 6 rows from the input tensor
    float4 src00 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y)); // Row0
    float4 src10 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y)); // Row1
    float4 src20 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y)); // Row2
    float4 src30 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y)); // Row3
    float4 src40 = vload4(0, (__global float *)(src_addr + 4 * src_stride_y)); // Row4
    float4 src50 = vload4(0, (__global float *)(src_addr + 5 * src_stride_y)); // Row5

    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels0, src00, weights_row0);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels0, src10, weights_row1);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels0, src20, weights_row2);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels1, src10, weights_row0);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels1, src20, weights_row1);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels1, src30, weights_row2);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels2, src20, weights_row0);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels2, src30, weights_row1);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels2, src40, weights_row2);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels3, src30, weights_row0);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels3, src40, weights_row1);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels3, src50, weights_row2);

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    float bias = *((__global float *)(vector_offset(&biases, get_global_id(2))));

    pixels0 += (float2)bias;
    pixels1 += (float2)bias;
    pixels2 += (float2)bias;
    pixels3 += (float2)bias;
#endif /* defined(HAS_BIAS) */

    vstore2(pixels0, 0, (__global float *)(dst.ptr + 0 * dst_stride_y));
    vstore2(pixels1, 0, (__global float *)(dst.ptr + 1 * dst_stride_y));
    vstore2(pixels2, 0, (__global float *)(dst.ptr + 2 * dst_stride_y));
    vstore2(pixels3, 0, (__global float *)(dst.ptr + 3 * dst_stride_y));
}

/** This OpenCL kernel is optimized for Bifrost architectures and computes the depthwise convolution 3x3 when both
 * stride_x and stride_y are equal to 2
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: F32
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: F32
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: F32
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the biases vector
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: F32
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */
__kernel void depthwise_convolution_3x3_stridex2_stridey2_bifrost_f32(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT(weights);

    float2 pixels0 = 0.0f;
    float2 pixels1 = 0.0f;

    __global uchar *weights_addr = (__global uchar *)weights.ptr;
    __global uchar *src_addr     = src.ptr - (get_global_id(2) - get_global_id(2) / DEPTH_MULTIPLIER) * src_step_z;

    // Load the weights
    float3 weights_row0 = vload3(0, (__global float *)(weights_addr + 0 * weights_stride_y));
    float3 weights_row1 = vload3(0, (__global float *)(weights_addr + 1 * weights_stride_y));
    float3 weights_row2 = vload3(0, (__global float *)(weights_addr + 2 * weights_stride_y));

    // Note: Since each work-item computes 4x2 elements, we need to load 5 rows from the input tensor
    float4 src00 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y)); // Row0
    float2 src01 = vload2(2, (__global float *)(src_addr + 0 * src_stride_y)); // Row0
    float4 src10 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y)); // Row1
    float2 src11 = vload2(2, (__global float *)(src_addr + 1 * src_stride_y)); // Row1
    float4 src20 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y)); // Row2
    float2 src21 = vload2(2, (__global float *)(src_addr + 2 * src_stride_y)); // Row2
    float4 src30 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y)); // Row3
    float2 src31 = vload2(2, (__global float *)(src_addr + 3 * src_stride_y)); // Row3
    float4 src40 = vload4(0, (__global float *)(src_addr + 4 * src_stride_y)); // Row4
    float2 src41 = vload2(2, (__global float *)(src_addr + 4 * src_stride_y)); // Row4

    CONVOLUTION1x3_BIFROST2X1_STRIDE2(pixels0, src00, src01, weights_row0);
    CONVOLUTION1x3_BIFROST2X1_STRIDE2(pixels0, src10, src11, weights_row1);
    CONVOLUTION1x3_BIFROST2X1_STRIDE2(pixels0, src20, src21, weights_row2);
    CONVOLUTION1x3_BIFROST2X1_STRIDE2(pixels1, src20, src21, weights_row0);
    CONVOLUTION1x3_BIFROST2X1_STRIDE2(pixels1, src30, src31, weights_row1);
    CONVOLUTION1x3_BIFROST2X1_STRIDE2(pixels1, src40, src41, weights_row2);

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    float bias = *((__global float *)(vector_offset(&biases, get_global_id(2))));

    pixels0 += (float2)bias;
    pixels1 += (float2)bias;
#endif /* defined(HAS_BIAS) */

    vstore2(pixels0, 0, (__global float *)(dst.ptr + 0 * dst_stride_y));
    vstore2(pixels1, 0, (__global float *)(dst.ptr + 1 * dst_stride_y));
}

#endif // defined(DEPTH_MULTIPLIER)

#if defined(NCHW)
#define in_stride_x src_stride_x
#define in_stride_y src_stride_y
#define in_stride_z src_stride_z
#define out_stride_x dst_stride_x
#define out_stride_y dst_stride_y
#define out_stride_z dst_stride_z
#else //defined(NCHW)
#define in_stride_x src_stride_y
#define in_stride_y src_stride_z
#define in_stride_z src_stride_x
#define out_stride_x dst_stride_y
#define out_stride_y dst_stride_z
#define out_stride_z dst_stride_x
#endif //defined(NCHW)

#if defined(SRC_WIDTH) && defined(DATA_TYPE)
/** This kernel reshapes each of the tensor's low three dimensions to single rows.
 *
 * @note Datatype and source width should be given as a preprocessor argument using -DDATA_TYPE=type and -DSRC_WIDTH=width. e.g. -DSRC_WIDTH=128
 *
 * @param[in]  src_ptr                              Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                           src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[out] dst_ptr                              Pointer to the destination tensor. Same as @p src_ptr
 * @param[in]  dst_stride_x                         Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                           dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                         Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                           dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes    The offset of the first element in the destination tensor
 * @param[in]  biases_ptr                           (Optional) Pointer to the biases vector. Supported data types: F16/F32
 * @param[in]  biases_stride_x                      (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in]  biases_step_x                        (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes (Optional) The offset of the first element in the biases vector
 */
__kernel void depthwise_weights_reshape(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst)
#ifdef HAS_BIAS
    ,
    VECTOR_DECLARATION(biases)
#endif /* HAS_BIAS */
)
{
#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif /* HAS_BIAS */

    __global uchar *input_ptr  = src_ptr + src_offset_first_element_in_bytes + get_global_id(1) * in_stride_y + get_global_id(2) * in_stride_z;
    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + get_global_id(1) * SRC_WIDTH * dst_stride_x + get_global_id(2) * dst_stride_y;

    for(int i = 0; i < SRC_WIDTH; ++i, input_ptr += in_stride_x)
    {
        *((__global DATA_TYPE *)(output_ptr + i * dst_stride_x)) = *((__global DATA_TYPE *)input_ptr);
    }

#if defined(HAS_BIAS)
    if(get_global_id(1) == 0)
    {
        *((__global DATA_TYPE *)(output_ptr + SRC_WIDTH * get_global_size(1) * dst_stride_x)) = *((__global DATA_TYPE *)(biases.ptr + get_global_id(2) * biases_stride_x));
    }
#endif // defined(HAS_BIAS)
}
#endif //defined(SRC_WIDTH) && defined(DATA_TYPE)

#if defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DATA_TYPE) && defined(PAD_VALUE) && defined(DEPTH_MULTIPLIER)
/** This kernel performs a reshaping of the input tensor to a tensor used to perform depthwise convolution using vector to matrix multiplication.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The convolution information must be passed at compile time using -DSTRIDE_X, -DSTRIDE_Y, -DPAD_LEFT, -DPAD_TOP, -DPAD_RIGHT, -DPAD_BOTTOM, -DKERNEL_WIDHT, -DKERNEL_HEIGHT, -DSRC_WIDTH, -DSRC_HEIGHT, -DDEPTH_MULTIPLIER
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QS8/QS16/F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void depthwise_im2col(TENSOR3D_DECLARATION(src), TENSOR3D_DECLARATION(dst))
{
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    const int src_pixel_linear = get_global_id(1) * STRIDE_X;
    const int full_length      = SRC_WIDTH + PAD_LEFT + PAD_RIGHT;
    const int max_initial_x    = STRIDE_X * (((full_length - KERNEL_WIDTH) / STRIDE_X) + 1);

    const int src_x = -PAD_LEFT + src_pixel_linear % max_initial_x;
    const int src_y = -PAD_TOP + src_pixel_linear / max_initial_x * STRIDE_Y;
    const int src_z = get_global_id(2) / DEPTH_MULTIPLIER;

    __global uchar *input_ptr      = src_ptr + src_offset_first_element_in_bytes + src_z * in_stride_z;
    __global DATA_TYPE *output_ptr = ((__global DATA_TYPE *)(dst.ptr));

    for(int y = src_y; y < src_y + KERNEL_HEIGHT; ++y)
    {
        for(int x = src_x; x < src_x + KERNEL_WIDTH; ++x, ++output_ptr)
        {
            if(x < 0 || x >= SRC_WIDTH || y < 0 || y >= SRC_HEIGHT)
            {
                *output_ptr = PAD_VALUE;
            }
            else
            {
                *output_ptr = *((__global DATA_TYPE *)(input_ptr + x * in_stride_x + y * in_stride_y));
            }
        }
    }
#if defined(HAS_BIAS)
    *output_ptr = (DATA_TYPE)(1);
#endif // defined(HAS_BIAS)
}

#endif //defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(SRC_WIDTH) && defined(DATA_TYPE) && defined(PAD_VALUE) && defined(DEPTH_MULTIPLIER)

#if defined(CONV_WIDTH) && defined(CONV_HEIGHT) && defined(DATA_TYPE)

/** This kernel performs a reshaping of the output of the depthwise generic convolution.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The convolution information must be passed at compile time using -DCONV_WIDTH, -DCONV_HEIGHT, e.g -DCONV_WIDTH=32, -DCONV_HEIGHT=42
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QS8/QS16/F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void depthwise_vector_to_tensor(
    VECTOR_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Vector src = CONVERT_TO_VECTOR_STRUCT(src);

    const int patch_size = CONV_WIDTH * CONV_HEIGHT;
    const int id0        = get_global_id(0);
    const int z          = id0 / patch_size;
    const int index2D    = id0 - z * patch_size;

    __global uchar *out_ptr          = dst_ptr + dst_offset_first_element_in_bytes + index2D % CONV_WIDTH * out_stride_x + index2D / CONV_WIDTH * out_stride_y + z * out_stride_z;
    *((__global DATA_TYPE *)out_ptr) = *((__global DATA_TYPE *)src.ptr);
}

#endif //defined(CONV_WIDTH) && defined(CONV_HEIGHT) && defined(DATA_TYPE)

#if defined(ARM_COMPUTE_OPENCL_FP16_ENABLED) && defined(DEPTH_MULTIPLIER)
#if defined(CONV_STRIDE_X)
#if CONV_STRIDE_X == 1
#define convolution1x3_f16 convolution1x3_stride_1_f16
#elif CONV_STRIDE_X == 2
#define convolution1x3_f16 convolution1x3_stride_2_f16
#elif CONV_STRIDE_X == 3
#define convolution1x3_f16 convolution1x3_stride_3_f16
#else /* CONV_STRIDE_X */
#error "Stride not supported"
#endif /* CONV_STRIDE_X */

/** Compute a 1D horizontal convolution of size 3 and stride 1 for 16bit floating point type.
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a half4 containing 4 convoluted values.
 */
inline half4 convolution1x3_stride_1_f16(__global const uchar *left_pixel,
                                         const half            left_coeff,
                                         const half            middle_coeff,
                                         const half            right_coeff)
{
    half8 temp = vload8(0, (__global half *)left_pixel);

    half4 left   = CONVERT(temp.s0123, half4);
    half4 middle = CONVERT(temp.s1234, half4);
    half4 right  = CONVERT(temp.s2345, half4);

    return left * (half4)left_coeff + middle * (half4)middle_coeff + right * (half4)right_coeff;
}

/** Compute a 1D horizontal convolution of size 3 and stride 2 for 16bit floating point type.
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a half4 containing 4 convoluted values.
 */
inline half4 convolution1x3_stride_2_f16(__global const uchar *left_pixel,
                                         const half            left_coeff,
                                         const half            middle_coeff,
                                         const half            right_coeff)
{
    half8 temp0 = vload8(0, (__global half *)left_pixel);
    half temp1  = *((__global half *)(left_pixel + 8 * sizeof(half)));

    half4 left   = CONVERT(temp0.s0246, half4);
    half4 middle = CONVERT(temp0.s1357, half4);
    half4 right  = CONVERT((half4)(temp0.s246, temp1), half4);

    return left * (half4)left_coeff + middle * (half4)middle_coeff + right * (half4)right_coeff;
}

/** Compute a 1D horizontal convolution of size 3 and stride 3 for 16bit floating point type.
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a half4 containing 4 convoluted values.
 */
inline half4 convolution1x3_stride_3_f16(__global const uchar *left_pixel,
                                         const half            left_coeff,
                                         const half            middle_coeff,
                                         const half            right_coeff)
{
    half16 temp0 = vload16(0, (__global half *)left_pixel);

    half4 left   = CONVERT(temp0.s0369, half4);
    half4 middle = CONVERT(temp0.s147A, half4);
    half4 right  = CONVERT(temp0.s258B, half4);

    return left * (half4)left_coeff + middle * (half4)middle_coeff + right * (half4)right_coeff;
}

/** Apply a 3x3 convolution matrix to a single channel F16 input image and return the result.
 *
 * Convolution matrix layout:
 *
 * [ mat0, mat1, mat2 ]\n
 * [ mat3, mat4, mat5 ]\n
 * [ mat6, mat7, mat8 ]\n
 *
 * @param[in] src  A pointer to source Image structure
 * @param[in] mat0 Coefficient from the convolution matrix
 * @param[in] mat1 Coefficient from the convolution matrix
 * @param[in] mat2 Coefficient from the convolution matrix
 * @param[in] mat3 Coefficient from the convolution matrix
 * @param[in] mat4 Coefficient from the convolution matrix
 * @param[in] mat5 Coefficient from the convolution matrix
 * @param[in] mat6 Coefficient from the convolution matrix
 * @param[in] mat0 Coefficient from the convolution matrix
 * @param[in] mat7 Coefficient from the convolution matrix
 * @param[in] mat8 Coefficient from the convolution matrix
 *
 * @return a half4 containing 4 convoluted values.
 */
inline half4 convolution3x3_f16(
    Image     *src,
    const half mat0, const half mat1, const half mat2,
    const half mat3, const half mat4, const half mat5,
    const half mat6, const half mat7, const half mat8)
{
    half4 pixels;

    pixels = convolution1x3_f16(offset(src, 0, 0), mat0, mat1, mat2);
    pixels += convolution1x3_f16(offset(src, 0, 1), mat3, mat4, mat5);
    pixels += convolution1x3_f16(offset(src, 0, 2), mat6, mat7, mat8);

    return pixels;
}

#if defined(DEPTH_MULTIPLIER)

/** This OpenCL kernel computes the depthwise convolution 3x3
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: F16
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the biases vector
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: F16/F32
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */
__kernel void depthwise_convolution_3x3_f16(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT(weights);
#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif //defined(HAS_BIAS)

    src.ptr -= (get_global_id(2) - get_global_id(2) / DEPTH_MULTIPLIER) * src_step_z;

    uchar3 offset         = (uchar3)(0, 1, 2) * (uchar3)weights_stride_y;
    half3 weights_values0 = vload3(0, (__global half *)(weights.ptr + offset.s0));
    half3 weights_values1 = vload3(0, (__global half *)(weights.ptr + offset.s1));
    half3 weights_values2 = vload3(0, (__global half *)(weights.ptr + offset.s2));

    half4 pixels = convolution3x3_f16(&src, weights_values0.s0, weights_values0.s1, weights_values0.s2,
                                      weights_values1.s0, weights_values1.s1, weights_values1.s2,
                                      weights_values2.s0, weights_values2.s1, weights_values2.s2);
#if defined(HAS_BIAS)
    pixels += (half4)(*((__global half *)(biases.ptr + get_global_id(2) * biases_stride_x)));
#endif //defined(HAS_BIAS)

    vstore4(pixels, 0, (__global half *)dst.ptr);
}
#endif // defined(DEPTH_MULTIPLIER)
#endif // defined(CONV_STRIDE_X)

/** This OpenCL kernel is optimized for Bifrost architectures and computes the 16bit floating point depthwise convolution 3x3
 * when both stride_x and stride_y are equal to 1
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: F16
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the biases vector
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: same as @p src_ptr
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */
__kernel void depthwise_convolution_3x3_stridex1_stridey1_bifrost_f16(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT(weights);

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    half bias = *((__global half *)(vector_offset(&biases, get_global_id(2))));
#endif /* defined(HAS_BIAS) */

    half4 pixels0 = 0.0f;
    half4 pixels1 = 0.0f;
    half4 pixels2 = 0.0f;
    half4 pixels3 = 0.0f;

    __global uchar *weights_addr = (__global uchar *)weights.ptr;
    __global uchar *src_addr     = (__global uchar *)offset(&src, 0, 0) - (get_global_id(2) - get_global_id(2) / DEPTH_MULTIPLIER) * src_step_z;

    // Load the weights
    half3 weights_row0 = vload3(0, (__global half *)(weights_addr + 0 * weights_stride_y));
    half3 weights_row1 = vload3(0, (__global half *)(weights_addr + 1 * weights_stride_y));
    half3 weights_row2 = vload3(0, (__global half *)(weights_addr + 2 * weights_stride_y));

    // Note: Since each work-item computes 4x4 elements, we need to load 6 rows from the input tensor
    half8 src00 = vload8(0, (__global half *)(src_addr + 0 * src_stride_y)); // Row0
    half8 src10 = vload8(0, (__global half *)(src_addr + 1 * src_stride_y)); // Row1
    half8 src20 = vload8(0, (__global half *)(src_addr + 2 * src_stride_y)); // Row2
    half8 src30 = vload8(0, (__global half *)(src_addr + 3 * src_stride_y)); // Row3
    half8 src40 = vload8(0, (__global half *)(src_addr + 4 * src_stride_y)); // Row4
    half8 src50 = vload8(0, (__global half *)(src_addr + 5 * src_stride_y)); // Row5

    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels0, src00, weights_row0);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels0, src10, weights_row1);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels0, src20, weights_row2);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels1, src10, weights_row0);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels1, src20, weights_row1);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels1, src30, weights_row2);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels2, src20, weights_row0);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels2, src30, weights_row1);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels2, src40, weights_row2);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels3, src30, weights_row0);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels3, src40, weights_row1);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels3, src50, weights_row2);

#ifdef HAS_BIAS
    pixels0 += (half4)bias;
    pixels1 += (half4)bias;
    pixels2 += (half4)bias;
    pixels3 += (half4)bias;
#endif /* defined(HAS_BIAS) */

    vstore4(pixels0, 0, (__global half *)(dst.ptr + 0 * dst_stride_y));
    vstore4(pixels1, 0, (__global half *)(dst.ptr + 1 * dst_stride_y));
    vstore4(pixels2, 0, (__global half *)(dst.ptr + 2 * dst_stride_y));
    vstore4(pixels3, 0, (__global half *)(dst.ptr + 3 * dst_stride_y));
}

/** This OpenCL kernel is optimized for Bifrost architectures and computes 16bit floating point the depthwise convolution 3x3
 * when both stride_x and stride_y are equal to 2
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: F16
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the biases vector
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: same as @p src_ptr
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */
__kernel void depthwise_convolution_3x3_stridex2_stridey2_bifrost_f16(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT(weights);

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    half bias = *((__global half *)(vector_offset(&biases, get_global_id(2))));
#endif /* defined(HAS_BIAS) */

    half4 pixels0 = 0.0f;
    half4 pixels1 = 0.0f;

    __global uchar *weights_addr = (__global uchar *)weights.ptr;
    __global uchar *src_addr     = (__global uchar *)offset(&src, 0, 0) - (get_global_id(2) - get_global_id(2) / DEPTH_MULTIPLIER) * src_step_z;

    // Load the weights
    half3 weights_row0 = vload3(0, (__global half *)(weights_addr + 0 * weights_stride_y));
    half3 weights_row1 = vload3(0, (__global half *)(weights_addr + 1 * weights_stride_y));
    half3 weights_row2 = vload3(0, (__global half *)(weights_addr + 2 * weights_stride_y));

    // Note: Since each work-item computes 2x4 elements, we need to load 5 rows from the input tensor
    half8 src00 = vload8(0, (__global half *)(src_addr + 0 * src_stride_y)); // Row0
    half2 src01 = vload2(4, (__global half *)(src_addr + 0 * src_stride_y)); // Row0
    half8 src10 = vload8(0, (__global half *)(src_addr + 1 * src_stride_y)); // Row1
    half2 src11 = vload2(4, (__global half *)(src_addr + 1 * src_stride_y)); // Row1
    half8 src20 = vload8(0, (__global half *)(src_addr + 2 * src_stride_y)); // Row2
    half2 src21 = vload2(4, (__global half *)(src_addr + 2 * src_stride_y)); // Row2
    half8 src30 = vload8(0, (__global half *)(src_addr + 3 * src_stride_y)); // Row3
    half2 src31 = vload2(4, (__global half *)(src_addr + 3 * src_stride_y)); // Row3
    half8 src40 = vload8(0, (__global half *)(src_addr + 4 * src_stride_y)); // Row4
    half2 src41 = vload2(4, (__global half *)(src_addr + 4 * src_stride_y)); // Row4

    CONVOLUTION1x3_BIFROST4X1_STRIDE2(pixels0, src00, src01, weights_row0);
    CONVOLUTION1x3_BIFROST4X1_STRIDE2(pixels0, src10, src11, weights_row1);
    CONVOLUTION1x3_BIFROST4X1_STRIDE2(pixels0, src20, src21, weights_row2);
    CONVOLUTION1x3_BIFROST4X1_STRIDE2(pixels1, src20, src21, weights_row0);
    CONVOLUTION1x3_BIFROST4X1_STRIDE2(pixels1, src30, src31, weights_row1);
    CONVOLUTION1x3_BIFROST4X1_STRIDE2(pixels1, src40, src41, weights_row2);

#ifdef HAS_BIAS
    pixels0 += (half4)bias;
    pixels1 += (half4)bias;
#endif /* defined(HAS_BIAS) */

    vstore4(pixels0, 0, (__global half *)(dst.ptr + 0 * dst_stride_y));
    vstore4(pixels1, 0, (__global half *)(dst.ptr + 1 * dst_stride_y));
}
#endif // defined(ARM_COMPUTE_OPENCL_FP16_ENABLED) && defined(DEPTH_MULTIPLIER)

#if defined(VEC_SIZE) && defined(SRC_DIM_2) && defined(CONV_PAD_TOP) && defined(CONV_PAD_LEFT)

#define VEC_FLOAT VEC_DATA_TYPE(float, VEC_SIZE)

#if defined(CONV_STRIDE_X) && defined(CONV_STRIDE_Y)
/** This function computes the depthwise convolution for NHWC data layout when the stride along the width or height is not 1.
 *
 * @note The number of elements read per thread must be passed at compile time using -DVEC_SIZE (e.g. -DVEC_SIZE=2)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_TOP (e.g. -DCONV_PAD_TOP=1)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_LEFT (e.g. -DCONV_PAD_LEFT=1)
 * @note The convolution stride along the width must be passed at compile time using -DCONV_STRIDE_X (e.g. -DCONV_STRIDE_Y=X)
 * @note The convolution stride along the height must be passed at compile time using -DCONV_STRIDE_Y (e.g. -DCONV_STRIDE_Y=1)
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: FP32
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: same as src_ptr
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: QASYMM8
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in] max_offset                            Max offset for the input tensor
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: same as src_ptr
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */
__kernel void depthwise_convolution_3x3_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    int max_offset)
{
    int x = get_global_id(0); // channels
    int y = get_global_id(1); // spatial coordinate x
    int z = get_global_id(2); // spatial coordinate y

    Vector weights = CONVERT_TO_VECTOR_STRUCT(weights);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(float) * VEC_SIZE;

    int  z_coord  = 0;
    int4 offset   = 0;
    int4 y_offset = ((int4)(y * CONV_STRIDE_X) + (int4)(0, 1, 2, 3) - CONV_PAD_LEFT) * (int4)src_stride_y;

    // We compute 2x1x1 [C,W,H] elements
    VEC_FLOAT acc = 0;

    // Load weights
    VEC_FLOAT w0 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 0 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w1 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 1 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w2 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 2 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w3 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 0 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w4 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 1 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w5 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 2 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w6 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 0 * weights_stride_y + 2 * weights_stride_z));
    VEC_FLOAT w7 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 1 * weights_stride_y + 2 * weights_stride_z));
    VEC_FLOAT w8 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 2 * weights_stride_y + 2 * weights_stride_z));

    // Load input values
    // z == 0
    // Clamp z_coord as for z = 0, it can be negative
    // z_coord is casted to unsigned int in order to use just a min() operation
    // A "-1" 32 bit signed variable converted to unsigned gives 4294967295
    z_coord = z * CONV_STRIDE_Y - (int)CONV_PAD_TOP;
    z_coord = min((uint)z_coord, (uint)SRC_DIM_2);
    offset  = y_offset + (int4)(z_coord * src_stride_z);
    offset  = min(offset, max_offset);

    VEC_FLOAT values0 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s0));
    VEC_FLOAT values1 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s1));
    VEC_FLOAT values2 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s2));

    // z == 1
    // z_coord can be only negative for z = 0 so we do not need to clamp it
    // Moreover z_coord cannot be out-of-bound for z = 1 so we do not need to clamp the offset
    z_coord           = z * CONV_STRIDE_Y - (int)CONV_PAD_TOP + 1;
    offset            = y_offset + (int4)(z_coord * src_stride_z);
    VEC_FLOAT values3 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s0));
    VEC_FLOAT values4 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s1));
    VEC_FLOAT values5 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s2));

    // z == 2
    // After z = 1 we can simply add src_stride_z to offset without updating z_coord
    // However offset can be out-of-bound so we need to check if it is greater than max_offset
    offset += (int4)src_stride_z;
    offset            = min(offset, max_offset);
    VEC_FLOAT values6 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s0));
    VEC_FLOAT values7 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s1));
    VEC_FLOAT values8 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s2));

    acc = fma(values0, w0, acc);
    acc = fma(values1, w1, acc);
    acc = fma(values2, w2, acc);

    acc = fma(values3, w3, acc);
    acc = fma(values4, w4, acc);
    acc = fma(values5, w5, acc);

    acc = fma(values6, w6, acc);
    acc = fma(values7, w7, acc);
    acc = fma(values8, w8, acc);

#if defined(HAS_BIAS)
    Vector    biases      = CONVERT_TO_VECTOR_STRUCT(biases);
    VEC_FLOAT bias_values = VLOAD(VEC_SIZE)(0, (__global float *)biases.ptr);
    acc += bias_values;
#endif // defined(HAS_BIAS)

    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    VSTORE(VEC_SIZE)
    (acc, 0, (__global float *)(dst.ptr));
}
#endif // defined(CONV_STRIDE_X) && defined(CONV_STRIDE_Y)

#if defined(NUM_ROWS_PROCESSED) && defined(NUM_PLANES_PROCESSED)
/** This function computes the depthwise convolution for NHWC data layout when the stride along the width and height is 1.
 *
 * @note The number of elements read per thread must be passed at compile time using -DVEC_SIZE (e.g. -DVEC_SIZE=2)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The number of rows processed per thread must be passed at compile time using -DNUM_ROWS_PROCESSED (i.e. -DNUM_ROWS_PROCESSED=2)
 * @note The number of planes processed per thread must be passed at compile time using -DNUM_PLANES_PROCESSED (i.e. -DNUM_PLANES_PROCESSED=2)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_TOP (e.g. -DCONV_PAD_TOP=1)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_LEFT (e.g. -DCONV_PAD_LEFT=1)
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: FP32
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: same as src_ptr
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: QASYMM8
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in] max_offset                            Max offset for the input tensor
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: same as src_ptr
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */
__kernel void depthwise_convolution_3x3_nhwc_stride1(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    int max_offset)
{
    int x = get_global_id(0); // channels
    int y = get_global_id(1); // spatial coordinate x
    int z = get_global_id(2); // spatial coordinate y

    Vector weights = CONVERT_TO_VECTOR_STRUCT(weights);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(float) * VEC_SIZE;

    int  z_coord  = 0;
    int4 offset   = 0;
    int4 y_offset = ((int4)(y * NUM_ROWS_PROCESSED) + (int4)(0, 1, 2, 3) - CONV_PAD_LEFT) * (int4)src_stride_y;

    // We compute 2x2x2 [C,W,H] elements
    VEC_FLOAT acc0 = 0;
    VEC_FLOAT acc1 = 0;
    VEC_FLOAT acc2 = 0;
    VEC_FLOAT acc3 = 0;

    // Load weights
    VEC_FLOAT w0 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 0 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w1 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 1 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w2 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 2 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w3 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 0 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w4 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 1 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w5 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 2 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w6 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 0 * weights_stride_y + 2 * weights_stride_z));
    VEC_FLOAT w7 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 1 * weights_stride_y + 2 * weights_stride_z));
    VEC_FLOAT w8 = VLOAD(VEC_SIZE)(0, (__global float *)(weights.ptr + 2 * weights_stride_y + 2 * weights_stride_z));

    // Load input values
    // z == 0
    // Clamp z_coord as for z = 0, it can be negative
    // z_coord is casted to unsigned int in order to use just a min() operation
    // A "-1" 32 bit signed variable converted to unsigned gives 4294967295
    z_coord = z * NUM_PLANES_PROCESSED - (int)CONV_PAD_TOP;
    z_coord = min((uint)z_coord, (uint)SRC_DIM_2);
    offset  = y_offset + (int4)(z_coord * src_stride_z);
    offset  = min(offset, max_offset);

    VEC_FLOAT values0 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s0));
    VEC_FLOAT values1 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s1));
    VEC_FLOAT values2 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s2));
    VEC_FLOAT values3 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s3));

    // z == 1
    // z_coord can be only negative for z = 0 so we do not need to clamp it
    // Moreover z_coord cannot be out-of-bound for z = 1 so we do not need to clamp the offset
    z_coord           = z * NUM_PLANES_PROCESSED - (int)CONV_PAD_TOP + 1;
    offset            = y_offset + (int4)(z_coord * src_stride_z);
    VEC_FLOAT values4 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s0));
    VEC_FLOAT values5 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s1));
    VEC_FLOAT values6 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s2));
    VEC_FLOAT values7 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s3));

    // z == 2
    // After z = 1 we can simply add src_stride_z to offset without updating z_coord
    // However offset can be out-of-bound so we need to check if it is greater than max_offset
    offset += (int4)src_stride_z;
    offset             = min(offset, max_offset);
    VEC_FLOAT values8  = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s0));
    VEC_FLOAT values9  = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s1));
    VEC_FLOAT values10 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s2));
    VEC_FLOAT values11 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s3));

    // z == 3
    // After z = 1 we can simply add src_stride_z to offset without updating z_coord
    // However offset can be out-of-bound so we need to check if it is greater than max_offset
    offset += (int4)(src_stride_z);
    offset             = min(offset, max_offset);
    VEC_FLOAT values12 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s0));
    VEC_FLOAT values13 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s1));
    VEC_FLOAT values14 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s2));
    VEC_FLOAT values15 = VLOAD(VEC_SIZE)(0, (__global float *)(src_addr + offset.s3));

    acc0 = fma(values0, w0, acc0);
    acc0 = fma(values1, w1, acc0);
    acc0 = fma(values2, w2, acc0);
    acc1 = fma(values1, w0, acc1);
    acc1 = fma(values2, w1, acc1);
    acc1 = fma(values3, w2, acc1);

    acc0 = fma(values4, w3, acc0);
    acc0 = fma(values5, w4, acc0);
    acc0 = fma(values6, w5, acc0);
    acc1 = fma(values5, w3, acc1);
    acc1 = fma(values6, w4, acc1);
    acc1 = fma(values7, w5, acc1);

    acc0 = fma(values8, w6, acc0);
    acc0 = fma(values9, w7, acc0);
    acc0 = fma(values10, w8, acc0);
    acc1 = fma(values9, w6, acc1);
    acc1 = fma(values10, w7, acc1);
    acc1 = fma(values11, w8, acc1);

    acc2 = fma(values4, w0, acc2);
    acc2 = fma(values5, w1, acc2);
    acc2 = fma(values6, w2, acc2);
    acc3 = fma(values5, w0, acc3);
    acc3 = fma(values6, w1, acc3);
    acc3 = fma(values7, w2, acc3);

    acc2 = fma(values8, w3, acc2);
    acc2 = fma(values9, w4, acc2);
    acc2 = fma(values10, w5, acc2);
    acc3 = fma(values9, w3, acc3);
    acc3 = fma(values10, w4, acc3);
    acc3 = fma(values11, w5, acc3);

    acc2 = fma(values12, w6, acc2);
    acc2 = fma(values13, w7, acc2);
    acc2 = fma(values14, w8, acc2);
    acc3 = fma(values13, w6, acc3);
    acc3 = fma(values14, w7, acc3);
    acc3 = fma(values15, w8, acc3);

#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);

    VEC_FLOAT bias_values = VLOAD(VEC_SIZE)(0, (__global float *)biases.ptr);

    acc0 += bias_values;
    acc1 += bias_values;
    acc2 += bias_values;
    acc3 += bias_values;
#endif // defined(HAS_BIAS)

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + (z * NUM_PLANES_PROCESSED) * dst_step_z;

    VSTORE(VEC_SIZE)
    (acc0, 0, (__global float *)(dst_addr + 0 * dst_stride_y));
    VSTORE(VEC_SIZE)
    (acc1, 0, (__global float *)(dst_addr + 1 * dst_stride_y));

#if((DST_DIM_2 % NUM_PLANES_PROCESSED) != 0)
    if((z * NUM_PLANES_PROCESSED + 1) < DST_DIM_2)
#endif // ((DST_DIM_2 % NUM_PLANES_PROCESSED) != 0)
    {
        VSTORE(VEC_SIZE)
        (acc2, 0, (__global float *)(dst_addr + 0 * dst_stride_y + 1 * dst_stride_z));
        VSTORE(VEC_SIZE)
        (acc3, 0, (__global float *)(dst_addr + 1 * dst_stride_y + 1 * dst_stride_z));
    }
}

#endif // defined(NUM_ROWS_PROCESSED) && defined(NUM_PLANES_PROCESSED)
#endif // defined(VEC_SIZE) && defined(SRC_DIM_2) && defined(CONV_PAD_TOP) && defined(CONV_PAD_LEFT)