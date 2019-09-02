/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#include "activation_float_helpers.h"

/** Get the pointer position at a certain offset in x and y direction.
 *
 * @param[in] ptr      Pointer to the starting position of the buffer
 * @param[in] x        Relative X position
 * @param[in] y        Relative Y position
 * @param[in] stride_x Stride of the source tensor in X dimension (in bytes)
 * @param[in] stride_y Stride of the source tensor in Y dimension (in bytes)
 *
 * @return a uchar
 */
inline __global uchar *ptr_offset(__global uchar *ptr, const int x, const int y, const int stride_x, const int stride_y)
{
    return ptr + x * stride_x + y * stride_y;
}

#if(DILATION_X == 1 && DILATION_Y == 1)

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

#else /* DILATION_X==1 && DILATION_Y==1 */

#define CONVOLUTION1x3_BIFROST2X1_STRIDE1(acc, src0_left, src0_mid, src0_right, weights_row0) \
    ({                                                                                        \
        acc.s0 = fma(src0_left.s0, weights_row0.s0, acc.s0);                                  \
        acc.s0 = fma(src0_mid.s0, weights_row0.s1, acc.s0);                                   \
        acc.s0 = fma(src0_right.s0, weights_row0.s2, acc.s0);                                 \
        acc.s1 = fma(src0_left.s1, weights_row0.s0, acc.s1);                                  \
        acc.s1 = fma(src0_mid.s1, weights_row0.s1, acc.s1);                                   \
        acc.s1 = fma(src0_right.s1, weights_row0.s2, acc.s1);                                 \
    })

#define CONVOLUTION1x3_BIFROST2X1_STRIDE2(acc, src0_left, src0_mid, src0_right, weights_row0) \
    ({                                                                                        \
        acc.s0 = fma(src0_left.s0, weights_row0.s0, acc.s0);                                  \
        acc.s0 = fma(src0_mid.s0, weights_row0.s1, acc.s0);                                   \
        acc.s0 = fma(src0_right.s0, weights_row0.s2, acc.s0);                                 \
        acc.s1 = fma(src0_left.s2, weights_row0.s0, acc.s1);                                  \
        acc.s1 = fma(src0_mid.s2, weights_row0.s1, acc.s1);                                   \
        acc.s1 = fma(src0_right.s2, weights_row0.s2, acc.s1);                                 \
    })

#define CONVOLUTION1x3_BIFROST4X1_STRIDE1(acc, src0_left, src0_mid, src0_right, weights_row0) \
    ({                                                                                        \
        acc.s0 = fma(src0_left.s0, weights_row0.s0, acc.s0);                                  \
        acc.s0 = fma(src0_mid.s0, weights_row0.s1, acc.s0);                                   \
        acc.s0 = fma(src0_right.s0, weights_row0.s2, acc.s0);                                 \
        acc.s1 = fma(src0_left.s1, weights_row0.s0, acc.s1);                                  \
        acc.s1 = fma(src0_mid.s1, weights_row0.s1, acc.s1);                                   \
        acc.s1 = fma(src0_right.s1, weights_row0.s2, acc.s1);                                 \
        acc.s2 = fma(src0_left.s2, weights_row0.s0, acc.s2);                                  \
        acc.s2 = fma(src0_mid.s2, weights_row0.s1, acc.s2);                                   \
        acc.s2 = fma(src0_right.s2, weights_row0.s2, acc.s2);                                 \
        acc.s3 = fma(src0_left.s3, weights_row0.s0, acc.s3);                                  \
        acc.s3 = fma(src0_mid.s3, weights_row0.s1, acc.s3);                                   \
        acc.s3 = fma(src0_right.s3, weights_row0.s2, acc.s3);                                 \
    })

#define CONVOLUTION1x3_BIFROST4X1_STRIDE2(acc, src0_left, src0_mid, src0_right, weights_row0) \
    ({                                                                                        \
        acc.s0 = fma(src0_left.s0, weights_row0.s0, acc.s0);                                  \
        acc.s0 = fma(src0_mid.s0, weights_row0.s1, acc.s0);                                   \
        acc.s0 = fma(src0_right.s0, weights_row0.s2, acc.s0);                                 \
        acc.s1 = fma(src0_left.s2, weights_row0.s0, acc.s1);                                  \
        acc.s1 = fma(src0_mid.s2, weights_row0.s1, acc.s1);                                   \
        acc.s1 = fma(src0_right.s2, weights_row0.s2, acc.s1);                                 \
        acc.s2 = fma(src0_left.s4, weights_row0.s0, acc.s2);                                  \
        acc.s2 = fma(src0_mid.s4, weights_row0.s1, acc.s2);                                   \
        acc.s2 = fma(src0_right.s4, weights_row0.s2, acc.s2);                                 \
        acc.s3 = fma(src0_left.s6, weights_row0.s0, acc.s3);                                  \
        acc.s3 = fma(src0_mid.s6, weights_row0.s1, acc.s3);                                   \
        acc.s3 = fma(src0_right.s6, weights_row0.s2, acc.s3);                                 \
    })

#endif /* DILATION_X==1 && DILATION_Y==1 */

#if defined(DEPTH_MULTIPLIER) && defined(DST_CHANNELS) && defined(IS_F32)
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
#if(DILATION_X == 1 && DILATION_Y == 1)
    float4 temp = vload4(0, (__global float *)left_pixel);

    float2 left   = CONVERT(temp.s01, float2);
    float2 middle = CONVERT(temp.s12, float2);
    float2 right  = CONVERT(temp.s23, float2);
    return left * (float2)left_coeff + middle * (float2)middle_coeff + right * (float2)right_coeff;
#else  /* DILATION_X==1 && DILATION_Y==1 */
    return vload2(0, (__global float *)left_pixel) * (float2)left_coeff
           + vload2(0, (__global float *)(left_pixel) + DILATION_X) * (float2)middle_coeff
           + vload2(0, (__global float *)(left_pixel) + 2 * DILATION_X) * (float2)right_coeff;
#endif /* DILATION_X==1 && DILATION_Y==1 */
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
#if(DILATION_X == 1 && DILATION_Y == 1)
    float4 temp0 = vload4(0, (__global float *)left_pixel);
    float  temp1 = *((__global float *)(left_pixel + 4 * sizeof(float)));

    float2 left   = CONVERT(temp0.s02, float2);
    float2 middle = CONVERT(temp0.s13, float2);
    float2 right  = CONVERT((float2)(temp0.s2, temp1), float2);

    return left * (float2)left_coeff + middle * (float2)middle_coeff + right * (float2)right_coeff;
#else /* DILATION_X==1 && DILATION_Y==1 */
    __global float *left_pixel_float = (__global float *)left_pixel;

    return vload4(0, left_pixel_float).s02 * (float2)left_coeff
           + vload4(0, left_pixel_float + DILATION_X).s02 * (float2)middle_coeff
           + vload4(0, left_pixel_float + DILATION_X * 2).s02 * (float2)right_coeff;

#endif /* DILATION_X==1 && DILATION_Y==1 */
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
#if(DILATION_X == 1 && DILATION_Y == 1)
    float4 temp0 = vload4(0, (__global float *)left_pixel);
    float2 temp1 = vload2(0, (__global float *)(left_pixel + 4 * sizeof(float)));

    float2 left   = CONVERT(temp0.s03, float2);
    float2 middle = CONVERT((float2)(temp0.s1, temp1.s0), float2);
    float2 right  = CONVERT((float2)(temp0.s2, temp1.s1), float2);

    return left * (float2)left_coeff + middle * (float2)middle_coeff + right * (float2)right_coeff;
#else  /* DILATION_X==1 && DILATION_Y==1 */
    __global float *left_pixel_float = (__global float *)left_pixel;

    return (float2)(*left_pixel_float, *(left_pixel_float + 3)) * (float2)left_coeff
           + (float2)(*(left_pixel_float + DILATION_X), *(left_pixel_float + DILATION_X + 3)) * (float2)middle_coeff
           + (float2)(*(left_pixel_float + DILATION_X * 2), *(left_pixel_float + DILATION_X * 2 + 3)) * (float2)right_coeff;
#endif /* DILATION_X==1 && DILATION_Y==1 */
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
    __global const uchar *src,
    unsigned int          src_stride_y,
    const float mat0, const float mat1, const float mat2,
    const float mat3, const float mat4, const float mat5,
    const float mat6, const float mat7, const float mat8)
{
    float2 pixels;

    pixels = convolution1x3((src + 0 * DILATION_Y * src_stride_y), mat0, mat1, mat2);
    pixels += convolution1x3((src + 1 * DILATION_Y * src_stride_y), mat3, mat4, mat5);
    pixels += convolution1x3((src + 2 * DILATION_Y * src_stride_y), mat6, mat7, mat8);

    return pixels;
}

/** This OpenCL kernel computes the depthwise convolution 3x3
 *
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: F32
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
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
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);

    float2 pixels = 0.0f;

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;
    // Load relevant input and weights data (Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)

    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;

    __global uchar *src_addr = src.ptr - batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z - (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;

    // Load the weights
    float3 weights_values0 = vload3(0, (__global float *)(weights_addr + 0 * weights_stride_y));
    float3 weights_values1 = vload3(0, (__global float *)(weights_addr + 1 * weights_stride_y));
    float3 weights_values2 = vload3(0, (__global float *)(weights_addr + 2 * weights_stride_y));

    pixels = convolution3x3(src_addr, src_stride_y,
                            weights_values0.s0, weights_values0.s1, weights_values0.s2,
                            weights_values1.s0, weights_values1.s1, weights_values1.s2,
                            weights_values2.s0, weights_values2.s1, weights_values2.s2);
#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    float bias = *((__global float *)(vector_offset(&biases, channel)));

    pixels += (float2)bias;
#endif //defined(HAS_BIAS)

    vstore2(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels, A_VAL, B_VAL), 0, (__global float *)dst.ptr);
}
#endif //defined(CONV_STRIDE_X)

#if(DILATION_X > 1 || DILATION_Y > 1)

/** Perform 3x3 convolution for stride_x=1 and stride_y=1 when DILATION_X>1 or DILATION_Y>1 for F32
 *
 * @param[in] src_addr         Pointer to the starting position of where to perform the convolution
 * @param[in] src_stride_x     Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_stride_y     Stride of the source tensor in Y dimension (in bytes)
 * @param[in] y_offset         Offset from the source tensor from which to start convolution
 * @param[in] weights_addr     Pointer from where to get weights
 * @param[in] weights_stride_y Stride of weights tesnsor in Y dimension
 */
inline float2 convolution_3x3_dilation_stridex1_stridey1_bifrost_f32(__global uchar *src_addr, const int stride_x_bytes, const int stride_y_bytes,
                                                                     const int y_offset, __global uchar *weights_addr, const int weights_stride_y)
{
    // Load the weights
    float3 weights_row0 = vload3(0, (__global float *)(weights_addr + 0 * weights_stride_y));
    float3 weights_row1 = vload3(0, (__global float *)(weights_addr + 1 * weights_stride_y));
    float3 weights_row2 = vload3(0, (__global float *)(weights_addr + 2 * weights_stride_y));

    float2 pixels0 = 0.0f;

    float2 src00_left  = vload2(0, (__global float *)ptr_offset(src_addr, 0, y_offset, stride_x_bytes, stride_y_bytes)); // Row0
    float2 src00_mid   = vload2(0, (__global float *)ptr_offset(src_addr, DILATION_X, y_offset, stride_x_bytes, stride_y_bytes));
    float2 src00_right = vload2(0, (__global float *)ptr_offset(src_addr, 2 * DILATION_X, y_offset, stride_x_bytes, stride_y_bytes));

    float2 src10_left  = vload2(0, (__global float *)ptr_offset(src_addr, 0, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes)); // Row1
    float2 src10_mid   = vload2(0, (__global float *)ptr_offset(src_addr, DILATION_X, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes));
    float2 src10_right = vload2(0, (__global float *)ptr_offset(src_addr, 2 * DILATION_X, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes));

    float2 src20_left  = vload2(0, (__global float *)ptr_offset(src_addr, 0, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes)); // Row2
    float2 src20_mid   = vload2(0, (__global float *)ptr_offset(src_addr, DILATION_X, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes));
    float2 src20_right = vload2(0, (__global float *)ptr_offset(src_addr, 2 * DILATION_X, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes));

    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels0, src00_left, src00_mid, src00_right, weights_row0);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels0, src10_left, src10_mid, src10_right, weights_row1);
    CONVOLUTION1x3_BIFROST2X1_STRIDE1(pixels0, src20_left, src20_mid, src20_right, weights_row2);

    return pixels0;
}

/** Perform 3x3 convolution for stride_x=2 and stride_y=2 when DILATION_X>1 or DILATION_Y>1 for F32
 *
 * @param[in] src_addr         Pointer to the starting position of where to perform the convolution
 * @param[in] src_stride_x     Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_stride_y     Stride of the source tensor in Y dimension (in bytes)
 * @param[in] y_offset         Offset from the source tensor from which to start convolution
 * @param[in] weights_addr     Pointer from where to get weights
 * @param[in] weights_stride_y Stride of weights tesnsor in Y dimension
 */
inline float2 convolution_3x3_dilation_stridex2_stridey2_bifrost_f32(__global uchar *src_addr, const int stride_x_bytes, const int stride_y_bytes,
                                                                     const int y_offset, __global uchar *weights_addr, const int weights_stride_y)
{
    // Load the weights
    float3 weights_row0 = vload3(0, (__global float *)(weights_addr + 0 * weights_stride_y));
    float3 weights_row1 = vload3(0, (__global float *)(weights_addr + 1 * weights_stride_y));
    float3 weights_row2 = vload3(0, (__global float *)(weights_addr + 2 * weights_stride_y));

    float2 pixels0 = 0.0f;

    float3 src00_left  = vload3(0, (__global float *)ptr_offset(src_addr, 0, y_offset, stride_x_bytes, stride_y_bytes)); // Row0
    float3 src00_mid   = vload3(0, (__global float *)ptr_offset(src_addr, DILATION_X, y_offset, stride_x_bytes, stride_y_bytes));
    float3 src00_right = vload3(0, (__global float *)ptr_offset(src_addr, 2 * DILATION_X, y_offset, stride_x_bytes, stride_y_bytes));

    float3 src10_left  = vload3(0, (__global float *)ptr_offset(src_addr, 0, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes)); // Row1
    float3 src10_mid   = vload3(0, (__global float *)ptr_offset(src_addr, DILATION_X, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes));
    float3 src10_right = vload3(0, (__global float *)ptr_offset(src_addr, 2 * DILATION_X, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes));

    float3 src20_left  = vload3(0, (__global float *)ptr_offset(src_addr, 0, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes)); // Row2
    float3 src20_mid   = vload3(0, (__global float *)ptr_offset(src_addr, DILATION_X, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes));
    float3 src20_right = vload3(0, (__global float *)ptr_offset(src_addr, 2 * DILATION_X, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes));

    CONVOLUTION1x3_BIFROST2X1_STRIDE2(pixels0, src00_left, src00_mid, src00_right, weights_row0);
    CONVOLUTION1x3_BIFROST2X1_STRIDE2(pixels0, src10_left, src10_mid, src10_right, weights_row1);
    CONVOLUTION1x3_BIFROST2X1_STRIDE2(pixels0, src20_left, src20_mid, src20_right, weights_row2);

    return pixels0;
}

#endif /* (DILATION_X > 1 || DILATION_Y > 1) */

/** This OpenCL kernel is optimized for Bifrost architectures and computes the depthwise convolution 3x3 when both
 * stride_x and stride_y are equal to 1
 *
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note If activation function is enabled, the data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float.
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: F32
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
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
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);

    float2 pixels0 = 0.0f;
    float2 pixels1 = 0.0f;
    float2 pixels2 = 0.0f;
    float2 pixels3 = 0.0f;

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;
    // Load relevant input and weights data (Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)
    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;
    __global uchar *src_addr     = src.ptr - batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z - (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;

#if(DILATION_X == 1 && DILATION_Y == 1)
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

#else /* DILATION_X==1 && DILATION_Y==1 */

    //3x3 Convolution of elements starting in 0th row
    pixels0 = convolution_3x3_dilation_stridex1_stridey1_bifrost_f32(src_addr, src.stride_x, src.stride_y, 0, weights_addr, weights_stride_y);
    //3x3 Convolution of elements starting in 1st row
    pixels1 = convolution_3x3_dilation_stridex1_stridey1_bifrost_f32(src_addr, src.stride_x, src.stride_y, 1, weights_addr, weights_stride_y);
    //3x3 Convolution of elements starting in 2nd row
    pixels2 = convolution_3x3_dilation_stridex1_stridey1_bifrost_f32(src_addr, src.stride_x, src.stride_y, 2, weights_addr, weights_stride_y);
    //3x3 Convolution of elements starting in 3rd row
    pixels3 = convolution_3x3_dilation_stridex1_stridey1_bifrost_f32(src_addr, src.stride_x, src.stride_y, 3, weights_addr, weights_stride_y);

#endif /* DILATION_X==1 && DILATION_Y==1 */

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    float bias = *((__global float *)(vector_offset(&biases, channel)));

    pixels0 += (float2)bias;
    pixels1 += (float2)bias;
    pixels2 += (float2)bias;
    pixels3 += (float2)bias;
#endif /* defined(HAS_BIAS) */

    vstore2(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels0, A_VAL, B_VAL), 0, (__global float *)(dst.ptr + 0 * dst_stride_y));
    vstore2(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels1, A_VAL, B_VAL), 0, (__global float *)(dst.ptr + 1 * dst_stride_y));
    vstore2(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels2, A_VAL, B_VAL), 0, (__global float *)(dst.ptr + 2 * dst_stride_y));
    vstore2(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels3, A_VAL, B_VAL), 0, (__global float *)(dst.ptr + 3 * dst_stride_y));
}

/** This OpenCL kernel is optimized for Bifrost architectures and computes the depthwise convolution 3x3 when both
 * stride_x and stride_y are equal to 2
 *
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note If activation function is enabled, the data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float.
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: F32
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
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
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);

    float2 pixels0 = 0.0f;
    float2 pixels1 = 0.0f;

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;
    // Load relevant input and weights data (Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)
    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;
    __global uchar *src_addr     = src.ptr - batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z - (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;

#if(DILATION_X == 1 && DILATION_Y == 1)

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

#else  /* DILATION_X==1 && DILATION_Y==1 */

    //3x3 Convolution of elements starting in 0th row
    pixels0 = convolution_3x3_dilation_stridex2_stridey2_bifrost_f32(src_addr, src.stride_x, src.stride_y, 0, weights_addr, weights_stride_y);
    //3x3 Convolution of elements starting in 2nd row
    pixels1 = convolution_3x3_dilation_stridex2_stridey2_bifrost_f32(src_addr, src.stride_x, src.stride_y, 2, weights_addr, weights_stride_y);
#endif /* DILATION_X==1 && DILATION_Y==1 */

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    float bias = *((__global float *)(vector_offset(&biases, channel)));

    pixels0 += (float2)bias;
    pixels1 += (float2)bias;
#endif /* defined(HAS_BIAS) */

    vstore2(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels0, A_VAL, B_VAL), 0, (__global float *)(dst.ptr + 0 * dst_stride_y));
    vstore2(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels1, A_VAL, B_VAL), 0, (__global float *)(dst.ptr + 1 * dst_stride_y));
}

#endif // defined(DEPTH_MULTIPLIER) && defined(DST_CHANNELS) && defined(IS_F32)

#if defined(VEC_SIZE) && defined(DATA_TYPE) && defined(DST_WIDTH)
/** Reshape the weights for quantized depthwise convolution
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type, e.g. -DDATA_TYPE=uint8
 * @note Output width should be given as a preprocessor argument using -DDST_WIDTH=width, e.g. -DDST_WIDTH=128
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=vec_size, e.g., -DVEC_SIZE=4
 * @attention Input's height and width should be 3
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: QASYMM8
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void depthwise_convolution_reshape_weights(
    TENSOR3D_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Vector    src = CONVERT_TO_VECTOR_STRUCT(src);
    const int x   = get_global_id(0);

    // Load 3x3xVEC_SIZE weights
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    w0 = VLOAD(VEC_SIZE)(0, src.ptr + 0 * src_stride_y + 0 * src_stride_z);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    w1 = VLOAD(VEC_SIZE)(0, src.ptr + 1 * src_stride_y + 0 * src_stride_z);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    w2 = VLOAD(VEC_SIZE)(0, src.ptr + 2 * src_stride_y + 0 * src_stride_z);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    w3 = VLOAD(VEC_SIZE)(0, src.ptr + 0 * src_stride_y + 1 * src_stride_z);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    w4 = VLOAD(VEC_SIZE)(0, src.ptr + 1 * src_stride_y + 1 * src_stride_z);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    w5 = VLOAD(VEC_SIZE)(0, src.ptr + 2 * src_stride_y + 1 * src_stride_z);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    w6 = VLOAD(VEC_SIZE)(0, src.ptr + 0 * src_stride_y + 2 * src_stride_z);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    w7 = VLOAD(VEC_SIZE)(0, src.ptr + 1 * src_stride_y + 2 * src_stride_z);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    w8 = VLOAD(VEC_SIZE)(0, src.ptr + 2 * src_stride_y + 2 * src_stride_z);

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * DST_WIDTH * sizeof(DATA_TYPE);

#if defined(TRANSPOSE)
#if VEC_SIZE != 4
#error "VEC_SIZE not supported"
#else  // VEC_SIZE != 4
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(w0.s0, w1.s0, w2.s0, w3.s0), 0, dst_addr + 0);
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(w4.s0, w5.s0, w6.s0, w7.s0), 0, dst_addr + 1 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(w8.s0, w0.s1, w1.s1, w2.s1), 0, dst_addr + 2 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(w3.s1, w4.s1, w5.s1, w6.s1), 0, dst_addr + 3 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(w7.s1, w8.s1, w0.s2, w1.s2), 0, dst_addr + 4 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(w2.s2, w3.s2, w4.s2, w5.s2), 0, dst_addr + 5 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(w6.s2, w7.s2, w8.s2, w0.s3), 0, dst_addr + 6 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(w1.s3, w2.s3, w3.s3, w4.s3), 0, dst_addr + 7 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(w5.s3, w6.s3, w7.s3, w8.s3), 0, dst_addr + 8 * sizeof(DATA_TYPE) * VEC_SIZE);
#endif // VEC_SIZE != 4
#else  // !defined(TRANSPOSE)
    VSTORE(VEC_SIZE)
    (w0, 0, dst_addr + 0);
    VSTORE(VEC_SIZE)
    (w1, 0, dst_addr + 1 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    (w2, 0, dst_addr + 2 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    (w3, 0, dst_addr + 3 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    (w4, 0, dst_addr + 4 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    (w5, 0, dst_addr + 5 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    (w6, 0, dst_addr + 6 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    (w7, 0, dst_addr + 7 * sizeof(DATA_TYPE) * VEC_SIZE);
    VSTORE(VEC_SIZE)
    (w8, 0, dst_addr + 8 * sizeof(DATA_TYPE) * VEC_SIZE);
#endif // defined(TRANSPOSE)
}
#endif // defined(VEC_SIZE) && defined(DATA_TYPE) && defined(DST_WIDTH)

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
__kernel void depthwise_convolution_reshape_weights_generic(
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

#if defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DATA_TYPE) && defined(PAD_VALUE) && defined(DEPTH_MULTIPLIER) && defined(DILATION_X) && defined(DILATION_Y)
/** This kernel performs a reshaping of the input tensor to a tensor used to perform depthwise convolution using vector to matrix multiplication.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The convolution information must be passed at compile time using -DSTRIDE_X, -DSTRIDE_Y, -DPAD_LEFT, -DPAD_TOP, -DPAD_RIGHT, -DPAD_BOTTOM, -DKERNEL_WIDHT, -DKERNEL_HEIGHT, -DSRC_WIDTH, -DSRC_HEIGHT, -DDEPTH_MULTIPLIER
 * @note The dilation_x and dilation_y must be passed at compile time using -DDILATION_X and -DDILATION_Y: e.g. -DDILATION_X=1, -DDILATION_Y=1
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16/F32
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
    const int max_initial_x    = STRIDE_X * (((full_length - (KERNEL_WIDTH + (KERNEL_WIDTH - 1) * (DILATION_X - 1))) / STRIDE_X) + 1);

    const int src_x = -PAD_LEFT + src_pixel_linear % max_initial_x;
    const int src_y = -PAD_TOP + src_pixel_linear / max_initial_x * STRIDE_Y;
    const int src_z = get_global_id(2) / DEPTH_MULTIPLIER;

    __global uchar *input_ptr      = src_ptr + src_offset_first_element_in_bytes + src_z * in_stride_z;
    __global DATA_TYPE *output_ptr = ((__global DATA_TYPE *)(dst.ptr));

    for(int y = src_y; y < src_y + KERNEL_HEIGHT + (KERNEL_HEIGHT - 1) * (DILATION_Y - 1); y += DILATION_Y)
    {
        for(int x = src_x; x < src_x + KERNEL_WIDTH + (KERNEL_WIDTH - 1) * (DILATION_X - 1); x += DILATION_X, ++output_ptr)
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
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16/F32
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

#if defined(ARM_COMPUTE_OPENCL_FP16_ENABLED) && defined(DEPTH_MULTIPLIER) && defined(DST_CHANNELS) && defined(IS_F16)
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

#if(DILATION_X > 1 || DILATION_Y > 1)

/** Perform 3x3 convolution for stride_x=1 and stride_y=1 when DILATION_X>1 or DILATION_Y>1 for f16
 *
 * @param[in] src_addr         Pointer to the starting position of where to perform the convolution
 * @param[in] src_stride_x     Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_stride_y     Stride of the source tensor in Y dimension (in bytes)
 * @param[in] y_offset         Offset from the source tensor from which to start convolution
 * @param[in] weights_addr     Pointer from where to get weights
 * @param[in] weights_stride_y Stride of weights tesnsor in Y dimension
 */
inline half4 convolution_3x3_dilation_stridex1_stridey1_bifrost_f16(__global uchar *src_addr, const int stride_x_bytes, const int stride_y_bytes,
                                                                    const int y_offset, __global uchar *weights_addr, const int weights_stride_y)
{
    // Load the weights
    half3 weights_row0 = vload3(0, (__global half *)(weights_addr + 0 * weights_stride_y));
    half3 weights_row1 = vload3(0, (__global half *)(weights_addr + 1 * weights_stride_y));
    half3 weights_row2 = vload3(0, (__global half *)(weights_addr + 2 * weights_stride_y));

    half4 pixels0 = 0.0f;

    half4 src00_left  = vload4(0, (__global half *)ptr_offset(src_addr, 0, y_offset, stride_x_bytes, stride_y_bytes)); // Row0
    half4 src00_mid   = vload4(0, (__global half *)ptr_offset(src_addr, DILATION_X, y_offset, stride_x_bytes, stride_y_bytes));
    half4 src00_right = vload4(0, (__global half *)ptr_offset(src_addr, 2 * DILATION_X, y_offset, stride_x_bytes, stride_y_bytes));

    half4 src10_left  = vload4(0, (__global half *)ptr_offset(src_addr, 0, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes)); // Row1
    half4 src10_mid   = vload4(0, (__global half *)ptr_offset(src_addr, DILATION_X, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes));
    half4 src10_right = vload4(0, (__global half *)ptr_offset(src_addr, 2 * DILATION_X, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes));

    half4 src20_left  = vload4(0, (__global half *)ptr_offset(src_addr, 0, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes)); // Row2
    half4 src20_mid   = vload4(0, (__global half *)ptr_offset(src_addr, DILATION_X, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes));
    half4 src20_right = vload4(0, (__global half *)ptr_offset(src_addr, 2 * DILATION_X, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes));

    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels0, src00_left, src00_mid, src00_right, weights_row0);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels0, src10_left, src10_mid, src10_right, weights_row1);
    CONVOLUTION1x3_BIFROST4X1_STRIDE1(pixels0, src20_left, src20_mid, src20_right, weights_row2);

    return pixels0;
}

/** Perform 3x3 convolution for stride_x=2 and stride_y=2 when DILATION_X>1 or DILATION_Y>1 for F16
 *
 * @param[in] src_addr         Pointer to the starting position of where to perform the convolution
 * @param[in] src_stride_x     Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_stride_y     Stride of the source tensor in Y dimension (in bytes)
 * @param[in] y_offset         Offset from the source tensor from which to start convolution
 * @param[in] weights_addr     Pointer from where to get weights
 * @param[in] weights_stride_y Stride of weights tesnsor in Y dimension
 */
inline half4 convolution_3x3_dilation_stridex2_stridey2_bifrost_f16(__global uchar *src_addr, const int stride_x_bytes, const int stride_y_bytes,
                                                                    const int y_offset, __global uchar *weights_addr, const int weights_stride_y)
{
    // Load the weights
    half3 weights_row0 = vload3(0, (__global half *)(weights_addr + 0 * weights_stride_y));
    half3 weights_row1 = vload3(0, (__global half *)(weights_addr + 1 * weights_stride_y));
    half3 weights_row2 = vload3(0, (__global half *)(weights_addr + 2 * weights_stride_y));

    half4 pixels0 = 0.0f;

    half8 src00_left  = vload8(0, (__global half *)ptr_offset(src_addr, 0, y_offset, stride_x_bytes, stride_y_bytes)); // Row0
    half8 src00_mid   = vload8(0, (__global half *)ptr_offset(src_addr, DILATION_X, y_offset, stride_x_bytes, stride_y_bytes));
    half8 src00_right = vload8(0, (__global half *)ptr_offset(src_addr, 2 * DILATION_X, y_offset, stride_x_bytes, stride_y_bytes));

    half8 src10_left  = vload8(0, (__global half *)ptr_offset(src_addr, 0, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes)); // Row1
    half8 src10_mid   = vload8(0, (__global half *)ptr_offset(src_addr, DILATION_X, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes));
    half8 src10_right = vload8(0, (__global half *)ptr_offset(src_addr, 2 * DILATION_X, y_offset + DILATION_Y, stride_x_bytes, stride_y_bytes));

    half8 src20_left  = vload8(0, (__global half *)ptr_offset(src_addr, 0, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes)); // Row2
    half8 src20_mid   = vload8(0, (__global half *)ptr_offset(src_addr, DILATION_X, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes));
    half8 src20_right = vload8(0, (__global half *)ptr_offset(src_addr, 2 * DILATION_X, y_offset + DILATION_Y * 2, stride_x_bytes, stride_y_bytes));

    CONVOLUTION1x3_BIFROST4X1_STRIDE2(pixels0, src00_left, src00_mid, src00_right, weights_row0);
    CONVOLUTION1x3_BIFROST4X1_STRIDE2(pixels0, src10_left, src10_mid, src10_right, weights_row1);
    CONVOLUTION1x3_BIFROST4X1_STRIDE2(pixels0, src20_left, src20_mid, src20_right, weights_row2);

    return pixels0;
}

#endif // (DILATION_X > 1 && DILATION_Y > 1)

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
#if(DILATION_X == 1 && DILATION_Y == 1)

    half8 temp = vload8(0, (__global half *)left_pixel);

    half4 left   = CONVERT(temp.s0123, half4);
    half4 middle = CONVERT(temp.s1234, half4);
    half4 right  = CONVERT(temp.s2345, half4);

    return left * (half4)left_coeff + middle * (half4)middle_coeff + right * (half4)right_coeff;
#else /* DILATION_X==1 && DILATION_Y==1 */
    return vload4(0, (__global half *)left_pixel) * (half4)left_coeff
           + vload4(0, (__global half *)(left_pixel) + DILATION_X) * (half4)middle_coeff
           + vload4(0, (__global half *)(left_pixel) + 2 * DILATION_X) * (half4)right_coeff;

#endif /* DILATION_X==1 && DILATION_Y==1 */
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
#if(DILATION_X == 1 && DILATION_Y == 1)

    half8 temp0 = vload8(0, (__global half *)left_pixel);
    half temp1  = *((__global half *)(left_pixel + 8 * sizeof(half)));

    half4 left   = CONVERT(temp0.s0246, half4);
    half4 middle = CONVERT(temp0.s1357, half4);
    half4 right  = CONVERT((half4)(temp0.s246, temp1), half4);

    return left * (half4)left_coeff + middle * (half4)middle_coeff + right * (half4)right_coeff;
#else /* DILATION_X==1 && DILATION_Y==1 */

    __global half *left_pixel_float = (__global half *)left_pixel;

    return (half4)(*left_pixel_float, *(left_pixel_float + 2), *(left_pixel_float + 4), *(left_pixel_float + 6)) * (half4)left_coeff
           + (half4)(*(left_pixel_float + DILATION_X), *(left_pixel_float + DILATION_X + 2), *(left_pixel_float + DILATION_X + 4), *(left_pixel_float + DILATION_X + 6)) * (half4)middle_coeff
           + (half4)(*(left_pixel_float + DILATION_X * 2), *(left_pixel_float + DILATION_X * 2 + 2), *(left_pixel_float + DILATION_X * 2 + 4), *(left_pixel_float + DILATION_X * 2 + 6)) * (half4)right_coeff;

#endif /* DILATION_X==1 && DILATION_Y==1 */
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
#if(DILATION_X == 1 && DILATION_Y == 1)

    half16 temp0 = vload16(0, (__global half *)left_pixel);

    half4 left   = CONVERT(temp0.s0369, half4);
    half4 middle = CONVERT(temp0.s147A, half4);
    half4 right  = CONVERT(temp0.s258B, half4);

    return left * (half4)left_coeff + middle * (half4)middle_coeff + right * (half4)right_coeff;
#else /* DILATION_X==1 && DILATION_Y==1 */

    __global half *left_pixel_float = (__global half *)left_pixel;

    return (half4)(*left_pixel_float, *(left_pixel_float + 3), *(left_pixel_float + 6), *(left_pixel_float + 9)) * (half4)left_coeff
           + (half4)(*(left_pixel_float + DILATION_X), *(left_pixel_float + DILATION_X + 3), *(left_pixel_float + DILATION_X + 6), *(left_pixel_float + DILATION_X + 9)) * (half4)middle_coeff
           + (half4)(*(left_pixel_float + DILATION_X * 2), *(left_pixel_float + DILATION_X * 2 + 3), *(left_pixel_float + DILATION_X * 2 + 6), *(left_pixel_float + DILATION_X * 2 + 9)) * (half4)right_coeff;

#endif /* DILATION_X==1 && DILATION_Y==1 */
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
    pixels += convolution1x3_f16(offset(src, 0, DILATION_Y), mat3, mat4, mat5);
    pixels += convolution1x3_f16(offset(src, 0, DILATION_Y * 2), mat6, mat7, mat8);

    return pixels;
}

#if defined(DEPTH_MULTIPLIER)

/** This OpenCL kernel computes the depthwise convolution 3x3
 *
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note If activation function is enabled, the data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types: half.
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: F16
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
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
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: F16
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
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);
#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif //defined(HAS_BIAS)

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;
    // Load relevant input and weights data (Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)
    src.ptr -= batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z + (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;
    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;

    uchar3 offset         = (uchar3)(0, 1, 2) * (uchar3)weights_stride_y;
    half3 weights_values0 = vload3(0, (__global half *)(weights_addr + offset.s0));
    half3 weights_values1 = vload3(0, (__global half *)(weights_addr + offset.s1));
    half3 weights_values2 = vload3(0, (__global half *)(weights_addr + offset.s2));

    half4 pixels = convolution3x3_f16(&src, weights_values0.s0, weights_values0.s1, weights_values0.s2,
                                      weights_values1.s0, weights_values1.s1, weights_values1.s2,
                                      weights_values2.s0, weights_values2.s1, weights_values2.s2);
#if defined(HAS_BIAS)
    pixels += (half4)(*((__global half *)(biases.ptr + channel * biases_stride_x)));
#endif //defined(HAS_BIAS)

    vstore4(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels, A_VAL, B_VAL), 0, (__global half *)dst.ptr);
}
#endif // defined(DEPTH_MULTIPLIER)
#endif // defined(CONV_STRIDE_X)

/** This OpenCL kernel is optimized for Bifrost architectures and computes the 16bit floating point depthwise convolution 3x3
 * when both stride_x and stride_y are equal to 1
 *
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note If activation function is enabled, the data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types: half.
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: F16
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
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
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    half bias = *((__global half *)(vector_offset(&biases, channel)));
#endif /* defined(HAS_BIAS) */

    half4 pixels0 = 0.0f;
    half4 pixels1 = 0.0f;
    half4 pixels2 = 0.0f;
    half4 pixels3 = 0.0f;

    // Load relevant input and weights data (Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)
    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;
    __global uchar *src_addr     = src.ptr - batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z - (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;

#if(DILATION_X == 1 && DILATION_Y == 1)
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

#else /* DILATION_X==1 && DILATION_Y==1 */

    //3x3 Convolution of elements starting in 0th row
    pixels0 = convolution_3x3_dilation_stridex1_stridey1_bifrost_f16(src_addr, src.stride_x, src.stride_y, 0, weights_addr, weights_stride_y);
    //3x3 Convolution of elements starting in 1st row
    pixels1 = convolution_3x3_dilation_stridex1_stridey1_bifrost_f16(src_addr, src.stride_x, src.stride_y, 1, weights_addr, weights_stride_y);
    //3x3 Convolution of elements starting in 2nd row
    pixels2 = convolution_3x3_dilation_stridex1_stridey1_bifrost_f16(src_addr, src.stride_x, src.stride_y, 2, weights_addr, weights_stride_y);
    //3x3 Convolution of elements starting in 3rd row
    pixels3 = convolution_3x3_dilation_stridex1_stridey1_bifrost_f16(src_addr, src.stride_x, src.stride_y, 3, weights_addr, weights_stride_y);

#endif /* DILATION_X==1 && DILATION_Y==1 */

#ifdef HAS_BIAS
    pixels0 += (half4)bias;
    pixels1 += (half4)bias;
    pixels2 += (half4)bias;
    pixels3 += (half4)bias;
#endif /* defined(HAS_BIAS) */

    vstore4(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels0, A_VAL, B_VAL), 0, (__global half *)(dst.ptr + 0 * dst_stride_y));
    vstore4(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels1, A_VAL, B_VAL), 0, (__global half *)(dst.ptr + 1 * dst_stride_y));
    vstore4(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels2, A_VAL, B_VAL), 0, (__global half *)(dst.ptr + 2 * dst_stride_y));
    vstore4(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels3, A_VAL, B_VAL), 0, (__global half *)(dst.ptr + 3 * dst_stride_y));
}

/** This OpenCL kernel is optimized for Bifrost architectures and computes 16bit floating point the depthwise convolution 3x3
 * when both stride_x and stride_y are equal to 2
 *
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note If activation function is enabled, the data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=half. Supported data types: half.
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: F16
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
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
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    half bias = *((__global half *)(vector_offset(&biases, channel)));
#endif /* defined(HAS_BIAS) */

    half4 pixels0 = 0.0f;
    half4 pixels1 = 0.0f;

    // Load relevant input and weights data ( Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)
    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;
    __global uchar *src_addr     = src.ptr - batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z - (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;

#if(DILATION_X == 1 && DILATION_Y == 1)

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

#else  /* DILATION_X==1 && DILATION_Y==1 */
    //3x3 Convolution of elements starting in 0th row
    pixels0 = convolution_3x3_dilation_stridex2_stridey2_bifrost_f16(src_addr, src.stride_x, src.stride_y, 0, weights_addr, weights_stride_y);
    //3x3 Convolution of elements starting in 2nd row
    pixels1 = convolution_3x3_dilation_stridex2_stridey2_bifrost_f16(src_addr, src.stride_x, src.stride_y, 2, weights_addr, weights_stride_y);
#endif /* DILATION_X==1 && DILATION_Y==1 */

#ifdef HAS_BIAS
    pixels0 += (half4)bias;
    pixels1 += (half4)bias;
#endif /* defined(HAS_BIAS) */

    vstore4(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels0, A_VAL, B_VAL), 0, (__global half *)(dst.ptr + 0 * dst_stride_y));
    vstore4(ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, pixels1, A_VAL, B_VAL), 0, (__global half *)(dst.ptr + 1 * dst_stride_y));
}
#endif // defined(ARM_COMPUTE_OPENCL_FP16_ENABLED) && defined(DEPTH_MULTIPLIER) && defined(DST_CHANNELS) && defined(IS_F16)

#if defined(VEC_SIZE) && defined(SRC_DIM_2) && defined(CONV_PAD_TOP) && defined(CONV_PAD_LEFT) && defined(DATA_TYPE)

#if DATA_TYPE != float || DATA_TYPE != half
#error "Unsupported data type"
#endif // DATA_TYPE != float || DATA_TYPE != half

#define VEC_FLOAT VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)

#if defined(CONV_STRIDE_X) && defined(CONV_STRIDE_Y)
/** This function computes the depthwise convolution for NHWC data layout when the stride along the width or height is not 1.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The number of elements read per thread must be passed at compile time using -DVEC_SIZE (e.g. -DVEC_SIZE=2)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_TOP (e.g. -DCONV_PAD_TOP=1)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_LEFT (e.g. -DCONV_PAD_LEFT=1)
 * @note The convolution stride along the width must be passed at compile time using -DCONV_STRIDE_X (e.g. -DCONV_STRIDE_Y=X)
 * @note The convolution stride along the height must be passed at compile time using -DCONV_STRIDE_Y (e.g. -DCONV_STRIDE_Y=1)
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: F16/F32
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                          Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                            src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: same as src_ptr
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_w                          Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                            dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: F16/F32
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
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    int max_offset)
{
    int x = get_global_id(0); // channels
    int y = get_global_id(1); // spatial coordinate x
#if defined(DST_DEPTH)
    int z = get_global_id(2) % (int)DST_DEPTH; // spatial coordinate y
    int b = get_global_id(2) / (int)DST_DEPTH; // batch
#else                                          // defined(DST_DEPTH)
    int      z               = get_global_id(2); // spatial coordinate y
#endif                                         // defined(DST_DEPTH)

    Vector weights = CONVERT_TO_VECTOR_STRUCT(weights);

#if defined(DST_DEPTH)
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) * VEC_SIZE + b * src_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) * VEC_SIZE;
#endif /* defined(DST_DEPTH) */

    int  z_coord  = 0;
    int4 offset   = 0;
    int4 y_offset = ((int4)(y * CONV_STRIDE_X) + (int4)(0, DILATION_X * 1, DILATION_X * 2, DILATION_X * 3) - CONV_PAD_LEFT) * (int4)src_stride_y;

    // We compute 2x1x1 [C,W,H] elements
    VEC_FLOAT acc = 0;

    // Load weights
    VEC_FLOAT w0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 0 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w1 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 1 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w2 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 2 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w3 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 0 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w4 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 1 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w5 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 2 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w6 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 0 * weights_stride_y + 2 * weights_stride_z));
    VEC_FLOAT w7 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 1 * weights_stride_y + 2 * weights_stride_z));
    VEC_FLOAT w8 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 2 * weights_stride_y + 2 * weights_stride_z));

    // Load input values
    // z == 0
    // Clamp z_coord as for z = 0, it can be negative
    // z_coord is casted to unsigned int in order to use just a min() operation
    // A "-1" 32 bit signed variable converted to unsigned gives 4294967295
    z_coord = z * CONV_STRIDE_Y - (int)CONV_PAD_TOP;
    z_coord = min((uint)z_coord, (uint)SRC_DIM_2);
    offset  = y_offset + (int4)(z_coord * src_stride_z);
    offset  = min(offset, (int4)max_offset);

    VEC_FLOAT values0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s0));
    VEC_FLOAT values1 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s1));
    VEC_FLOAT values2 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s2));

    // z == 1
    // z_coord can be only negative for z = 0 so we do not need to clamp it
    // Moreover z_coord cannot be out-of-bound for z = 1 so we do not need to clamp the offset
    z_coord           = z * CONV_STRIDE_Y - (int)CONV_PAD_TOP + DILATION_Y;
    offset            = y_offset + (int4)(z_coord * src_stride_z);
    VEC_FLOAT values3 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s0));
    VEC_FLOAT values4 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s1));
    VEC_FLOAT values5 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s2));

    // z == 2
    // Offset can be out-of-bound so we need to check if it is greater than max_offset
    z_coord           = z * CONV_STRIDE_Y - (int)CONV_PAD_TOP + DILATION_Y * 2;
    offset            = y_offset + (int4)(z_coord * src_stride_z);
    offset            = min(offset, (int4)max_offset);
    VEC_FLOAT values6 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s0));
    VEC_FLOAT values7 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s1));
    VEC_FLOAT values8 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s2));

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
    VEC_FLOAT bias_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)biases.ptr);
    acc += bias_values;
#endif // defined(HAS_BIAS)

#if defined(DST_DEPTH)
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + z * dst_step_z + b * dst_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + z * dst_step_z;
#endif /* defined(DST_DEPTH) */

    VSTORE(VEC_SIZE)
    (ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, acc, A_VAL, B_VAL), 0, (__global DATA_TYPE *)(dst_addr));
}
#endif // defined(CONV_STRIDE_X) && defined(CONV_STRIDE_Y)

#if defined(NUM_ROWS_PROCESSED) && defined(NUM_PLANES_PROCESSED)
/** This function computes the depthwise convolution for NHWC data layout when the stride along the width and height is 1.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The number of elements read per thread must be passed at compile time using -DVEC_SIZE (e.g. -DVEC_SIZE=2)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The number of rows processed per thread must be passed at compile time using -DNUM_ROWS_PROCESSED (i.e. -DNUM_ROWS_PROCESSED=2)
 * @note The number of planes processed per thread must be passed at compile time using -DNUM_PLANES_PROCESSED (i.e. -DNUM_PLANES_PROCESSED=2)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_TOP (e.g. -DCONV_PAD_TOP=1)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_LEFT (e.g. -DCONV_PAD_LEFT=1)
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: F16/F32
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                          Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                            src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: same as src_ptr
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_w                          Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                            dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: F16/F32
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
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    int max_offset)
{
    int x = get_global_id(0); // channels
    int y = get_global_id(1); // spatial coordinate x
#if defined(DST_DEPTH)
    int z = get_global_id(2) % (int)DST_DEPTH; // spatial coordinate y
    int b = get_global_id(2) / (int)DST_DEPTH; // batch
#else                                          // defined(DST_DEPTH)
    int             z        = get_global_id(2); // spatial coordinate y
#endif                                         // defined(DST_DEPTH)

    Vector weights = CONVERT_TO_VECTOR_STRUCT(weights);

#if defined(DST_DEPTH)
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) * VEC_SIZE + b * src_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) * VEC_SIZE;
#endif /* defined(DST_DEPTH) */

    int  z_coord  = 0;
    int4 offset   = 0;
    int4 y_offset = ((int4)(y * NUM_ROWS_PROCESSED) + (int4)(0, 1, 2, 3) - (int)CONV_PAD_LEFT) * (int4)src_stride_y;

    // We compute 2x2x2 [C,W,H] elements
    VEC_FLOAT acc0 = 0;
    VEC_FLOAT acc1 = 0;
    VEC_FLOAT acc2 = 0;
    VEC_FLOAT acc3 = 0;

    // Load weights
    VEC_FLOAT w0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 0 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w1 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 1 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w2 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 2 * weights_stride_y + 0 * weights_stride_z));
    VEC_FLOAT w3 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 0 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w4 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 1 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w5 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 2 * weights_stride_y + 1 * weights_stride_z));
    VEC_FLOAT w6 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 0 * weights_stride_y + 2 * weights_stride_z));
    VEC_FLOAT w7 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 1 * weights_stride_y + 2 * weights_stride_z));
    VEC_FLOAT w8 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights.ptr + 2 * weights_stride_y + 2 * weights_stride_z));

    // Load input values
    // z == 0
    // Clamp z_coord as for z = 0, it can be negative
    // z_coord is casted to unsigned int in order to use just a min() operation
    // A "-1" 32 bit signed variable converted to unsigned gives 4294967295
    z_coord = z * (int)NUM_PLANES_PROCESSED - (int)CONV_PAD_TOP;
    z_coord = min((uint)z_coord, (uint)SRC_DIM_2);
    offset  = y_offset + (int4)(z_coord * src_stride_z);
    offset  = min(offset, (int4)max_offset);

    VEC_FLOAT values0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s0));
    VEC_FLOAT values1 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s1));
    VEC_FLOAT values2 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s2));
    VEC_FLOAT values3 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s3));

    // z == 1
    // z_coord can be only negative for z = 0 so we do not need to clamp it
    // Moreover z_coord cannot be out-of-bound for z = 1 so we do not need to clamp the offset
    z_coord           = z * (int)NUM_PLANES_PROCESSED - (int)CONV_PAD_TOP + 1;
    offset            = y_offset + (int4)(z_coord * src_stride_z);
    VEC_FLOAT values4 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s0));
    VEC_FLOAT values5 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s1));
    VEC_FLOAT values6 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s2));
    VEC_FLOAT values7 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s3));

    // z == 2
    // After z = 1 we can simply add src_stride_z to offset without updating z_coord
    // However offset can be out-of-bound so we need to check if it is greater than max_offset
    offset += (int4)src_stride_z;
    offset             = min(offset, (int4)max_offset);
    VEC_FLOAT values8  = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s0));
    VEC_FLOAT values9  = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s1));
    VEC_FLOAT values10 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s2));
    VEC_FLOAT values11 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s3));

    // z == 3
    // After z = 1 we can simply add src_stride_z to offset without updating z_coord
    // However offset can be out-of-bound so we need to check if it is greater than max_offset
    offset += (int4)src_stride_z;
    offset             = min(offset, (int4)max_offset);
    VEC_FLOAT values12 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s0));
    VEC_FLOAT values13 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s1));
    VEC_FLOAT values14 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s2));
    VEC_FLOAT values15 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(src_addr + offset.s3));

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

    VEC_FLOAT bias_values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)biases.ptr);

    acc0 += bias_values;
    acc1 += bias_values;
    acc2 += bias_values;
    acc3 += bias_values;
#endif // defined(HAS_BIAS)

#if defined(DST_DEPTH)
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + (z * NUM_PLANES_PROCESSED) * dst_step_z + b * dst_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + (z * NUM_PLANES_PROCESSED) * dst_step_z;
#endif /* defined(DST_DEPTH) */

    VSTORE(VEC_SIZE)
    (ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, acc0, A_VAL, B_VAL), 0, (__global DATA_TYPE *)(dst_addr + 0 * dst_stride_y));
    VSTORE(VEC_SIZE)
    (ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, acc1, A_VAL, B_VAL), 0, (__global DATA_TYPE *)(dst_addr + 1 * dst_stride_y));

#if((DST_DIM_2 % NUM_PLANES_PROCESSED) != 0)
    if((z * NUM_PLANES_PROCESSED + 1) < DST_DIM_2)
#endif // ((DST_DIM_2 % NUM_PLANES_PROCESSED) != 0)
    {
        VSTORE(VEC_SIZE)
        (ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, acc2, A_VAL, B_VAL), 0, (__global DATA_TYPE *)(dst_addr + 0 * dst_stride_y + 1 * dst_stride_z));
        VSTORE(VEC_SIZE)
        (ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, acc3, A_VAL, B_VAL), 0, (__global DATA_TYPE *)(dst_addr + 1 * dst_stride_y + 1 * dst_stride_z));
    }
}

#endif // defined(NUM_ROWS_PROCESSED) && defined(NUM_PLANES_PROCESSED)
#endif // defined(VEC_SIZE) && defined(SRC_DIM_2) && defined(CONV_PAD_TOP) && defined(CONV_PAD_LEFT) && defined(DATA_TYPE)