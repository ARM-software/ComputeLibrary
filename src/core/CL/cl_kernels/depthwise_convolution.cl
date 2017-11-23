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

#include "helpers.h"

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

/** This function computes the horizontal integral of the image.
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: U8
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: F16/F32
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: F16/F32
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
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif /* HAS_BIAS */

    __global DATA_TYPE *input_ptr = (__global DATA_TYPE *)src.ptr;
    __global uchar *output_ptr    = dst_ptr + dst_offset_first_element_in_bytes + get_global_id(1) * SRC_WIDTH * dst_stride_x + get_global_id(2) * dst_stride_y;

    for(int i = 0; i < SRC_WIDTH; ++i, ++input_ptr)
    {
        *((__global DATA_TYPE *)(output_ptr + i * dst_stride_x)) = *input_ptr;
    }

#if defined(HAS_BIAS)
    if(get_global_id(1) == 0)
    {
        *((__global DATA_TYPE *)(output_ptr + SRC_WIDTH * get_global_size(1) * dst_stride_x)) = *((__global float *)(biases.ptr + get_global_id(2) * biases_stride_x));
    }
#endif // defined(HAS_BIAS)
}
#endif //defined(SRC_WIDTH) && defined(DATA_TYPE)

#if defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(DATA_TYPE)
/** This kernel performs a reshaping of the input tensor to a tensor used to perform depthwise convolution using vector to matrix multiplication.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The convolution information must be passed at compile time using -DSTRIDE_X, -DSTRIDE_Y, -DPAD_LEFT, -DPAD_TOP, -DPAD_RIGHT, -DPAD_BOTTOM, -DKERNEL_WIDHT, -DKERNEL_HEIGHT, -DSRC_WIDTH, -DSRC_HEIGHT
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
    const int src_z = get_global_id(2);

    __global uchar *input_ptr      = src_ptr + src_offset_first_element_in_bytes + src_z * src_stride_z;
    __global DATA_TYPE *output_ptr = ((__global DATA_TYPE *)(dst.ptr));

    for(int y = src_y; y < src_y + KERNEL_HEIGHT; ++y)
    {
        for(int x = src_x; x < src_x + KERNEL_WIDTH; ++x, ++output_ptr)
        {
            if(x < 0 || x >= SRC_WIDTH || y < 0 || y >= SRC_HEIGHT)
            {
                *output_ptr = 0;
            }
            else
            {
                *output_ptr = *((__global DATA_TYPE *)(input_ptr + x * src_stride_x + y * src_stride_y));
            }
        }
    }
#if defined(HAS_BIAS)
    *output_ptr = (DATA_TYPE)(1);
#endif // defined(HAS_BIAS)
}

#endif //defined(STRIDE_X) && defined(STRIDE_Y) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(PAD_RIGHT) && defined(PAD_BOTTOM) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(SRC_WIDTH) && defined(DATA_TYPE)

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

    __global uchar *out_ptr          = dst_ptr + dst_offset_first_element_in_bytes + index2D % CONV_WIDTH * dst_stride_x + index2D / CONV_WIDTH * dst_stride_y + z * dst_stride_z;
    *((__global DATA_TYPE *)out_ptr) = *((__global DATA_TYPE *)src.ptr);
}

#endif //defined(CONV_WIDTH) && defined(CONV_HEIGHT) && defined(DATA_TYPE)
