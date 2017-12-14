/*
 * Copyright (c) 2016, 2017 ARM Limited.
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

#if defined(FIXED_POINT_POSITION)
#include "fixed_point.h"

#define ADD_OP(a, b) ADD_SAT_OP_EXPAND((a), (b), DATA_TYPE_PROMOTED, 8)
#define MUL_OP(a, b) MUL_SAT_OP_EXPAND(CONVERT((a), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 8)), CONVERT((b), VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 8)), DATA_TYPE_PROMOTED, 8, FIXED_POINT_POSITION)

// There is no need to have a larger intermediate type for qs32 because all the arguments are already promoted
MULQ_SAT_IMPL(qs32x8, qs32x8)

#else /* FIXED_POINT_POSITION */
#undef CONVERT_SAT

#define ADD_OP(a, b) ((a) + (b))
#define MUL_OP(a, b) ((a) * (b))
#define CONVERT_SAT(a, b) ((a))

#endif /* FIXED_POINT_POSITION */

#if defined(DATA_TYPE) && defined(DATA_SIZE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH)

#if STRIDE_X == 3
#define INPUT_PIXEL_STR(data_size) extract_input_stride3_##data_size
#define INPUT_PIXEL(data_size) INPUT_PIXEL_STR(data_size)
#elif STRIDE_X == 2
#define INPUT_PIXEL(data_size) extract_input_stride2
#elif STRIDE_X == 1
#define INPUT_PIXEL(data_size) extract_input_stride1
#else /* STRIDE_X not equals 1, 2 or 3 */
#error "Only support strides 1, 2 and 3"
#endif /* STRIDE_X == 3 */

/** Extracts a 1D horizontal vector from the input tensor with stride as 1.
 *
 * @param[in] input_pixel Pointer to the first pixel.
 *
 * @return extracted input pixels.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) extract_input_stride1(__global const DATA_TYPE *input_pixel)
{
    return vload8(0, input_pixel);
}

/** Extracts a 1D horizontal vector from the input tensor with stride as 2.
 *
 * @param[in] input_pixel Pointer to the first pixel.
 *
 * @return extracted input pixels.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) extract_input_stride2(__global const DATA_TYPE *input_pixel)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    temp = vload16(0, input_pixel);
    return temp.s02468ace;
}

/** Extracts a 1D horizontal vector from the input tensor with stride as 3 and 32-bit data size.
 *
 * @param[in] input_pixel Pointer to the first pixel.
 *
 * @return extracted input pixels.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) extract_input_stride3_32(__global const DATA_TYPE *input_pixel)
{
    VEC_DATA_TYPE(DATA_TYPE, 4)
    temp1 = vload4(0, input_pixel);
    VEC_DATA_TYPE(DATA_TYPE, 4)
    temp2 = vload4(0, input_pixel + 6);
    VEC_DATA_TYPE(DATA_TYPE, 4)
    temp3 = vload4(0, input_pixel + 12);
    VEC_DATA_TYPE(DATA_TYPE, 4)
    temp4 = vload4(0, input_pixel + 18);
    return (VEC_DATA_TYPE(DATA_TYPE, 8))(temp1.s03, temp2.s03, temp3.s03, temp4.s03);
}

/** Extracts a 1D horizontal vector from the input tensor with stride as 3 and 16-bit data size.
 *
 * @param[in] input_pixel Pointer to the first pixel.
 *
 * @return extracted input pixels.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) extract_input_stride3_16(__global const DATA_TYPE *input_pixel)
{
    VEC_DATA_TYPE(DATA_TYPE, 8)
    temp1 = vload8(0, input_pixel);
    VEC_DATA_TYPE(DATA_TYPE, 8)
    temp2 = vload8(0, input_pixel + 8);
    VEC_DATA_TYPE(DATA_TYPE, 8)
    temp3 = vload8(0, input_pixel + 16);
    return (VEC_DATA_TYPE(DATA_TYPE, 8))(temp1.s036, temp2.s147, temp3.s25);
}

/** Extracts a 1D horizontal vector from the input tensor with stride as 3 and 8-bit data size.
 *
 * @param[in] input_pixel Pointer to the first pixel.
 *
 * @return extracted input pixels.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) extract_input_stride3_8(__global const DATA_TYPE *input_pixel)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    temp1 = vload16(0, input_pixel);
    VEC_DATA_TYPE(DATA_TYPE, 16)
    temp2 = vload16(0, input_pixel + 12);
    return (VEC_DATA_TYPE(DATA_TYPE, 8))(temp1.s0369, temp2.s0369);
}

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data size must be passed at compile time using -DDATA_SIZE e.g. -DDATA_SIZE=32
 * @note The convolution stride x must be passed at compile time using -DSTRIDE_X e.g. -DSTRIDE_X=1
 * @note The third dimensions of the weights tensors must be passed at compile time using -DWEIGHTS_DEPTH
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                            src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[out] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                            dst_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                            dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in]  weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in]  weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  weights_step_y                        weights_stride_y * number of elements along y processed per workitem(in bytes)
 * @param[in]  weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  weights_step_z                        weights_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in]  biases_ptr                            Pointer to the biases tensor. Same as @p src_ptr
 * @param[in]  biases_stride_x                       Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                         biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes  The offset of the first element in the biases tensor
 * @param[in]  weights_stride_w                      Stride of the weights tensor in the 4th dimension
 */
__kernel void direct_convolution1x1(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#ifdef HAS_BIAS
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    unsigned int weights_stride_w)
{
    Image    src     = CONVERT_TO_IMAGE_STRUCT(src);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);
    Tensor3D dst     = CONVERT_TO_TENSOR3D_STRUCT(dst);

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif /* defined(HAS_BIAS) */

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 8)
    pixels = 0;

    const uint z_index = get_global_id(2);

    weights.ptr += z_index * weights_stride_w;

    for(volatile int d = 0; d < WEIGHTS_DEPTH; ++d)
    {
        DATA_TYPE weight = *(__global DATA_TYPE *)weights.ptr;
        VEC_DATA_TYPE(DATA_TYPE, 8)
        input_pixel = INPUT_PIXEL(DATA_SIZE)((__global DATA_TYPE *)src.ptr);
        pixels      = ADD_OP(pixels, MUL_OP((VEC_DATA_TYPE(DATA_TYPE, 8))weight, input_pixel));
        src.ptr += src_stride_z;
        weights.ptr += weights_stride_z;
    }

#ifdef HAS_BIAS
    pixels = ADD_OP(pixels, (VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 8)) * ((__global DATA_TYPE *)(vector_offset(&biases, z_index))));
#endif /* defined(HAS_BIAS) */

    vstore8(CONVERT_SAT(pixels, VEC_DATA_TYPE(DATA_TYPE, 8)), 0, (__global DATA_TYPE *)dst.ptr);
}
#endif // defined(DATA_TYPE) && defined(DATA_SIZE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH)

#if defined(WEIGHTS_DEPTH)

#define CONVOLUTION1x1_BIFROST(acc, src, weight_value) \
    ({                                                 \
        acc.s0 = mad(src.s0, weight_value, acc.s0);    \
        acc.s1 = mad(src.s1, weight_value, acc.s1);    \
        acc.s2 = mad(src.s2, weight_value, acc.s2);    \
        acc.s3 = mad(src.s3, weight_value, acc.s3);    \
    })

/** An optimized direct convolution 1x1 OpenCL kernel for Bifrost architectures when the data type is F32
 *
 * @note This OpenCL kernel works only with stride_x and stride_y equal to 1
 * @note The third dimensions of the weights tensors must be passed at compile time using -DWEIGHTS_DEPTH
 * @note In case biases, -DHAS_BIAS must to be passed at compile
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                            src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[out] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                            dst_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                            dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in]  weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in]  weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  weights_step_y                        weights_stride_y * number of elements along y processed per workitem(in bytes)
 * @param[in]  weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  weights_step_z                        weights_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in]  biases_ptr                            Pointer to the biases tensor. Same as @p src_ptr
 * @param[in]  biases_stride_x                       Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                         biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes  The offset of the first element in the biases tensor
 * @param[in]  weights_stride_w                      Stride of the weights tensor in the 4th dimension
 */
__kernel void direct_convolution1x1_f32_bifrost(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#ifdef HAS_BIAS
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    unsigned int weights_stride_w)
{
    // Get the kernel index
    const int kernel_index = get_global_id(2);

    Image    src = CONVERT_TO_IMAGE_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    float4 acc0 = 0.0f;
    float4 acc1 = 0.0f;
    float4 acc2 = 0.0f;
    float4 acc3 = 0.0f;

    __global uchar *weights_addr = (__global uchar *)(weights_ptr + weights_offset_first_element_in_bytes + kernel_index * weights_stride_w);
    __global uchar *src_addr     = (__global uchar *)offset(&src, 0, 0);

    for(ushort d = 0; d < (ushort)WEIGHTS_DEPTH; ++d)
    {
        // Load the weights
        float weight = *((__global float *)weights_addr);

        // Load values from row0 of input tensor
        float4 src0 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
        float4 src1 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y));
        float4 src2 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
        float4 src3 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y));

        CONVOLUTION1x1_BIFROST(acc0, src0, weight);
        CONVOLUTION1x1_BIFROST(acc1, src1, weight);
        CONVOLUTION1x1_BIFROST(acc2, src2, weight);
        CONVOLUTION1x1_BIFROST(acc3, src3, weight);

        src_addr += src_stride_z;
        weights_addr += weights_stride_z;
    }

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    float bias = (float) * ((__global float *)(vector_offset(&biases, kernel_index)));

    acc0.s0 += bias;
    acc0.s1 += bias;
    acc0.s2 += bias;
    acc0.s3 += bias;
    acc1.s0 += bias;
    acc1.s1 += bias;
    acc1.s2 += bias;
    acc1.s3 += bias;
    acc2.s0 += bias;
    acc2.s1 += bias;
    acc2.s2 += bias;
    acc2.s3 += bias;
    acc3.s0 += bias;
    acc3.s1 += bias;
    acc3.s2 += bias;
    acc3.s3 += bias;
#endif /* defined(HAS_BIAS) */

    vstore4(acc0, 0, (__global float *)(dst.ptr + 0 * dst_stride_y));
    vstore4(acc1, 0, (__global float *)(dst.ptr + 1 * dst_stride_y));
    vstore4(acc2, 0, (__global float *)(dst.ptr + 2 * dst_stride_y));
    vstore4(acc3, 0, (__global float *)(dst.ptr + 3 * dst_stride_y));
}
#endif // defined(WEIGHTS_DEPTH)