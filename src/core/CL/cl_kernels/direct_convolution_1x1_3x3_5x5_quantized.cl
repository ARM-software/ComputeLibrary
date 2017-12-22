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
#include "helpers_asymm.h"

#undef CONVERT_SAT

#if defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH)

#if KERNEL_SIZE == 5

#if STRIDE_X == 1
#define CONVOLUTION1x5(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x5_STRIDE1(acc, src_row_ptr, weights_row_ptr)
#elif STRIDE_X == 2
#define CONVOLUTION1x5(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x5_STRIDE2(acc, src_row_ptr, weights_row_ptr)
#else /* STRIDE_X not equals 1 or 2 */
#error "STRIDE_X larger than 2 is not supported"
#endif /* STRIDE_X */

#define CONVOLUTION1x5_STRIDE1(acc, src_row_ptr, weights_row_ptr)                                                    \
    ({                                                                                                               \
        int4 weights_values0 = convert_int4(vload4(0, weights_row_ptr));                                             \
        int  weights_value1  = convert_int(*(weights_row_ptr + 4));                                                  \
        int8 src0            = convert_int8(vload8(0, src_row_ptr));                                                 \
        int4 src1            = convert_int4(vload4(0, src_row_ptr + 8));                                             \
        acc += (src0 + input_offset) * ((int8)weights_values0.s0 + weight_offset);                                   \
        acc += ((int8)(src0.s1234, src0.s567, src1.s0) + input_offset) * ((int8)weights_values0.s1 + weight_offset); \
        acc += ((int8)(src0.s234, src0.s567, src1.s01) + input_offset) * ((int8)weights_values0.s2 + weight_offset); \
        acc += ((int8)(src0.s345, src0.s67, src1.s012) + input_offset) * ((int8)weights_values0.s3 + weight_offset); \
        acc += ((int8)(src0.s45, src0.s67, src1.s0123) + input_offset) * ((int8)weights_value1 + weight_offset);     \
    })

#define CONVOLUTION1x5_STRIDE2(acc, src_row_ptr, weights_row_ptr)                                                    \
    ({                                                                                                               \
        int4  weights_values0 = convert_int4(vload4(0, weights_row_ptr));                                            \
        int   weights_value1  = convert_int(*(weights_row_ptr + 4));                                                 \
        int16 src0            = convert_int16(vload16(0, src_row_ptr));                                              \
        int4  src1            = convert_int4(vload4(0, src_row_ptr + 16));                                           \
        acc += (src0.even + input_offset) * ((int8)weights_values0.s0 + weight_offset);                              \
        acc += ((int8)(src0.s1357, src0.s9BDF) + input_offset) * ((int8)weights_values0.s1 + weight_offset);         \
        acc += ((int8)(src0.s2468, src0.sACE, src1.s0) + input_offset) * ((int8)weights_values0.s2 + weight_offset); \
        acc += ((int8)(src0.s3579, src0.sBDF, src1.s1) + input_offset) * ((int8)weights_values0.s3 + weight_offset); \
        acc += ((int8)(src0.s468a, src0.sCE, src1.s02) + input_offset) * ((int8)weights_value1 + weight_offset);     \
    })

#elif KERNEL_SIZE == 3

#if STRIDE_X == 1
#define CONVOLUTION1x3(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x3_STRIDE1(acc, src_row_ptr, weights_row_ptr)
#elif STRIDE_X == 2
#define CONVOLUTION1x3(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x3_STRIDE2(acc, src_row_ptr, weights_row_ptr)
#else /* STRIDE_X not equals 1 or 2 */
#error "STRIDE_X larger than 2 is not supported"
#endif /* STRIDE_X */

#define CONVOLUTION1x3_STRIDE1(acc, src_row_ptr, weights_row_ptr)                                                    \
    ({                                                                                                               \
        int3 weights_values0 = convert_int3(vload3(0, weights_row_ptr));                                             \
        int8 src0            = convert_int8(vload8(0, src_row_ptr));                                                 \
        int2 src1            = convert_int2(vload2(0, src_row_ptr + 8));                                             \
        acc += (src0 + input_offset) * ((int8)weights_values0.s0 + weight_offset);                                   \
        acc += ((int8)(src0.s1234, src0.s567, src1.s0) + input_offset) * ((int8)weights_values0.s1 + weight_offset); \
        acc += ((int8)(src0.s234, src0.s567, src1.s01) + input_offset) * ((int8)weights_values0.s2 + weight_offset); \
    })

#define CONVOLUTION1x3_STRIDE2(acc, src_row_ptr, weights_row_ptr)                                                 \
    ({                                                                                                            \
        int3  weights_values0 = convert_int3(vload3(0, weights_row_ptr));                                         \
        int16 src0            = convert_int16(vload16(0, src_row_ptr));                                           \
        int   src1            = convert_int(*(src_row_ptr + 16));                                                 \
        acc += (src0.even + input_offset) * ((int8)weights_values0.s0 + weight_offset);                           \
        acc += ((int8)(src0.s1357, src0.s9BDF) + input_offset) * ((int8)weights_values0.s1 + weight_offset);      \
        acc += ((int8)(src0.s2468, src0.sACE, src1) + input_offset) * ((int8)weights_values0.s2 + weight_offset); \
    })

#elif KERNEL_SIZE == 1

#if STRIDE_X == 3
#define INPUT_PIXEL extract_input_stride3
#elif STRIDE_X == 2
#define INPUT_PIXEL extract_input_stride2
#elif STRIDE_X == 1
#define INPUT_PIXEL extract_input_stride1

#else /* STRIDE_X not equals 1, 2 or 3 */
#error "Only support strides 1, 2 and 3"
#endif /* STRIDE_X */

/** Extracts a 1D horizontal vector from the input tensor with stride as 1.
 *
 * @param[in] input_pixel Pointer to the first pixel.
 *
 * @return extracted input pixels.
 */
inline uchar8 extract_input_stride1(__global const uchar *input_pixel)
{
    return vload8(0, input_pixel);
}

/** Extracts a 1D horizontal vector from the input tensor with stride as 2.
 *
 * @param[in] input_pixel Pointer to the first pixel.
 *
 * @return extracted input pixels.
 */
inline uchar8 extract_input_stride2(__global const uchar *input_pixel)
{
    uchar16 temp = vload16(0, input_pixel);
    return temp.s02468ace;
}

/** Extracts a 1D horizontal vector from the input tensor with stride as 3 and 8-bit data size.
 *
 * @param[in] input_pixel Pointer to the first pixel.
 *
 * @return extracted input pixels.
 */
inline uchar8 extract_input_stride3(__global const uchar *input_pixel)
{
    uchar16 temp1 = vload16(0, input_pixel);
    uchar16 temp2 = vload16(0, input_pixel + 12);
    return (uchar8)(temp1.s0369, temp2.s0369);
}

#else /* KERNEL_SIZE not equals 1, 3 or 5 */
#error "Only kernel sizes 1, 3 and 5 are supported"
#endif /* KERNEL_SIZE */

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The convolution stride x must be passed at compile time using -DSTRIDE_X e.g. -DSTRIDE_X=1
 * @note The third dimensions of the weights tensors must be passed at compile time using -DWEIGHTS_DEPTH
 * @note If biases are used then -DHAS_BIAS has to be passed at compile time
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: QASYMM8
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
 * @param[in]  weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p weights_ptr
 * @param[in]  weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  weights_step_y                        weights_stride_y * number of elements along y processed per workitem(in bytes)
 * @param[in]  weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  weights_step_z                        weights_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in]  biases_ptr                            Pointer to the biases tensor. Supported data types: S32
 * @param[in]  biases_stride_x                       Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                         biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes  The offset of the first element in the biases tensor
 * @param[in]  weights_stride_w                      Stride of the weights tensor in the 4th dimension
 * @param[in]  input_offset                          Input offset quantization parameter
 * @param[in]  weight_offset                         Weights offset quantization parameter
 * @param[in]  output_offset                         Output offset quantization parameter
 * @param[in]  output_multiplier                     Output integer multiplier quantization parameter
 * @param[in]  output_shift                          Output integer shift quantization parameter
 */
__kernel void direct_convolution_1x1_3x3_5x5_quantized(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#ifdef HAS_BIAS
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    unsigned int weights_stride_w,
    int          input_offset,
    int          weight_offset,
    int          output_offset,
    int          output_multiplier,
    int          output_shift)
{
    Image    src     = CONVERT_TO_IMAGE_STRUCT(src);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);
    Tensor3D dst     = CONVERT_TO_TENSOR3D_STRUCT(dst);

    int8 pixels0 = 0;

    __global uchar *weights_addr = (__global uchar *)tensor3D_offset(&weights, 0, 0, 0);
    __global uchar *src_addr     = (__global uchar *)offset(&src, 0, 0);

    const int kernel_index = get_global_id(2);
    weights_addr += kernel_index * weights_stride_w;

    for(volatile int d = 0; d < WEIGHTS_DEPTH; ++d)
    {
#if KERNEL_SIZE == 5
        CONVOLUTION1x5(pixels0, (__global uchar *)src_addr, (__global uchar *)weights_addr);
        CONVOLUTION1x5(pixels0, (__global uchar *)(src_addr + 1 * src_stride_y), (__global uchar *)(weights_addr + 1 * weights_stride_y));
        CONVOLUTION1x5(pixels0, (__global uchar *)(src_addr + 2 * src_stride_y), (__global uchar *)(weights_addr + 2 * weights_stride_y));
        CONVOLUTION1x5(pixels0, (__global uchar *)(src_addr + 3 * src_stride_y), (__global uchar *)(weights_addr + 3 * weights_stride_y));
        CONVOLUTION1x5(pixels0, (__global uchar *)(src_addr + 4 * src_stride_y), (__global uchar *)(weights_addr + 4 * weights_stride_y));
#elif KERNEL_SIZE == 3
        CONVOLUTION1x3(pixels0, (__global uchar *)(src_addr + 0 * src_stride_y), (__global uchar *)(weights_addr + 0 * weights_stride_y));
        CONVOLUTION1x3(pixels0, (__global uchar *)(src_addr + 1 * src_stride_y), (__global uchar *)(weights_addr + 1 * weights_stride_y));
        CONVOLUTION1x3(pixels0, (__global uchar *)(src_addr + 2 * src_stride_y), (__global uchar *)(weights_addr + 2 * weights_stride_y));
#elif KERNEL_SIZE == 1
        int weight       = convert_int(*(__global uchar *)weights_addr);
        int8 input_pixel = convert_int8(INPUT_PIXEL((__global uchar *)src_addr));
        pixels0 += (input_pixel + input_offset) * ((int8)weight + weight_offset);
#endif /* (KERNEL_SIZE == 1) || (KERNEL_SIZE == 3) || (KERNEL_SIZE == 5) */

        src_addr += src_stride_z;
        weights_addr += weights_stride_z;
    }

#ifdef HAS_BIAS
    Vector        biases    = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
    __global int *bias_addr = ((__global int *)(vector_offset(&biases, kernel_index)));
    pixels0 += (int8)(*bias_addr);
#endif /* defined(HAS_BIAS) */

    pixels0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(pixels0, output_multiplier, output_shift, 8);
    pixels0 = pixels0 + output_offset;

    vstore8(convert_uchar8_sat(pixels0), 0, (__global uchar *)dst.ptr);
}
#endif // defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH)
