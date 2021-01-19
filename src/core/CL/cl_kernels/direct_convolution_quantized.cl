/*
 * Copyright (c) 2017-2021 Arm Limited.
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

#undef CONVERT_SAT_STR
#undef CONVERT_SAT

#if defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT)

#define CONVERT_SAT_STR(x, type) (convert_##type##8_sat((x)))
#define CONVERT_SAT(x, type) CONVERT_SAT_STR(x, type)

#if KERNEL_SIZE == 9

#if STRIDE_X == 1
#define CONVOLUTION1x9(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x9_STRIDE1(acc, src_row_ptr, weights_row_ptr)
#elif STRIDE_X == 2
#define CONVOLUTION1x9(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x9_STRIDE2(acc, src_row_ptr, weights_row_ptr)
#else /* STRIDE_X not equals 1 or 2 */
#error "STRIDE_X larger than 2 is not supported"
#endif /* STRIDE_X */

#define CONVOLUTION1x9_STRIDE1(acc, src_row_ptr, weights_row_ptr)                                             \
    ({                                                                                                        \
        int8  weights_values0 = convert_int8(vload8(0, weights_row_ptr));                                     \
        int   weights_value1  = convert_int(*(weights_row_ptr + 8));                                          \
        int16 src0            = convert_int16(vload16(0, src_row_ptr));                                       \
        acc += (src0.lo + INPUT_OFFSET) * ((int8)weights_values0.s0 + WEIGHTS_OFFSET);                        \
        acc += ((int8)(src0.s1234, src0.s5678) + INPUT_OFFSET) * ((int8)weights_values0.s1 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s2345, src0.s6789) + INPUT_OFFSET) * ((int8)weights_values0.s2 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s3456, src0.s789A) + INPUT_OFFSET) * ((int8)weights_values0.s3 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s4567, src0.s89AB) + INPUT_OFFSET) * ((int8)weights_values0.s4 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s5678, src0.s9ABC) + INPUT_OFFSET) * ((int8)weights_values0.s5 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s6789, src0.sABCD) + INPUT_OFFSET) * ((int8)weights_values0.s6 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s789A, src0.sBCDE) + INPUT_OFFSET) * ((int8)weights_values0.s7 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s89AB, src0.sCDEF) + INPUT_OFFSET) * ((int8)weights_value1 + WEIGHTS_OFFSET);     \
    })

#define CONVOLUTION1x9_STRIDE2(acc, src_row_ptr, weights_row_ptr)                                                     \
    ({                                                                                                                \
        int8  weights_values0 = convert_int8(vload8(0, weights_row_ptr));                                             \
        int   weights_value1  = convert_int(*(weights_row_ptr + 8));                                                  \
        int16 src0            = convert_int16(vload16(0, src_row_ptr));                                               \
        int8  src1            = convert_int8(vload8(0, src_row_ptr + 16));                                            \
        acc += (src0.even + INPUT_OFFSET) * ((int8)weights_values0.s0 + WEIGHTS_OFFSET);                              \
        acc += ((int8)(src0.s1357, src0.s9BDF) + INPUT_OFFSET) * ((int8)weights_values0.s1 + WEIGHTS_OFFSET);         \
        acc += ((int8)(src0.s2468, src0.sACE, src1.s0) + INPUT_OFFSET) * ((int8)weights_values0.s2 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s3579, src0.sBDF, src1.s1) + INPUT_OFFSET) * ((int8)weights_values0.s3 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s468A, src0.sCE, src1.s02) + INPUT_OFFSET) * ((int8)weights_values0.s4 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s579B, src0.sDF, src1.s13) + INPUT_OFFSET) * ((int8)weights_values0.s5 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s68AC, src0.sE, src1.s024) + INPUT_OFFSET) * ((int8)weights_values0.s6 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s79BD, src0.sF, src1.s135) + INPUT_OFFSET) * ((int8)weights_values0.s7 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s8ACE, src1.s0246) + INPUT_OFFSET) * ((int8)weights_value1 + WEIGHTS_OFFSET);             \
    })

#elif KERNEL_SIZE == 5

#if STRIDE_X == 1
#define CONVOLUTION1x5(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x5_STRIDE1(acc, src_row_ptr, weights_row_ptr)
#elif STRIDE_X == 2
#define CONVOLUTION1x5(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x5_STRIDE2(acc, src_row_ptr, weights_row_ptr)
#else /* STRIDE_X not equals 1 or 2 */
#error "STRIDE_X larger than 2 is not supported"
#endif /* STRIDE_X */

#define CONVOLUTION1x5_STRIDE1(acc, src_row_ptr, weights_row_ptr)                                                     \
    ({                                                                                                                \
        int4 weights_values0 = convert_int4(vload4(0, weights_row_ptr));                                              \
        int  weights_value1  = convert_int(*(weights_row_ptr + 4));                                                   \
        int8 src0            = convert_int8(vload8(0, src_row_ptr));                                                  \
        int4 src1            = convert_int4(vload4(0, src_row_ptr + 8));                                              \
        acc += (src0 + INPUT_OFFSET) * ((int8)weights_values0.s0 + WEIGHTS_OFFSET);                                   \
        acc += ((int8)(src0.s1234, src0.s567, src1.s0) + INPUT_OFFSET) * ((int8)weights_values0.s1 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s234, src0.s567, src1.s01) + INPUT_OFFSET) * ((int8)weights_values0.s2 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s345, src0.s67, src1.s012) + INPUT_OFFSET) * ((int8)weights_values0.s3 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s45, src0.s67, src1.s0123) + INPUT_OFFSET) * ((int8)weights_value1 + WEIGHTS_OFFSET);     \
    })

#define CONVOLUTION1x5_STRIDE2(acc, src_row_ptr, weights_row_ptr)                                                     \
    ({                                                                                                                \
        int4  weights_values0 = convert_int4(vload4(0, weights_row_ptr));                                             \
        int   weights_value1  = convert_int(*(weights_row_ptr + 4));                                                  \
        int16 src0            = convert_int16(vload16(0, src_row_ptr));                                               \
        int4  src1            = convert_int4(vload4(0, src_row_ptr + 16));                                            \
        acc += (src0.even + INPUT_OFFSET) * ((int8)weights_values0.s0 + WEIGHTS_OFFSET);                              \
        acc += ((int8)(src0.s1357, src0.s9BDF) + INPUT_OFFSET) * ((int8)weights_values0.s1 + WEIGHTS_OFFSET);         \
        acc += ((int8)(src0.s2468, src0.sACE, src1.s0) + INPUT_OFFSET) * ((int8)weights_values0.s2 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s3579, src0.sBDF, src1.s1) + INPUT_OFFSET) * ((int8)weights_values0.s3 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s468a, src0.sCE, src1.s02) + INPUT_OFFSET) * ((int8)weights_value1 + WEIGHTS_OFFSET);     \
    })

#elif KERNEL_SIZE == 3

#if STRIDE_X == 1
#define CONVOLUTION1x3(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x3_STRIDE1(acc, src_row_ptr, weights_row_ptr)
#elif STRIDE_X == 2
#define CONVOLUTION1x3(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x3_STRIDE2(acc, src_row_ptr, weights_row_ptr)
#else /* STRIDE_X not equals 1 or 2 */
#error "STRIDE_X larger than 2 is not supported"
#endif /* STRIDE_X */

#define CONVOLUTION1x3_STRIDE1(acc, src_row_ptr, weights_row_ptr)                                                     \
    ({                                                                                                                \
        int3 weights_values0 = convert_int3(vload3(0, weights_row_ptr));                                              \
        int8 src0            = convert_int8(vload8(0, src_row_ptr));                                                  \
        int2 src1            = convert_int2(vload2(0, src_row_ptr + 8));                                              \
        acc += (src0 + INPUT_OFFSET) * ((int8)weights_values0.s0 + WEIGHTS_OFFSET);                                   \
        acc += ((int8)(src0.s1234, src0.s567, src1.s0) + INPUT_OFFSET) * ((int8)weights_values0.s1 + WEIGHTS_OFFSET); \
        acc += ((int8)(src0.s234, src0.s567, src1.s01) + INPUT_OFFSET) * ((int8)weights_values0.s2 + WEIGHTS_OFFSET); \
    })

#define CONVOLUTION1x3_STRIDE2(acc, src_row_ptr, weights_row_ptr)                                                  \
    ({                                                                                                             \
        int3  weights_values0 = convert_int3(vload3(0, weights_row_ptr));                                          \
        int16 src0            = convert_int16(vload16(0, src_row_ptr));                                            \
        int   src1            = convert_int(*(src_row_ptr + 16));                                                  \
        acc += (src0.even + INPUT_OFFSET) * ((int8)weights_values0.s0 + WEIGHTS_OFFSET);                           \
        acc += ((int8)(src0.s1357, src0.s9BDF) + INPUT_OFFSET) * ((int8)weights_values0.s1 + WEIGHTS_OFFSET);      \
        acc += ((int8)(src0.s2468, src0.sACE, src1) + INPUT_OFFSET) * ((int8)weights_values0.s2 + WEIGHTS_OFFSET); \
    })

#elif KERNEL_SIZE == 1

#if STRIDE_X == 3
#define INPUT_VALUE extract_input_stride3
#elif STRIDE_X == 2
#define INPUT_VALUE extract_input_stride2
#elif STRIDE_X == 1
#define INPUT_VALUE extract_input_stride1

#else /* STRIDE_X not equals 1, 2 or 3 */
#error "Only support strides 1, 2 and 3"
#endif /* STRIDE_X */

/** Extracts a 1D horizontal vector from the input tensor with stride as 1.
 *
 * @param[in] input_value Pointer to the first value.
 *
 * @return extracted input values.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) extract_input_stride1(__global const DATA_TYPE *input_value)
{
    return vload8(0, input_value);
}

/** Extracts a 1D horizontal vector from the input tensor with stride as 2.
 *
 * @param[in] input_value Pointer to the first value.
 *
 * @return extracted input values.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) extract_input_stride2(__global const DATA_TYPE *input_value)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    temp = vload16(0, input_value);
    return temp.s02468ace;
}

/** Extracts a 1D horizontal vector from the input tensor with stride as 3 and 8-bit data size.
 *
 * @param[in] input_value Pointer to the first value.
 *
 * @return extracted input values.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) extract_input_stride3(__global const DATA_TYPE *input_value)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    temp1 = vload16(0, input_value);
    VEC_DATA_TYPE(DATA_TYPE, 16)
    temp2 = vload16(0, input_value + 12);
    return (VEC_DATA_TYPE(DATA_TYPE, 8))(temp1.s0369, temp2.s0369);
}

#else /* KERNEL_SIZE not equals 1, 3 , 5, 9 */
#error "Only kernel sizes 1, 3, 5 and 9 are supported"
#endif /* KERNEL_SIZE */

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The convolution stride x must be passed at compile time using -DSTRIDE_X e.g. -DSTRIDE_X=1
 * @note The third dimensions of the weights tensors must be passed at compile time using -DWEIGHTS_DEPTH
 * @note If biases are used then -DHAS_BIAS has to be passed at compile time
 * @note The output quantization multiplier must be passed at compile time using -DOUTPUT_MULTIPLIER e.g. -DOUTPUT_MULTIPLIER=1234
 * @note The output quantization shift must be passed at compile time using -DOUTPUT_SHIFT e.g. -DOUTPUT_SHIFT=4
 * @note The input offset quantization parameter must be passed at compile time using -DINPUT_OFFSET e.g. -DINPUT_OFFSET=3
 * @note The weights offset quantization parameter must be passed at compile time using -DWEIGHTS_OFFSET e.g. -DWEIGHTS_OFFSET=3
 * @note The destination offset quantization parameter must be passed at compile time using -DOUTPUT_OFFSET e.g. -DOUTPUT_OFFSET=3
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED
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
 * @param[in]  biases_ptr                            Pointer to the biases tensor. Supported data types: S32
 * @param[in]  biases_stride_x                       Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                         biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes  The offset of the first element in the biases tensor
 * @param[in]  weights_stride_w                      Stride of the weights tensor in the 4th dimension
 */
__kernel void direct_convolution_quantized(
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

    int8 values0 = 0;

    __global DATA_TYPE *weights_addr = (__global DATA_TYPE *)tensor3D_offset(&weights, 0, 0, 0);
    __global DATA_TYPE *src_addr     = (__global DATA_TYPE *)offset(&src, 0, 0);

    const int kernel_index = get_global_id(2);
    weights_addr += kernel_index * weights_stride_w;

    for(volatile int d = 0; d < WEIGHTS_DEPTH; ++d)
    {
#if KERNEL_SIZE == 9
        CONVOLUTION1x9(values0, (__global DATA_TYPE *)(src_addr + 0 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 0 * weights_stride_y));
        CONVOLUTION1x9(values0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 1 * weights_stride_y));
        CONVOLUTION1x9(values0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 2 * weights_stride_y));
        CONVOLUTION1x9(values0, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 3 * weights_stride_y));
        CONVOLUTION1x9(values0, (__global DATA_TYPE *)(src_addr + 4 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 4 * weights_stride_y));
        CONVOLUTION1x9(values0, (__global DATA_TYPE *)(src_addr + 5 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 5 * weights_stride_y));
        CONVOLUTION1x9(values0, (__global DATA_TYPE *)(src_addr + 6 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 6 * weights_stride_y));
        CONVOLUTION1x9(values0, (__global DATA_TYPE *)(src_addr + 7 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 7 * weights_stride_y));
        CONVOLUTION1x9(values0, (__global DATA_TYPE *)(src_addr + 8 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 8 * weights_stride_y));
#elif KERNEL_SIZE == 5
        CONVOLUTION1x5(values0, (__global DATA_TYPE *)src_addr, (__global DATA_TYPE *)weights_addr);
        CONVOLUTION1x5(values0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 1 * weights_stride_y));
        CONVOLUTION1x5(values0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 2 * weights_stride_y));
        CONVOLUTION1x5(values0, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 3 * weights_stride_y));
        CONVOLUTION1x5(values0, (__global DATA_TYPE *)(src_addr + 4 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 4 * weights_stride_y));
#elif KERNEL_SIZE == 3
        CONVOLUTION1x3(values0, (__global DATA_TYPE *)(src_addr + 0 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 0 * weights_stride_y));
        CONVOLUTION1x3(values0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 1 * weights_stride_y));
        CONVOLUTION1x3(values0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 2 * weights_stride_y));
#elif KERNEL_SIZE == 1
        int weight       = convert_int(*(__global DATA_TYPE *)weights_addr);
        int8 input_value = convert_int8(INPUT_VALUE((__global DATA_TYPE *)src_addr));
        values0 += (input_value + INPUT_OFFSET) * ((int8)weight + WEIGHTS_OFFSET);
#endif /* (KERNEL_SIZE == 1) || (KERNEL_SIZE == 3) || (KERNEL_SIZE == 5) */

        src_addr += src_stride_z;
        weights_addr += weights_stride_z;
    }

#ifdef HAS_BIAS
    Vector        biases    = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
    __global int *bias_addr = ((__global int *)(vector_offset(&biases, kernel_index)));
    values0 += (int8)(*bias_addr);
#endif /* defined(HAS_BIAS) */

#if OUTPUT_SHIFT < 0
    values0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(values0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#else  // OUTPUT_SHIFT < 0
    values0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#endif // OUTPUT_SHIFT < 0
    values0 = values0 + OUTPUT_OFFSET;

    vstore8(CONVERT_SAT(values0, DATA_TYPE), 0, (__global DATA_TYPE *)dst.ptr);
}
#endif // defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT)
