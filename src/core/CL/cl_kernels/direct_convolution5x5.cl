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

#undef CONVERT_SAT

#if defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH)

#if STRIDE_X == 1
#define CONVOLUTION1x5(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x5_STRIDE1(acc, src_row_ptr, weights_row_ptr)
#elif STRIDE_X == 2 /* STRIDE_X == 1 */
#define CONVOLUTION1x5(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x5_STRIDE2(acc, src_row_ptr, weights_row_ptr)
#else /* STRIDE_X not equals 1 or 2 */
#error "STRIDE_X larger than 2 is not supported"
#endif /* STRIDE_X == 2 */

#define CONVOLUTION1x5_STRIDE1(acc, src_row_ptr, weights_row_ptr)                                                               \
    ({                                                                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                             \
        weights_values0          = vload4(0, weights_row_ptr);                                                                  \
        DATA_TYPE weights_value1 = *(weights_row_ptr + 4);                                                                      \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                             \
        src0 = vload8(0, src_row_ptr);                                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                             \
        src1 = vload4(0, src_row_ptr + 8);                                                                                      \
        \
        acc += src0 * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s0;                                                          \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s1234, src0.s567, src1.s0) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s1; \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s234, src0.s567, src1.s01) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s2; \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s345, src0.s67, src1.s012) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s3; \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s45, src0.s67, src1.s0123) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_value1;     \
    })

#define CONVOLUTION1x5_STRIDE2(acc, src_row_ptr, weights_row_ptr)                                                               \
    ({                                                                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                             \
        weights_values0          = vload4(0, weights_row_ptr);                                                                  \
        DATA_TYPE weights_value1 = *(weights_row_ptr + 4);                                                                      \
        VEC_DATA_TYPE(DATA_TYPE, 16)                                                                                            \
        src0 = vload16(0, src_row_ptr);                                                                                         \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                                             \
        src1 = vload4(0, src_row_ptr + 16);                                                                                     \
        acc += src0.even * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s0;                                                     \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s1357, src0.s9BDF) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s1;         \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s2468, src0.sACE, src1.s0) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s2; \
        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s3579, src0.sBDF, src1.s1) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s3; \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s468a, src0.sCE, src1.s02) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_value1;     \
    })

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The third dimensions of the weights tensors must be passed at compile time using -DWEIGHTS_DEPTH
 * @note If biases are used then -DHAS_BIAS has to be passed at compile time
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
__kernel void direct_convolution5x5(
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

    VEC_DATA_TYPE(DATA_TYPE, 8)
    pixels0 = 0;

    __global uchar *weights_addr = (__global uchar *)tensor3D_offset(&weights, 0, 0, 0);
    __global uchar *src_addr     = (__global uchar *)offset(&src, 0, 0);

    const int kernel_index = get_global_id(2);
    weights_addr += kernel_index * weights_stride_w;

    for(volatile int d = 0; d < WEIGHTS_DEPTH; ++d)
    {
        CONVOLUTION1x5(pixels0, (__global DATA_TYPE *)src_addr, (__global DATA_TYPE *)weights_addr);
        CONVOLUTION1x5(pixels0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 1 * weights_stride_y));
        CONVOLUTION1x5(pixels0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 2 * weights_stride_y));
        CONVOLUTION1x5(pixels0, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 3 * weights_stride_y));
        CONVOLUTION1x5(pixels0, (__global DATA_TYPE *)(src_addr + 4 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 4 * weights_stride_y));

        src_addr += src_stride_z;
        weights_addr += weights_stride_z;
    }

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    pixels0 += (VEC_DATA_TYPE(DATA_TYPE, 8)) * ((__global DATA_TYPE *)(vector_offset(&biases, kernel_index)));
#endif /* defined(HAS_BIAS) */

    vstore8(pixels0, 0, (__global DATA_TYPE *)dst.ptr);
}
#endif // defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH)

#if defined(WEIGHTS_DEPTH)

#define CONVOLUTION1x5_BIFROST(acc, src0, weights_row00, weights_row01) \
    ({                                                                  \
        acc.s0 = mad(src0.s0, weights_row00.s0, acc.s0);                \
        acc.s1 = mad(src0.s1, weights_row00.s0, acc.s1);                \
        acc.s2 = mad(src0.s2, weights_row00.s0, acc.s2);                \
        acc.s3 = mad(src0.s3, weights_row00.s0, acc.s3);                \
        acc.s0 = mad(src0.s1, weights_row00.s1, acc.s0);                \
        acc.s1 = mad(src0.s2, weights_row00.s1, acc.s1);                \
        acc.s2 = mad(src0.s3, weights_row00.s1, acc.s2);                \
        acc.s3 = mad(src0.s4, weights_row00.s1, acc.s3);                \
        acc.s0 = mad(src0.s2, weights_row00.s2, acc.s0);                \
        acc.s1 = mad(src0.s3, weights_row00.s2, acc.s1);                \
        acc.s2 = mad(src0.s4, weights_row00.s2, acc.s2);                \
        acc.s3 = mad(src0.s5, weights_row00.s2, acc.s3);                \
        acc.s0 = mad(src0.s3, weights_row00.s3, acc.s0);                \
        acc.s1 = mad(src0.s4, weights_row00.s3, acc.s1);                \
        acc.s2 = mad(src0.s5, weights_row00.s3, acc.s2);                \
        acc.s3 = mad(src0.s6, weights_row00.s3, acc.s3);                \
        acc.s0 = mad(src0.s4, weights_row01, acc.s0);                   \
        acc.s1 = mad(src0.s5, weights_row01, acc.s1);                   \
        acc.s2 = mad(src0.s6, weights_row01, acc.s2);                   \
        acc.s3 = mad(src0.s7, weights_row01, acc.s3);                   \
    })

/** An optimized direct convolution 5x5 OpenCL kernel for Bifrost architectures when the data type is F32
 *
 * @note This OpenCL kernel works only with stride_x and stride_y equal to 1
 * @note The third dimensions of the weights tensors must be passed at compile time using -DWEIGHTS_DEPTH
 * @note If biases are used then -DHAS_BIAS has to be passed at compile time
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
__kernel void direct_convolution5x5_f32_bifrost(
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

    float4 pixels0 = 0.0f;
    float4 pixels1 = 0.0f;

    __global uchar *weights_addr = (__global uchar *)(weights_ptr + weights_offset_first_element_in_bytes + kernel_index * weights_stride_w);
    __global uchar *src_addr     = (__global uchar *)offset(&src, 0, 0);

    // Note: Since each work-item computes 4x2 elements, we need to load 6 rows from the input tensor

    for(ushort d = 0; d < (ushort)WEIGHTS_DEPTH; ++d)
    {
        // Load the weights from row0 and row1
        float4 weights_row00 = vload4(0, (__global float *)(weights_addr + 0 * weights_stride_y));
        float  weights_row01 = *((__global float *)(weights_addr + 0 * weights_stride_y) + 4);
        float4 weights_row10 = vload4(0, (__global float *)(weights_addr + 1 * weights_stride_y));
        float  weights_row11 = *((__global float *)(weights_addr + 1 * weights_stride_y) + 4);
        float8 src0;

        // Load values from row0 of input tensor
        src0 = vload8(0, (__global float *)(src_addr + 0 * src_stride_y));

        // Accumulate
        CONVOLUTION1x5_BIFROST(pixels0, src0, weights_row00, weights_row01);

        // Load values from row1 of input tensor
        src0 = vload8(0, (__global float *)(src_addr + 1 * src_stride_y));

        // Accumulate
        CONVOLUTION1x5_BIFROST(pixels0, src0, weights_row10, weights_row11);
        CONVOLUTION1x5_BIFROST(pixels1, src0, weights_row00, weights_row01);

        // Load values from row2 of input tensor
        src0 = vload8(0, (__global float *)(src_addr + 2 * src_stride_y));

        // Load weights from row2
        weights_row00 = vload4(0, (__global float *)(weights_addr + 2 * weights_stride_y));
        weights_row01 = *((__global float *)(weights_addr + 2 * weights_stride_y) + 4);

        // Accumulate
        CONVOLUTION1x5_BIFROST(pixels0, src0, weights_row00, weights_row01);
        CONVOLUTION1x5_BIFROST(pixels1, src0, weights_row10, weights_row11);

        // Load values from row3 of input tensor
        src0 = vload8(0, (__global float *)(src_addr + 3 * src_stride_y));

        // Load weights from row3
        weights_row10 = vload4(0, (__global float *)(weights_addr + 3 * weights_stride_y));
        weights_row11 = *((__global float *)(weights_addr + 3 * weights_stride_y) + 4);

        // Accumulate
        CONVOLUTION1x5_BIFROST(pixels0, src0, weights_row10, weights_row11);
        CONVOLUTION1x5_BIFROST(pixels1, src0, weights_row00, weights_row01);

        // Load values from row4 of input tensor
        src0 = vload8(0, (__global float *)(src_addr + 4 * src_stride_y));

        // Load weights from row4
        weights_row00 = vload4(0, (__global float *)(weights_addr + 4 * weights_stride_y));
        weights_row01 = *((__global float *)(weights_addr + 4 * weights_stride_y) + 4);

        CONVOLUTION1x5_BIFROST(pixels0, src0, weights_row00, weights_row01);
        CONVOLUTION1x5_BIFROST(pixels1, src0, weights_row10, weights_row11);

        // Load values from row5 of input tensor
        src0 = vload8(0, (__global float *)(src_addr + 5 * src_stride_y));

        // Accumulate
        CONVOLUTION1x5_BIFROST(pixels1, src0, weights_row00, weights_row01);

        src_addr += src_stride_z;
        weights_addr += weights_stride_z;
    }

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    float4 bias = (float4) * ((__global float *)(vector_offset(&biases, kernel_index)));

    pixels0 += bias;
    pixels1 += bias;
#endif /* defined(HAS_BIAS) */

    vstore4(pixels0, 0, (__global float *)(dst.ptr + 0 * dst_stride_y));
    vstore4(pixels1, 0, (__global float *)(dst.ptr + 1 * dst_stride_y));
}
#endif // defined(WEIGHTS_DEPTH)
