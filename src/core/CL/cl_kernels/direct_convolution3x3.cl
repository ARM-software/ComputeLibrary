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

#if defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH)

#if STRIDE_X == 1
#define CONVOLUTION1x3(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x3_STRIDE1(acc, src_row_ptr, weights_row_ptr)
#elif STRIDE_X == 2 /* STRIDE_X == 1 */
#define CONVOLUTION1x3(acc, src_row_ptr, weights_row_ptr) CONVOLUTION1x3_STRIDE2(acc, src_row_ptr, weights_row_ptr)
#else /* STRIDE_X not equals 1 or 2 */
#error "STRIDE_X larger than 2 is not supported"
#endif /* STRIDE_X == 2 */

#define CONVOLUTION1x3_STRIDE1(acc, src_row_ptr, weights_row_ptr)                                                                                  \
    ({                                                                                                                                             \
        VEC_DATA_TYPE(DATA_TYPE, 3)                                                                                                                \
        weights_values0 = vload3(0, weights_row_ptr);                                                                                              \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                                \
        src0 = vload8(0, src_row_ptr);                                                                                                             \
        VEC_DATA_TYPE(DATA_TYPE, 2)                                                                                                                \
        src1 = vload2(0, src_row_ptr + 8);                                                                                                         \
        \
        acc = ADD_OP(acc, MUL_OP(src0, (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s0));                                                          \
        acc = ADD_OP(acc, MUL_OP((VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s1234, src0.s567, src1.s0), (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s1)); \
        acc = ADD_OP(acc, MUL_OP((VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s234, src0.s567, src1.s01), (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s2)); \
    })

#define CONVOLUTION1x3_STRIDE2(acc, src_row_ptr, weights_row_ptr)                                                                               \
    ({                                                                                                                                          \
        VEC_DATA_TYPE(DATA_TYPE, 3)                                                                                                             \
        weights_values0 = vload3(0, weights_row_ptr);                                                                                           \
        VEC_DATA_TYPE(DATA_TYPE, 16)                                                                                                            \
        src0           = vload16(0, src_row_ptr);                                                                                               \
        DATA_TYPE src1 = *(src_row_ptr + 16);                                                                                                   \
        \
        acc = ADD_OP(acc, MUL_OP(src0.even, (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s0));                                                  \
        acc = ADD_OP(acc, MUL_OP((VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s1357, src0.s9BDF), (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s1));      \
        acc = ADD_OP(acc, MUL_OP((VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s2468, src0.sACE, src1), (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s2)); \
    })

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note This OpenCL kernel works with stride_x = 1 and 2
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The third dimensions of the weights tensors must be passed at compile time using -DWEIGHTS_DEPTH
 * @note If biases are used then -DHAS_BIAS has to be passed at compile time
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: QS8/QS16/F16/F32
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
__kernel void direct_convolution3x3(
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

    VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 8)
    pixels0 = 0;

    __global uchar *weights_addr = (__global uchar *)tensor3D_offset(&weights, 0, 0, 0);
    __global uchar *src_addr     = (__global uchar *)offset(&src, 0, 0);

    const int kernel_index = get_global_id(2);
    weights_addr += kernel_index * weights_stride_w;

    for(volatile int d = 0; d < WEIGHTS_DEPTH; ++d)
    {
        CONVOLUTION1x3(pixels0, (__global DATA_TYPE *)(src_addr + 0 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 0 * weights_stride_y));
        CONVOLUTION1x3(pixels0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 1 * weights_stride_y));
        CONVOLUTION1x3(pixels0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y), (__global DATA_TYPE *)(weights_addr + 2 * weights_stride_y));

        src_addr += src_stride_z;
        weights_addr += weights_stride_z;
    }

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    pixels0 = ADD_OP(pixels0, (VEC_DATA_TYPE(DATA_TYPE_PROMOTED, 8)) * ((__global DATA_TYPE *)(vector_offset(&biases, kernel_index))));
#endif /* defined(HAS_BIAS) */

    vstore8(CONVERT_SAT(pixels0, VEC_DATA_TYPE(DATA_TYPE, 8)), 0, (__global DATA_TYPE *)dst.ptr);
}
#endif //defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH)

#if defined(WEIGHTS_DEPTH)

#define CONVOLUTION1x3_BIFROST(acc, src0, src1, weights_row0) \
    ({                                                        \
        acc.s0 = mad(src0.s0, weights_row0.s0, acc.s0);       \
        acc.s1 = mad(src0.s1, weights_row0.s0, acc.s1);       \
        acc.s2 = mad(src0.s2, weights_row0.s0, acc.s2);       \
        acc.s3 = mad(src0.s3, weights_row0.s0, acc.s3);       \
        acc.s0 = mad(src0.s1, weights_row0.s1, acc.s0);       \
        acc.s1 = mad(src0.s2, weights_row0.s1, acc.s1);       \
        acc.s2 = mad(src0.s3, weights_row0.s1, acc.s2);       \
        acc.s3 = mad(src1.s0, weights_row0.s1, acc.s3);       \
        acc.s0 = mad(src0.s2, weights_row0.s2, acc.s0);       \
        acc.s1 = mad(src0.s3, weights_row0.s2, acc.s1);       \
        acc.s2 = mad(src1.s0, weights_row0.s2, acc.s2);       \
        acc.s3 = mad(src1.s1, weights_row0.s2, acc.s3);       \
    })

/** An optimized direct convolution 3x3 OpenCL kernel for Bifrost architectures when the data type is F32
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
__kernel void direct_convolution3x3_f32_bifrost(
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

    float4 pixels0 = 0;
    float4 pixels1 = 0;
    float4 pixels2 = 0;

    __global uchar *weights_addr = (__global uchar *)(weights_ptr + weights_offset_first_element_in_bytes + kernel_index * weights_stride_w);
    __global uchar *src_addr     = (__global uchar *)offset(&src, 0, 0);

    // Note: Since each work-item computes 4x3 elements, we need to load 5 rows from the input tensor

    for(ushort d = 0; d < (ushort)WEIGHTS_DEPTH; ++d)
    {
        // Load the weights
        float3 weights_row0 = vload3(0, (__global float *)(weights_addr + 0 * weights_stride_y));
        float3 weights_row1 = vload3(0, (__global float *)(weights_addr + 1 * weights_stride_y));
        float3 weights_row2 = vload3(0, (__global float *)(weights_addr + 2 * weights_stride_y));
        float4 src0;
        float2 src1;

        // Load values from row0 of input tensor
        src0 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
        src1 = vload2(0, (__global float *)(src_addr + 0 * src_stride_y) + 4);

        CONVOLUTION1x3_BIFROST(pixels0, src0, src1, weights_row0);

        // Load values from row1 of input tensor
        src0 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y));
        src1 = vload2(0, (__global float *)(src_addr + 1 * src_stride_y) + 4);

        // Accumulate
        CONVOLUTION1x3_BIFROST(pixels0, src0, src1, weights_row1);
        CONVOLUTION1x3_BIFROST(pixels1, src0, src1, weights_row0);

        // Load values from row2 of input tensor
        src0 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
        src1 = vload2(0, (__global float *)(src_addr + 2 * src_stride_y) + 4);

        // Accumulate
        CONVOLUTION1x3_BIFROST(pixels0, src0, src1, weights_row2);
        CONVOLUTION1x3_BIFROST(pixels1, src0, src1, weights_row1);
        CONVOLUTION1x3_BIFROST(pixels2, src0, src1, weights_row0);

        // Load values from row3 of input tensor
        src0 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y));
        src1 = vload2(0, (__global float *)(src_addr + 3 * src_stride_y) + 4);

        // Accumulate
        CONVOLUTION1x3_BIFROST(pixels1, src0, src1, weights_row2);
        CONVOLUTION1x3_BIFROST(pixels2, src0, src1, weights_row1);

        // Row4
        src0 = vload4(0, (__global float *)(src_addr + 4 * src_stride_y));
        src1 = vload2(0, (__global float *)(src_addr + 4 * src_stride_y) + 4);

        // Accumulate
        CONVOLUTION1x3_BIFROST(pixels2, src0, src1, weights_row2);

        src_addr += src_stride_z;
        weights_addr += weights_stride_z;
    }

#ifdef HAS_BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    float bias = (float) * ((__global float *)(vector_offset(&biases, kernel_index)));

    pixels0 += (float4)bias;
    pixels1 += (float4)bias;
    pixels2 += (float4)bias;
#endif /* defined(HAS_BIAS) */

    vstore4(pixels0, 0, (__global float *)(dst.ptr + 0 * dst_stride_y));
    vstore4(pixels1, 0, (__global float *)(dst.ptr + 1 * dst_stride_y));
    vstore4(pixels2, 0, (__global float *)(dst.ptr + 2 * dst_stride_y));
}
#endif // defined(WEIGHTS_DEPTH)
