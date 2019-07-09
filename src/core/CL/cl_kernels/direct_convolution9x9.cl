/*
 * Copyright (c) 2019 ARM Limited.
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

#if defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH) && defined(DATA_LAYOUT_NHWC)

#define PTR_TO_VALUE(PTR, DATA_TYPE) *((__global DATA_TYPE *)(PTR))

#define CONVOLUTION1x9_STRIDE1_NHWC(acc, row_ptr, weights_ptr)                                                                         \
    ({                                                                                                                                 \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                    \
        src0 = (VEC_DATA_TYPE(DATA_TYPE, 8))(                                                                                          \
                PTR_TO_VALUE(row_ptr + 0 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 1 * src_stride_y, DATA_TYPE),                  \
                PTR_TO_VALUE(row_ptr + 2 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 3 * src_stride_y, DATA_TYPE),                  \
                PTR_TO_VALUE(row_ptr + 4 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 5 * src_stride_y, DATA_TYPE),                  \
                PTR_TO_VALUE(row_ptr + 6 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 7 * src_stride_y, DATA_TYPE));                 \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                    \
        src1 = (VEC_DATA_TYPE(DATA_TYPE, 8))(                                                                                          \
                PTR_TO_VALUE(row_ptr + 8 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 9 * src_stride_y, DATA_TYPE),                  \
                PTR_TO_VALUE(row_ptr + 10 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 11 * src_stride_y, DATA_TYPE),                \
                PTR_TO_VALUE(row_ptr + 12 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 13 * src_stride_y, DATA_TYPE),                \
                PTR_TO_VALUE(row_ptr + 14 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 15 * src_stride_y, DATA_TYPE));               \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                    \
        weights_values0 = (VEC_DATA_TYPE(DATA_TYPE, 8))(                                                                               \
                          PTR_TO_VALUE(weights_ptr + 0 * weights_stride_y, DATA_TYPE), PTR_TO_VALUE(weights_ptr + 1 * weights_stride_y, DATA_TYPE),  \
                          PTR_TO_VALUE(weights_ptr + 2 * weights_stride_y, DATA_TYPE), PTR_TO_VALUE(weights_ptr + 3 * weights_stride_y, DATA_TYPE),  \
                          PTR_TO_VALUE(weights_ptr + 4 * weights_stride_y, DATA_TYPE), PTR_TO_VALUE(weights_ptr + 5 * weights_stride_y, DATA_TYPE),  \
                          PTR_TO_VALUE(weights_ptr + 6 * weights_stride_y, DATA_TYPE), PTR_TO_VALUE(weights_ptr + 7 * weights_stride_y, DATA_TYPE)); \
        DATA_TYPE weights_value1 = PTR_TO_VALUE(weights_ptr + 8 * weights_stride_y, DATA_TYPE);                                        \
        acc += src0 * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s0;                                                                 \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s1234, src0.s567, src1.s0) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s1;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s234, src0.s567, src1.s01) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s2;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s345, src0.s67, src1.s012) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s3;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s4567, src1.s0123) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s4;                \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s567, src1.s0123, src1.s4) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s5;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s67, src1.s012, src1.s345) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s6;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s7, src1.s0123, src1.s456) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s7;        \
        acc += src1 * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_value1;                                                                     \
    })

#define CONVOLUTION1x9_STRIDE2_NHWC(acc, row_ptr, weights_ptr)                                                                         \
    ({                                                                                                                                 \
        VEC_DATA_TYPE(DATA_TYPE, 16)                                                                                                   \
        src0 = (VEC_DATA_TYPE(DATA_TYPE, 16))(                                                                                         \
                PTR_TO_VALUE(row_ptr + 0 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 1 * src_stride_y, DATA_TYPE),                  \
                PTR_TO_VALUE(row_ptr + 2 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 3 * src_stride_y, DATA_TYPE),                  \
                PTR_TO_VALUE(row_ptr + 4 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 5 * src_stride_y, DATA_TYPE),                  \
                PTR_TO_VALUE(row_ptr + 6 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 7 * src_stride_y, DATA_TYPE),                  \
                PTR_TO_VALUE(row_ptr + 8 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 9 * src_stride_y, DATA_TYPE),                  \
                PTR_TO_VALUE(row_ptr + 10 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 11 * src_stride_y, DATA_TYPE),                \
                PTR_TO_VALUE(row_ptr + 12 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 13 * src_stride_y, DATA_TYPE),                \
                PTR_TO_VALUE(row_ptr + 14 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 15 * src_stride_y, DATA_TYPE));               \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                    \
        src1 = (VEC_DATA_TYPE(DATA_TYPE, 8))(                                                                                          \
                PTR_TO_VALUE(row_ptr + 16 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 17 * src_stride_y, DATA_TYPE),                \
                PTR_TO_VALUE(row_ptr + 18 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 19 * src_stride_y, DATA_TYPE),                \
                PTR_TO_VALUE(row_ptr + 20 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 21 * src_stride_y, DATA_TYPE),                \
                PTR_TO_VALUE(row_ptr + 22 * src_stride_y, DATA_TYPE), PTR_TO_VALUE(row_ptr + 23 * src_stride_y, DATA_TYPE));               \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                                                                                    \
        weights_values0 = (VEC_DATA_TYPE(DATA_TYPE, 8))(                                                                               \
                          PTR_TO_VALUE(weights_ptr + 0 * weights_stride_y, DATA_TYPE), PTR_TO_VALUE(weights_ptr + 1 * weights_stride_y, DATA_TYPE),  \
                          PTR_TO_VALUE(weights_ptr + 2 * weights_stride_y, DATA_TYPE), PTR_TO_VALUE(weights_ptr + 3 * weights_stride_y, DATA_TYPE),  \
                          PTR_TO_VALUE(weights_ptr + 4 * weights_stride_y, DATA_TYPE), PTR_TO_VALUE(weights_ptr + 5 * weights_stride_y, DATA_TYPE),  \
                          PTR_TO_VALUE(weights_ptr + 6 * weights_stride_y, DATA_TYPE), PTR_TO_VALUE(weights_ptr + 7 * weights_stride_y, DATA_TYPE)); \
        DATA_TYPE weights_value1 = PTR_TO_VALUE(weights_ptr + 8 * weights_stride_y, DATA_TYPE);                                        \
        acc += src0.s02468ACE * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s0;                                                       \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s1357, src0.s9BDF) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s1;                \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s2468, src0.sACE, src1.s0) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s2;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s3579, src0.sBDF, src1.s1) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s3;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s468A, src0.sCE, src1.s02) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s4;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s579, src0.sBDF, src1.s13) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s5;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s68A, src0.sCE, src1.s024) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s6;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s79B, src0.sDF, src1.s135) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_values0.s7;        \
        acc += (VEC_DATA_TYPE(DATA_TYPE, 8))(src0.s8AC, src0.sE, src1.s0246) * (VEC_DATA_TYPE(DATA_TYPE, 8))weights_value1;            \
    })

#if defined(VEC_SIZE)
#define VFMA(acc, w, src0, src1, src2, src3, src4, src5, src6, src7) \
    ({                                                               \
        acc##0 = fma(src0, w, acc##0);                               \
        acc##1 = fma(src1, w, acc##1);                               \
        acc##2 = fma(src2, w, acc##2);                               \
        acc##3 = fma(src3, w, acc##3);                               \
        acc##4 = fma(src4, w, acc##4);                               \
        acc##5 = fma(src5, w, acc##5);                               \
        acc##6 = fma(src6, w, acc##6);                               \
        acc##7 = fma(src7, w, acc##7);                               \
    })

#define CONVOLUTION1x9_STRIDE1_NHWC_BIFROST(acc, row_ptr, weights_ptr)                       \
    ({                                                                                       \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)row_ptr);                            \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src1 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + src_stride_y));           \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src2 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 2 * src_stride_y));       \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src3 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 3 * src_stride_y));       \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src4 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 4 * src_stride_y));       \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src5 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 5 * src_stride_y));       \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src6 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 6 * src_stride_y));       \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src7 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 7 * src_stride_y));       \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src8 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 8 * src_stride_y));       \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src9 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 9 * src_stride_y));       \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src10 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 10 * src_stride_y));     \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src11 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 11 * src_stride_y));     \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src12 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 12 * src_stride_y));     \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src13 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 13 * src_stride_y));     \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src14 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 14 * src_stride_y));     \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        src15 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(row_ptr + 15 * src_stride_y));     \
        \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        w0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights_ptr + 0 * weights_stride_y)); \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        w1 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights_ptr + 1 * weights_stride_y)); \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        w2 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights_ptr + 2 * weights_stride_y)); \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        w3 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights_ptr + 3 * weights_stride_y)); \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        w4 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights_ptr + 4 * weights_stride_y)); \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        w5 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights_ptr + 5 * weights_stride_y)); \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        w6 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights_ptr + 6 * weights_stride_y)); \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        w7 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights_ptr + 7 * weights_stride_y)); \
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)                                                   \
        w8 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(weights_ptr + 8 * weights_stride_y)); \
        \
        VFMA(acc, w0, src0, src1, src2, src3, src4, src5, src6, src7);                       \
        VFMA(acc, w1, src1, src2, src3, src4, src5, src6, src7, src8);                       \
        VFMA(acc, w2, src2, src3, src4, src5, src6, src7, src8, src9);                       \
        VFMA(acc, w3, src3, src4, src5, src6, src7, src8, src9, src10);                      \
        VFMA(acc, w4, src4, src5, src6, src7, src8, src9, src10, src11);                     \
        VFMA(acc, w5, src5, src6, src7, src8, src9, src10, src11, src12);                    \
        VFMA(acc, w6, src6, src7, src8, src9, src10, src11, src12, src13);                   \
        VFMA(acc, w7, src7, src8, src9, src10, src11, src12, src13, src14);                  \
        VFMA(acc, w8, src8, src9, src10, src11, src12, src13, src14, src15);                 \
    })

#if VEC_SIZE == 4
#define REDUCE(out, vec)            \
    ({                              \
        VEC_DATA_TYPE(DATA_TYPE, 2) \
        tmp1 = vec.s01 + vec.s23;   \
        out  = tmp1.s0 + tmp1.s1;   \
    })
#else // VEC_SIZE == 4
#error("Not supported")
#endif // VEC_SIZE == 4

#if STRIDE_X == 1
#define CONVOLUTION1x9_NHWC(acc, row_ptr, weights_ptr) CONVOLUTION1x9_STRIDE1_NHWC_BIFROST(acc, row_ptr, weights_ptr)
#else // STRIDE_X == 1
#error "Not supported"
#endif // STRIDE_X == 1

#else // defined(VEC_SIZE)

#if STRIDE_X == 1
#define CONVOLUTION1x9_NHWC(acc, row_ptr, weights_ptr) CONVOLUTION1x9_STRIDE1_NHWC(acc, row_ptr, weights_ptr)
#elif STRIDE_X == 2 // STRIDE_X == 1
#define CONVOLUTION1x9_NHWC(acc, row_ptr, weights_ptr) CONVOLUTION1x9_STRIDE2_NHWC(acc, row_ptr, weights_ptr)
#else // STRIDE_X == 1
#error "STRIDE_X larger than 2 is not supported"
#endif // STRIDE_X == 1

#endif // defined(VEC_SIZE)

//#if defined(VEC_SIZE)
/** This kernel performs a direct convolution to convolve the low three dimensions in a tensor with the NHWC data layout
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
 * @param[in]  biases_ptr                            (Optional) Pointer to the biases tensor. Same as @p src_ptr
 * @param[in]  biases_stride_x                       (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases tensor
 * @param[in]  weights_stride_w                      (Optional) Stride of the weights tensor in the 4th dimension
 */
__kernel void direct_convolution9x9_nhwc(
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
    values = 0;

#if defined(VEC_SIZE)
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    values0 = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    values1 = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    values2 = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    values3 = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    values4 = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    values5 = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    values6 = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    values7 = 0;
#define STEP_X (VEC_SIZE)
#else // defined(VEC_SIZE)
#define STEP_X (1)
#endif // defined(VEC_SIZE)

    const int id0 = get_global_id(0);
    const int id1 = get_global_id(1);
    const int id2 = get_global_id(2);

    __global uchar *weights_addr = (__global uchar *)tensor3D_offset(&weights, 0, 0, 0);
    __global uchar *src_addr     = (__global uchar *)offset(&src, 0, 0) - src_stride_x * id0 + ((id2 * STRIDE_Y) - PAD_TOP) * (int)src_stride_z;

    weights_addr += id0 * weights_stride_w;

#if(PAD_TOP == 1)
    const int coordy = id2 - PAD_TOP;
    for(volatile int d = 0; d < WEIGHTS_DEPTH; d += STEP_X)
    {
        if(coordy < 0) // special case Z = -1 doesn't exists
        {
            //skip first row and load the two next ones
            CONVOLUTION1x9_NHWC(values, (src_addr + 1 * (int)src_stride_z), (weights_addr + 1 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 2 * (int)src_stride_z), (weights_addr + 2 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 3 * (int)src_stride_z), (weights_addr + 3 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 4 * (int)src_stride_z), (weights_addr + 4 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 5 * (int)src_stride_z), (weights_addr + 5 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 6 * (int)src_stride_z), (weights_addr + 6 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 7 * (int)src_stride_z), (weights_addr + 7 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 8 * (int)src_stride_z), (weights_addr + 8 * (int)weights_stride_z));
        }
        else if(coordy == (DST_HEIGHT - PAD_TOP - 1))
        {
            // special case when computing the last row of the output we must read the last three rows from the input buffer (including padding) but the
            // Z axis has no padding at all.
            CONVOLUTION1x9_NHWC(values, src_addr, weights_addr);
            CONVOLUTION1x9_NHWC(values, (src_addr + 1 * (int)src_stride_z), (weights_addr + 1 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 2 * (int)src_stride_z), (weights_addr + 2 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 3 * (int)src_stride_z), (weights_addr + 3 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 4 * (int)src_stride_z), (weights_addr + 4 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 5 * (int)src_stride_z), (weights_addr + 5 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 6 * (int)src_stride_z), (weights_addr + 6 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 7 * (int)src_stride_z), (weights_addr + 7 * (int)weights_stride_z));
        }
        else
        {
            CONVOLUTION1x9_NHWC(values, src_addr, weights_addr);
            CONVOLUTION1x9_NHWC(values, (src_addr + 1 * (int)src_stride_z), (weights_addr + 1 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 2 * (int)src_stride_z), (weights_addr + 2 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 3 * (int)src_stride_z), (weights_addr + 3 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 4 * (int)src_stride_z), (weights_addr + 4 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 5 * (int)src_stride_z), (weights_addr + 5 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 6 * (int)src_stride_z), (weights_addr + 6 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 7 * (int)src_stride_z), (weights_addr + 7 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 8 * (int)src_stride_z), (weights_addr + 8 * (int)weights_stride_z));
        }
        src_addr += STEP_X * sizeof(DATA_TYPE);
        weights_addr += STEP_X * sizeof(DATA_TYPE);
    }
#elif(PAD_TOP == 2) // PAD_TOP == 1
    const int coordy = id2 * STRIDE_Y;
    for(volatile int d = 0; d < WEIGHTS_DEPTH; d += STEP_X)
    {
        if(coordy == 0) // special case Z = -2 doesn't exists
        {
            //skip first row and load the two next ones
            CONVOLUTION1x9_NHWC(values, (src_addr + 2 * (int)src_stride_z), (weights_addr + 2 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 3 * (int)src_stride_z), (weights_addr + 3 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 4 * (int)src_stride_z), (weights_addr + 4 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 5 * (int)src_stride_z), (weights_addr + 5 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 6 * (int)src_stride_z), (weights_addr + 6 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 7 * (int)src_stride_z), (weights_addr + 7 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 8 * (int)src_stride_z), (weights_addr + 8 * (int)weights_stride_z));
        }
        else if(coordy == 1) // special case Z = -1 doesn't exists
        {
            //skip first row and load the two next ones
            CONVOLUTION1x9_NHWC(values, (src_addr + 1 * (int)src_stride_z), (weights_addr + 1 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 2 * (int)src_stride_z), (weights_addr + 2 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 3 * (int)src_stride_z), (weights_addr + 3 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 4 * (int)src_stride_z), (weights_addr + 4 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 5 * (int)src_stride_z), (weights_addr + 5 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 6 * (int)src_stride_z), (weights_addr + 6 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 7 * (int)src_stride_z), (weights_addr + 7 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 8 * (int)src_stride_z), (weights_addr + 8 * (int)weights_stride_z));
        }
        else if(coordy == (SRC_HEIGHT - 5))
        {
            // special case when computing the last row of the output we must read the last three rows from the input buffer (including padding) but the
            // Z axis has no padding at all.
            CONVOLUTION1x9_NHWC(values, src_addr, weights_addr);
            CONVOLUTION1x9_NHWC(values, (src_addr + 1 * (int)src_stride_z), (weights_addr + 1 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 2 * (int)src_stride_z), (weights_addr + 2 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 3 * (int)src_stride_z), (weights_addr + 3 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 4 * (int)src_stride_z), (weights_addr + 4 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 5 * (int)src_stride_z), (weights_addr + 5 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 6 * (int)src_stride_z), (weights_addr + 6 * (int)weights_stride_z));
        }
        else if(coordy == (SRC_HEIGHT - 6))
        {
            // special case when computing the last row of the output we must read the last three rows from the input buffer (including padding) but the
            // Z axis has no padding at all.
            CONVOLUTION1x9_NHWC(values, src_addr, weights_addr);
            CONVOLUTION1x9_NHWC(values, (src_addr + 1 * (int)src_stride_z), (weights_addr + 1 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 2 * (int)src_stride_z), (weights_addr + 2 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 3 * (int)src_stride_z), (weights_addr + 3 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 4 * (int)src_stride_z), (weights_addr + 4 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 5 * (int)src_stride_z), (weights_addr + 5 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 6 * (int)src_stride_z), (weights_addr + 6 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 7 * (int)src_stride_z), (weights_addr + 7 * (int)weights_stride_z));
        }
        else
        {
            CONVOLUTION1x9_NHWC(values, src_addr, weights_addr);
            CONVOLUTION1x9_NHWC(values, (src_addr + 1 * (int)src_stride_z), (weights_addr + 1 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 2 * (int)src_stride_z), (weights_addr + 2 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 3 * (int)src_stride_z), (weights_addr + 3 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 4 * (int)src_stride_z), (weights_addr + 4 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 5 * (int)src_stride_z), (weights_addr + 5 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 6 * (int)src_stride_z), (weights_addr + 6 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 7 * (int)src_stride_z), (weights_addr + 7 * (int)weights_stride_z));
            CONVOLUTION1x9_NHWC(values, (src_addr + 8 * (int)src_stride_z), (weights_addr + 8 * (int)weights_stride_z));
        }
        src_addr += STEP_X * sizeof(DATA_TYPE);
        weights_addr += STEP_X * sizeof(DATA_TYPE);
    }

#else  // PAD_TOP == 1
    for(volatile int d = 0; d < WEIGHTS_DEPTH; d += STEP_X)
    {
        CONVOLUTION1x9_NHWC(values, src_addr, weights_addr);
        CONVOLUTION1x9_NHWC(values, (src_addr + 1 * (int)src_stride_z), (weights_addr + 1 * (int)weights_stride_z));
        CONVOLUTION1x9_NHWC(values, (src_addr + 2 * (int)src_stride_z), (weights_addr + 2 * (int)weights_stride_z));
        CONVOLUTION1x9_NHWC(values, (src_addr + 3 * (int)src_stride_z), (weights_addr + 3 * (int)weights_stride_z));
        CONVOLUTION1x9_NHWC(values, (src_addr + 4 * (int)src_stride_z), (weights_addr + 4 * (int)weights_stride_z));
        CONVOLUTION1x9_NHWC(values, (src_addr + 5 * (int)src_stride_z), (weights_addr + 5 * (int)weights_stride_z));
        CONVOLUTION1x9_NHWC(values, (src_addr + 6 * (int)src_stride_z), (weights_addr + 6 * (int)weights_stride_z));
        CONVOLUTION1x9_NHWC(values, (src_addr + 7 * (int)src_stride_z), (weights_addr + 7 * (int)weights_stride_z));
        CONVOLUTION1x9_NHWC(values, (src_addr + 8 * (int)src_stride_z), (weights_addr + 8 * (int)weights_stride_z));
        src_addr += STEP_X * sizeof(DATA_TYPE);
        weights_addr += STEP_X * sizeof(DATA_TYPE);
    }
#endif // PAD_TOP == 1

#if defined(VEC_SIZE)
    REDUCE(values.s0, values0);
    REDUCE(values.s1, values1);
    REDUCE(values.s2, values2);
    REDUCE(values.s3, values3);
    REDUCE(values.s4, values4);
    REDUCE(values.s5, values5);
    REDUCE(values.s6, values6);
    REDUCE(values.s7, values7);
#endif // defined(VEC_SIZE)

#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
    values += (VEC_DATA_TYPE(DATA_TYPE, 8)) * ((__global DATA_TYPE *)(vector_offset(&biases, id0)));
#endif // defined(HAS_BIAS)

    *((__global DATA_TYPE *)(dst.ptr + 0 * dst_stride_y)) = values.s0;
    *((__global DATA_TYPE *)(dst.ptr + 1 * dst_stride_y)) = values.s1;
    *((__global DATA_TYPE *)(dst.ptr + 2 * dst_stride_y)) = values.s2;
    *((__global DATA_TYPE *)(dst.ptr + 3 * dst_stride_y)) = values.s3;
    *((__global DATA_TYPE *)(dst.ptr + 4 * dst_stride_y)) = values.s4;
    *((__global DATA_TYPE *)(dst.ptr + 5 * dst_stride_y)) = values.s5;
    *((__global DATA_TYPE *)(dst.ptr + 6 * dst_stride_y)) = values.s6;
    *((__global DATA_TYPE *)(dst.ptr + 7 * dst_stride_y)) = values.s7;
#undef STEP_X
}
#endif // defined(DATA_TYPE) && defined(STRIDE_X) && defined(WEIGHTS_DEPTH) && defined(DATA_LAYOUT_NHWC)
