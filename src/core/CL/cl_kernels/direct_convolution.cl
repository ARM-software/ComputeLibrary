/*
 * Copyright (c) 2021 Arm Limited.
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
#include "gemm_helpers.h"
#include "helpers_asymm.h"
#include "repeat.h"

#if defined(IS_QUANTIZED)

#if defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val) val = arm_dot_acc((x), (y), (val));
#elif defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8) // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#define ARM_DOT(x, y, val) val += arm_dot((x), (y));
#else // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val)                                \
    ({                                                    \
        val += (ACC_DATA_TYPE)x.s0 * (ACC_DATA_TYPE)y.s0; \
        val += (ACC_DATA_TYPE)x.s1 * (ACC_DATA_TYPE)y.s1; \
        val += (ACC_DATA_TYPE)x.s2 * (ACC_DATA_TYPE)y.s2; \
        val += (ACC_DATA_TYPE)x.s3 * (ACC_DATA_TYPE)y.s3; \
    })
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)

#define ARM_DOT1(a, b, c)                                                                                                                                                   \
    ({                                                                                                                                                                      \
        ARM_DOT(((VEC_DATA_TYPE(SRC_DATA_TYPE, 4))(a, (VEC_DATA_TYPE(SRC_DATA_TYPE, 3))0)), ((VEC_DATA_TYPE(WEI_DATA_TYPE, 4))(b, (VEC_DATA_TYPE(WEI_DATA_TYPE, 3))0)), c); \
    })
#define ARM_DOT2(a, b, c)                                                                                                                                                   \
    ({                                                                                                                                                                      \
        ARM_DOT(((VEC_DATA_TYPE(SRC_DATA_TYPE, 4))(a, (VEC_DATA_TYPE(SRC_DATA_TYPE, 2))0)), ((VEC_DATA_TYPE(WEI_DATA_TYPE, 4))(b, (VEC_DATA_TYPE(WEI_DATA_TYPE, 2))0)), c); \
    })
#define ARM_DOT3(a, b, c)                                                                                                               \
    ({                                                                                                                                  \
        ARM_DOT(((VEC_DATA_TYPE(SRC_DATA_TYPE, 4))(a, (SRC_DATA_TYPE)0)), ((VEC_DATA_TYPE(WEI_DATA_TYPE, 4))(b, (WEI_DATA_TYPE)0)), c); \
    })
#define ARM_DOT4(a, b, c) \
    ({                    \
        ARM_DOT(a, b, c); \
    })
#define ARM_DOT8(a, b, c)            \
    ({                               \
        ARM_DOT4((a.lo), (b.lo), c); \
        ARM_DOT4((a.hi), (b.hi), c); \
    })
#define ARM_DOT16(a, b, c)           \
    ({                               \
        ARM_DOT8((a.lo), (b.lo), c); \
        ARM_DOT8((a.hi), (b.hi), c); \
    })

#define ARM_OFFSET1(a, b, c)                      \
    ({                                            \
        c += (ACC_DATA_TYPE)a * (ACC_DATA_TYPE)b; \
    })
#define ARM_OFFSET2(a, b, c)                         \
    ({                                               \
        c += (ACC_DATA_TYPE)a.s0 * (ACC_DATA_TYPE)b; \
        c += (ACC_DATA_TYPE)a.s1 * (ACC_DATA_TYPE)b; \
    })
#define ARM_OFFSET3(a, b, c)                         \
    ({                                               \
        ARM_OFFSET2(a, b, c);                        \
        c += (ACC_DATA_TYPE)a.s2 * (ACC_DATA_TYPE)b; \
    })
#define ARM_OFFSET4(a, b, c)                         \
    ({                                               \
        ARM_OFFSET3(a, b, c);                        \
        c += (ACC_DATA_TYPE)a.s3 * (ACC_DATA_TYPE)b; \
    })
#define ARM_OFFSET8(a, b, c)         \
    ({                               \
        ARM_OFFSET4((a.lo), (b), c); \
        ARM_OFFSET4((a.hi), (b), c); \
    })
#define ARM_OFFSET16(a, b, c)        \
    ({                               \
        ARM_OFFSET8((a.lo), (b), c); \
        ARM_OFFSET8((a.hi), (b), c); \
    })

#if N0 == 1
#define ARM_OFFSET_K0XN0(k0, a, b, a_offset, b_offset, c) \
    ({                                                    \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c));                           \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##0), (a_offset), (c));                        \
    })
#elif N0 == 2 // N) == 3
#define ARM_OFFSET_K0XN0(k0, a, b, a_offset, b_offset, c) \
    ({                                                    \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s0));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##0), (a_offset), (c.s0));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s1));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##1), (a_offset), (c.s1));                     \
    })
#elif N0 == 3 // N0 == 3
#define ARM_OFFSET_K0XN0(k0, a, b, a_offset, b_offset, c) \
    ({                                                    \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s0));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##0), (a_offset), (c.s0));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s1));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##1), (a_offset), (c.s1));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s2));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##2), (a_offset), (c.s2));                     \
    })
#elif N0 == 4 // N0 == 4
#define ARM_OFFSET_K0XN0(k0, a, b, a_offset, b_offset, c) \
    ({                                                    \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s0));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##0), (a_offset), (c.s0));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s1));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##1), (a_offset), (c.s1));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s2));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##2), (a_offset), (c.s2));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s3));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##3), (a_offset), (c.s3));                     \
    })
#elif N0 == 8 // N0 == 8
#define ARM_OFFSET_K0XN0(k0, a, b, a_offset, b_offset, c) \
    ({                                                    \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s0));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##0), (a_offset), (c.s0));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s1));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##1), (a_offset), (c.s1));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s2));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##2), (a_offset), (c.s2));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s3));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##3), (a_offset), (c.s3));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s4));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##4), (a_offset), (c.s4));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s5));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##5), (a_offset), (c.s5));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s6));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##6), (a_offset), (c.s6));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s7));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##7), (a_offset), (c.s7));                     \
    })
#elif N0 == 16 // N0 == 16
#define ARM_OFFSET_K0XN0(k0, a, b, a_offset, b_offset, c) \
    ({                                                    \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s0));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##0), (a_offset), (c.s0));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s1));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##1), (a_offset), (c.s1));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s2));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##2), (a_offset), (c.s2));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s3));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##3), (a_offset), (c.s3));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s4));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##4), (a_offset), (c.s4));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s5));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##5), (a_offset), (c.s5));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s6));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##6), (a_offset), (c.s6));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s7));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##7), (a_offset), (c.s7));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s8));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##8), (a_offset), (c.s8));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.s9));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##9), (a_offset), (c.s9));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.sA));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##A), (a_offset), (c.sA));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.sB));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##B), (a_offset), (c.sB));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.sC));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##C), (a_offset), (c.sC));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.sD));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##D), (a_offset), (c.sD));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.sE));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##E), (a_offset), (c.sE));                     \
        CONCAT(ARM_OFFSET, k0)                            \
        ((a), (b_offset), (c.sF));                        \
        CONCAT(ARM_OFFSET, k0)                            \
        ((b##F), (a_offset), (c.sF));                     \
    })
#else // N0 not supported
#error "N0 value not supported"
#endif // N0 conditions
#else  // defined(IS_QUANTIZED)

#define ARM_DOT1(a, b, c)                         \
    ({                                            \
        c += (ACC_DATA_TYPE)a * (ACC_DATA_TYPE)b; \
    })
#define ARM_DOT2(a, b, c)                               \
    ({                                                  \
        c += (ACC_DATA_TYPE)a.s0 * (ACC_DATA_TYPE)b.s0; \
        c += (ACC_DATA_TYPE)a.s1 * (ACC_DATA_TYPE)b.s1; \
    })
#define ARM_DOT3(a, b, c)                               \
    ({                                                  \
        ARM_DOT2(a, b, c);                              \
        c += (ACC_DATA_TYPE)a.s2 * (ACC_DATA_TYPE)b.s2; \
    })
#define ARM_DOT4(a, b, c)                               \
    ({                                                  \
        ARM_DOT3(a, b, c);                              \
        c += (ACC_DATA_TYPE)a.s3 * (ACC_DATA_TYPE)b.s3; \
    })
#define ARM_DOT8(a, b, c)            \
    ({                               \
        ARM_DOT4((a.lo), (b.lo), c); \
        ARM_DOT4((a.hi), (b.hi), c); \
    })
#define ARM_DOT16(a, b, c)           \
    ({                               \
        ARM_DOT8((a.lo), (b.lo), c); \
        ARM_DOT8((a.hi), (b.hi), c); \
    })
#endif // defined(IS_QUANTIZED)

#if N0 == 1
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c));        \
    })
#elif N0 == 2 // N) == 3
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
    })
#elif N0 == 3 // N0 == 3
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
    })
#elif N0 == 4 // N0 == 4
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
    })
#elif N0 == 8 // N0 == 8
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##4), (c.s4));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##5), (c.s5));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##6), (c.s6));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##7), (c.s7));     \
    })
#elif N0 == 16 // N0 == 16
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##4), (c.s4));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##5), (c.s5));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##6), (c.s6));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##7), (c.s7));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##8), (c.s8));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##9), (c.s9));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##A), (c.sA));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##B), (c.sB));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##C), (c.sC));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##D), (c.sD));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##E), (c.sE));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##F), (c.sF));     \
    })
#else // N0 not supported
#error "N0 value not supported"
#endif // N0 conditions

/** OpenCL kernel to compute the direct convolution.
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16/QASYMM8/QASYMM8_SIGNED
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The accumulation data type must be passed at compile time using -DACC_DATA_TYPE (e.g. -DDATA_TYPE_PROMOTED=half)
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The convolution strides must be passed at compile time using -DSTRIDE and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the weights must be passed at compile time using -DWEI_WIDTH and -DWEI_HEIGHT (e.g. -DWEI_WIDTH=9, -DWEI_HEIGHT=9)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The spatial dimensions of the destination tensor must be passed at compile time using -DDST_WIDTH and -DDST_HEIGHT (e.g. -DDST_WIDTH=96, -DDST_HEIGHT=64)
 * @note The channels of the source tensor must be passed at compile time using -DSRC_CHANNELS (e.g. -DSRC_CHANNELS=64)
 * @note The channels of the destination tensor must be passed at compile time using -DDST_CHANNELS (e.g. -DDDST_CHANNELS=64)
 * @note The data type of the source tensor must be passed at compile time using -DSRC_DATA_TYPE (e.g. -DSRC_DATA_TYPE=float)
 * @note The data type of the weights tensor must be passed at compile time using -DWEI_DATA_TYPE (e.g. -DWEI_DATA_TYPE=float)
 * @note The data type of the destination tensor must be passed at compile time using -DDST_DATA_TYPE (e.g. -DDST_DATA_TYPE=float)
 * @note The data type of the accumulators must be passed at compile time using -DACC_DATA_TYPE (e.g. -DACC_DATA_TYPE=float)
 * @note The number of M0 rows (width*height) to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of N0 output channels to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The number of K0 inner accumulations must be passed at compile time using -DK0 (e.g. -DK0=2)
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *
 *@note In case of QASYMM8/QASYMM8_SIGNED, the following extra information must be passed at compile time:
 * - -DIS_QUANTIZED
 * - The destination quantization multiplier e.g. -DDST_MULTIPLIER=1234
 * - The destination quantization shift e.g. -DDST_SHIFT=4
 * - The destination offset e.g. -DDST_OFFSET=4
 * - The source offset e.g. -DSRC_OFFSET=4
 * - The weights offset e.g. -DWEI_OFFSET=4
 * - The quantized zero value e.g. -DZERO_VALUE=4
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data type: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  wei_ptr                           Pointer to the weights tensor. Supported data type: same as @p src_ptr
 * @param[in]  wei_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  wei_step_x                        wei_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  wei_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  wei_step_y                        wei_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  wei_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  wei_step_z                        wei_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  wei_offset_first_element_in_bytes The offset of the first element in the bias matrix
 * @param[in]  bia_ptr                           (Optional) Pointer to the bias tensor Supported data type: same as @p src_ptr (if F32/F16) or S32 (if QASYMM8/QASYMM8_SIGNED)
 * @param[in]  bia_stride_x                      (Optional) Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  bia_step_x                        (Optional) bia_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bia_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[in]  wei_stride_w                      Stride of the weights tensor in W dimension (in bytes)
 */
__kernel void direct_convolution_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(wei),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bia),
#endif // defined(HAS_BIAS)
    unsigned int wei_stride_w)
{
#if M0 != 1
#error "M0: Only supported 1"
#endif // M0 != 1

    const int cout = max((int)(get_global_id(0) * N0 - (N0 - PARTIAL_STORE_N0) % N0), 0); // input channels
    const int mout = get_global_id(1);                                                    // width x height
    const int zout = get_global_id(2);                                                    // batch size index

    REPEAT_VAR_INIT_TO_CONST(16, int, zero, 0);
    REPEAT_VAR_INIT_TO_CONST(M0, int, xi, 0);
    REPEAT_VAR_INIT_TO_CONST(M0, int, yi, 0);

#define LINEAR_2_COORDS(i)                            \
    xi##i = ((mout * M0 + i) % DST_WIDTH) * STRIDE_X; \
    yi##i = ((mout * M0 + i) / DST_WIDTH) * STRIDE_Y; \
    xi##i -= PAD_LEFT;                                \
    yi##i -= PAD_TOP;

    // Convert the linear index to coordinate
    LINEAR_2_COORDS(0);

#undef LINEAR_2_COORDS

    uint src_offset = src_offset_first_element_in_bytes + zout * src_stride_y * (SRC_WIDTH * SRC_HEIGHT);
    uint wei_offset = wei_offset_first_element_in_bytes + cout * wei_stride_w;

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(ACC_DATA_TYPE, N0), c, 0);

    for(int i = 0; i < (WEI_WIDTH * WEI_HEIGHT); ++i)
    {
        int xk = i % WEI_WIDTH;
        int yk = i / WEI_WIDTH;

        REPEAT_VAR_INIT_TO_CONST(M0, int, mi_valid_row, 0);
        REPEAT_VAR_INIT_TO_CONST(M0, int, mi_mask, 0);

        // Calculate the input row to read from source tensor
#define MI_INIT(i)                                                                                                  \
    mi_valid_row##i = max(min(xi##i + xk, SRC_WIDTH - 1), 0) + max(min(yi##i + yk, SRC_HEIGHT - 1), 0) * SRC_WIDTH; \
    mi_mask##i      = (xi##i + xk) >= 0 && (xi##i + xk) < SRC_WIDTH && (yi##i + yk) >= 0 && (yi##i + yk) < SRC_HEIGHT;

        MI_INIT(0);

#undef MI_INIT

        int k = 0;
        for(; k <= (SRC_CHANNELS - K0); k += K0)
        {
            // Load values from src tensor
            LOAD_BLOCK_INDIRECT(M0, K0, SRC_DATA_TYPE, a, src_ptr, src_offset + k * sizeof(SRC_DATA_TYPE), src_stride_y, mi_valid_row, mi_mask);

            // Load values from weights tensor
            LOAD_BLOCK(N0, K0, WEI_DATA_TYPE, b, wei_ptr, wei_offset, wei_stride_w, zero);

#if defined(IS_QUANTIZED)
#define TENSOR_DOT(K0, i)                                                                                      \
    if(mi_mask##i != 0)                                                                                        \
    {                                                                                                          \
        ARM_DOT_K0XN0(K0, a##i, b, c##i);                                                                      \
        ARM_OFFSET_K0XN0(K0, a##i, b, SRC_OFFSET, WEI_OFFSET, c##i);                                           \
    }                                                                                                          \
    else                                                                                                       \
    {                                                                                                          \
        ARM_DOT_K0XN0(K0, ((VEC_DATA_TYPE(SRC_DATA_TYPE, K0))ZERO_VALUE), b, c##i);                            \
        ARM_OFFSET_K0XN0(K0, ((VEC_DATA_TYPE(SRC_DATA_TYPE, K0))ZERO_VALUE), b, SRC_OFFSET, WEI_OFFSET, c##i); \
    }
#else // defined(IS_QUANTIZED)
#define TENSOR_DOT(K0, i) \
    ARM_DOT_K0XN0(K0, a##i, b, c##i);
#endif // defined(IS_QUANTIZED)

            TENSOR_DOT(K0, 0);

            wei_offset += K0 * sizeof(WEI_DATA_TYPE);
        }

#if(SRC_CHANNELS % K0) != 0
        // Left-over accumulations
        for(; k < SRC_CHANNELS; ++k)
        {
            // Load values from src tensor
            LOAD_BLOCK_INDIRECT(M0, 1, SRC_DATA_TYPE, a, src_ptr, src_offset + k * sizeof(SRC_DATA_TYPE), src_stride_y, mi_valid_row, mi_mask);

            // Load values from weights tensor
            LOAD_BLOCK(N0, 1, WEI_DATA_TYPE, b, wei_ptr, wei_offset, wei_stride_w, zero);

            TENSOR_DOT(1, 0);

#undef TENSOR_DOT

            wei_offset += sizeof(WEI_DATA_TYPE);
        }
#endif // (SRC_CHANNELS % K0) != 0

        c0 += (SRC_CHANNELS * SRC_OFFSET * WEI_OFFSET);
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (cout * sizeof(DST_DATA_TYPE)) + (mout * M0 * dst_stride_y);

    // Batched direct convolution
    dst_addr += zout * dst_stride_y * (DST_WIDTH * DST_HEIGHT);

#if defined(HAS_BIAS)
    __global uchar *bias_addr = bia_ptr + bia_offset_first_element_in_bytes + (cout * sizeof(BIA_DATA_TYPE));

    LOAD_BLOCK(1, N0, BIA_DATA_TYPE, bias, bias_addr, 0, zero0, zero);

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);
#endif // HAS_BIAS

#if defined(IS_QUANTIZED)

    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DST_DATA_TYPE, N0), cq, 0);

#if DST_SHIFT < 0
#define QUANTIZE(i)                                                                               \
    c##i  = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(c##i, DST_MULTIPLIER, DST_SHIFT, N0); \
    c##i  = c##i + DST_OFFSET;                                                                    \
    cq##i = CONVERT_SAT(c##i, VEC_DATA_TYPE(DST_DATA_TYPE, N0));
#else // OUTPUT_SHIFT < 0
#define QUANTIZE(i)                                                                            \
    c##i  = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(c##i, DST_MULTIPLIER, DST_SHIFT, N0); \
    c##i  = c##i + DST_OFFSET;                                                                 \
    cq##i = CONVERT_SAT(c##i, VEC_DATA_TYPE(DST_DATA_TYPE, N0));
#endif // OUTPUT_SHIFT < 0

    QUANTIZE(0);

#undef QUANTIZE

    STORE_VECTOR_SELECT(cq, DST_DATA_TYPE, dst_addr, N0, PARTIAL_STORE_N0, PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0);
#else  // defined(IS_QUANTIZED)
    STORE_VECTOR_SELECT(c, DST_DATA_TYPE, dst_addr, N0, PARTIAL_STORE_N0, PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0);
#endif // defined(IS_QUANTIZED)
}