/*
 * Copyright (c) 2019-2021 Arm Limited.
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

#if defined(DATA_TYPE)
/** Calculates and applies the twiddle factor to a given input.
 *
 * @param[in]     phi   The angle.
 * @param[in,out] input The input on which the factor should be applied.
 */
#define TWIDDLE_FACTOR_MULTIPLICATION(phi, input)  \
    {                                              \
        VEC_DATA_TYPE(DATA_TYPE, 2)                \
        w, tmp;                                    \
        w.x   = cos(phi);                          \
        w.y   = sin(phi);                          \
        tmp.x = (w.x * input.x) - (w.y * input.y); \
        tmp.y = (w.x * input.y) + (w.y * input.x); \
        input = tmp;                               \
    }

/** Computes radix-2 butterfly unit.
 *
 * @param[in,out] c0 Complex input 0.
 * @param[in,out] c1 Complex input 1.
 */
#define DFT_2(c0, c1)               \
    {                               \
        VEC_DATA_TYPE(DATA_TYPE, 2) \
        v0;                         \
        v0 = c0;                    \
        c0 = v0 + c1;               \
        c1 = v0 - c1;               \
    }

// radix-3 butterfly unit factors
#define SQRT3DIV2 0.86602540378443f

/** Computes radix-3 butterfly unit.
 *
 * @param[in,out] c0 Complex input 0.
 * @param[in,out] c1 Complex input 1.
 * @param[in,out] c2 Complex input 2.
 */
#define DFT_3(c0, c1, c2)                             \
    {                                                 \
        VEC_DATA_TYPE(DATA_TYPE, 2)                   \
        v0 = c1 + c2;                                 \
        VEC_DATA_TYPE(DATA_TYPE, 2)                   \
        v1   = c1 - c2;                               \
        c1.x = c0.x - 0.5f * v0.x + v1.y * SQRT3DIV2; \
        c1.y = c0.y - 0.5f * v0.y - v1.x * SQRT3DIV2; \
        c2.x = c0.x - 0.5f * v0.x - v1.y * SQRT3DIV2; \
        c2.y = c0.y - 0.5f * v0.y + v1.x * SQRT3DIV2; \
        c0   = c0 + v0;                               \
    }

/**Computes radix-4 butterfly unit.
 *
 * @param[in,out] c0 Complex input 0.
 * @param[in,out] c1 Complex input 1.
 * @param[in,out] c2 Complex input 2.
 * @param[in,out] c3 Complex input 3.
 */
#define DFT_4(c0, c1, c2, c3)       \
    {                               \
        VEC_DATA_TYPE(DATA_TYPE, 2) \
        v0, v1, v2, v3;             \
        v0   = c0 + c2;             \
        v1   = c1 + c3;             \
        v2   = c0 - c2;             \
        v3.x = c1.y - c3.y;         \
        v3.y = c3.x - c1.x;         \
        c0   = v0 + v1;             \
        c2   = v0 - v1;             \
        c1   = v2 + v3;             \
        c3   = v2 - v3;             \
    }

// radix-5 butterfly unit factors
#define W5_A (DATA_TYPE)0.30901699437494f
#define W5_B (DATA_TYPE)0.95105651629515f
#define W5_C (DATA_TYPE)0.80901699437494f
#define W5_D (DATA_TYPE)0.58778525229247f

/** Computes radix-5 butterfly unit.
 *
 * @param[in,out] c0 Complex input 0.
 * @param[in,out] c1 Complex input 1.
 * @param[in,out] c2 Complex input 2.
 * @param[in,out] c3 Complex input 3.
 * @param[in,out] c4 Complex input 4.
 */
#define DFT_5(c0, c1, c2, c3, c4)                                  \
    {                                                              \
        VEC_DATA_TYPE(DATA_TYPE, 2)                                \
        v0, v1, v2, v3, v4;                                        \
        v0 = c0;                                                   \
        v1 = W5_A * (c1 + c4) - W5_C * (c2 + c3);                  \
        v2 = W5_C * (c1 + c4) - W5_A * (c2 + c3);                  \
        v3 = W5_D * (c1 - c4) - W5_B * (c2 - c3);                  \
        v4 = W5_B * (c1 - c4) + W5_D * (c2 - c3);                  \
        c0 = v0 + c1 + c2 + c3 + c4;                               \
        c1 = v0 + v1 + (VEC_DATA_TYPE(DATA_TYPE, 2))(v4.y, -v4.x); \
        c2 = v0 - v2 + (VEC_DATA_TYPE(DATA_TYPE, 2))(v3.y, -v3.x); \
        c3 = v0 - v2 + (VEC_DATA_TYPE(DATA_TYPE, 2))(-v3.y, v3.x); \
        c4 = v0 + v1 + (VEC_DATA_TYPE(DATA_TYPE, 2))(-v4.y, v4.x); \
    }

// radix-7 butterfly unit factors
#define W7_A (DATA_TYPE)0.62348980185873f
#define W7_B (DATA_TYPE)0.78183148246802f
#define W7_C (DATA_TYPE)0.22252093395631f
#define W7_D (DATA_TYPE)0.97492791218182f
#define W7_E (DATA_TYPE)0.90096886790241f
#define W7_F (DATA_TYPE)0.43388373911755f

/** Computes radix-7 butterfly unit.
 *
 * @param[in,out] c0 Complex input 0.
 * @param[in,out] c1 Complex input 1.
 * @param[in,out] c2 Complex input 2.
 * @param[in,out] c3 Complex input 3.
 * @param[in,out] c4 Complex input 4.
 * @param[in,out] c5 Complex input 5.
 * @param[in,out] c6 Complex input 6.
 */
#define DFT_7(c0, c1, c2, c3, c4, c5, c6)                            \
    {                                                                \
        VEC_DATA_TYPE(DATA_TYPE, 2)                                  \
        v0, v1, v2, v3, v4, v5, v6;                                  \
        v0 = c0;                                                     \
        v1 = W7_A * (c1 + c6) - W7_C * (c2 + c5) - W7_E * (c3 + c4); \
        v2 = W7_C * (c1 + c6) + W7_E * (c2 + c5) - W7_A * (c3 + c4); \
        v3 = W7_E * (c1 + c6) - W7_A * (c2 + c5) + W7_C * (c3 + c4); \
        v4 = W7_B * (c1 - c6) + W7_D * (c2 - c5) + W7_F * (c3 - c4); \
        v5 = W7_D * (c1 - c6) - W7_F * (c2 - c5) - W7_B * (c3 - c4); \
        v6 = W7_F * (c1 - c6) - W7_B * (c2 - c5) + W7_D * (c3 - c4); \
        c0 = v0 + c1 + c2 + c3 + c4 + c5 + c6;                       \
        c1 = v0 + v1 + (VEC_DATA_TYPE(DATA_TYPE, 2))(v4.y, -v4.x);   \
        c2 = v0 - v2 + (VEC_DATA_TYPE(DATA_TYPE, 2))(v5.y, -v5.x);   \
        c3 = v0 - v3 + (VEC_DATA_TYPE(DATA_TYPE, 2))(v6.y, -v6.x);   \
        c4 = v0 - v3 + (VEC_DATA_TYPE(DATA_TYPE, 2))(-v6.y, v6.x);   \
        c5 = v0 - v2 + (VEC_DATA_TYPE(DATA_TYPE, 2))(-v5.y, v5.x);   \
        c6 = v0 + v1 + (VEC_DATA_TYPE(DATA_TYPE, 2))(-v4.y, v4.x);   \
    }

/** Computes radix-8 butterfly unit.
 *
 * @param[in,out] c0 Complex input 0.
 * @param[in,out] c1 Complex input 1.
 * @param[in,out] c2 Complex input 2.
 * @param[in,out] c3 Complex input 3.
 * @param[in,out] c4 Complex input 4.
 * @param[in,out] c5 Complex input 5.
 * @param[in,out] c6 Complex input 6.
 * @param[in,out] c7 Complex input 7.
 */
#define DFT_8(c0, c1, c2, c3, c4, c5, c6, c7) \
    {                                         \
        VEC_DATA_TYPE(DATA_TYPE, 2)           \
        v0, v1, v2, v3, v4, v5, v6, v7;       \
        VEC_DATA_TYPE(DATA_TYPE, 2)           \
        s0, s1, s2, s3, s4, s5, s6, s7;       \
        VEC_DATA_TYPE(DATA_TYPE, 2)           \
        t0, t1, t2;                           \
        v0   = c0 + c4;                       \
        v1   = c1 + c5;                       \
        v2   = c2 + c6;                       \
        v3   = c3 + c7;                       \
        v4   = c0 - c4;                       \
        v5   = c1 - c5;                       \
        v6   = c2 - c6;                       \
        v7   = c3 - c7;                       \
        s0   = v0 + v2;                       \
        s1   = v1 + v3;                       \
        s2   = v0 - v2;                       \
        s3   = v1 - v3;                       \
        s4.x = v4.x - v6.y;                   \
        s4.y = v4.y + v6.x;                   \
        s5.x = v5.x - v7.y;                   \
        s5.y = v5.y + v7.x;                   \
        s6.x = v4.x + v6.y;                   \
        s6.y = v4.y - v6.x;                   \
        s7.x = v5.x + v7.y;                   \
        s7.y = v5.y - v7.x;                   \
        t0.x = -s3.y;                         \
        t0.y = s3.x;                          \
        t1.x = M_SQRT1_2_F * (s5.x - s5.y);   \
        t1.y = M_SQRT1_2_F * (s5.x + s5.y);   \
        t2.x = -M_SQRT1_2_F * (s7.x + s7.y);  \
        t2.y = M_SQRT1_2_F * (s7.x - s7.y);   \
        c0   = s0 + s1;                       \
        c1   = s6 - t2;                       \
        c2   = s2 - t0;                       \
        c3   = s4 - t1;                       \
        c4   = s0 - s1;                       \
        c5   = s6 + t2;                       \
        c6   = s2 + t0;                       \
        c7   = s4 + t1;                       \
    }

/** Computes the first stage of a radix-2 DFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_2_first_stage_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load two complex input values
    VEC_DATA_TYPE(DATA_TYPE, 4)
    data = vload4(0, (__global DATA_TYPE *)input.ptr);

    // Compute DFT N = 2
    DFT_2(data.s01, data.s23);

    // Store two complex output values
    vstore4(data, 0, (__global DATA_TYPE *)output.ptr);
}

/** Computes the first stage of a radix-2 DFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_2_first_stage_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load two complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));

    // Compute DFT N = 2
    DFT_2(data1, data2);

    // Store two complex output values
    vstore2(data1, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(data2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 1, 0));
}

/** Computes the first stage of a radix-3 DFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_3_first_stage_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load three complex input values
    VEC_DATA_TYPE(DATA_TYPE, 4)
    data0 = vload4(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 2, 0, 0));

    // Compute DFT N = 3
    DFT_3(data0.s01, data0.s23, data1.s01);

    // Store three complex output values
    vstore4(data0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(data1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 2, 0, 0));
}

/** Computes the first stage of a radix-3 DFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_3_first_stage_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load three complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));

    // Compute DFT N = 3
    DFT_3(data0, data1, data2);

    // Store three complex output values
    vstore2(data0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(data1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 1, 0));
    vstore2(data2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2, 0));
}

/** Computes the first stage of a radix-4 DFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_4_first_stage_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load four complex input values
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data = vload8(0, (__global DATA_TYPE *)input.ptr);

    // Compute DFT N = 4
    DFT_4(data.s01, data.s23, data.s45, data.s67);

    // Store four complex output values
    vstore8(data, 0, (__global DATA_TYPE *)output.ptr);
}

/** Computes the first stage of a radix-4 DFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_4_first_stage_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load four complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 3, 0));

    // Compute DFT N = 4
    DFT_4(data0, data1, data2, data3);

    // Store four complex output values
    vstore2(data0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(data1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 1, 0));
    vstore2(data2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2, 0));
    vstore2(data3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 3, 0));
}

/** Computes the first stage of a radix-5 DFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_5_first_stage_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load five complex input values
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data0 = vload8(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 4, 0, 0));

    // Compute DFT N = 5
    DFT_5(data0.s01, data0.s23, data0.s45, data0.s67, data1.s01);

    // Store five complex output values
    vstore8(data0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(data1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 4, 0, 0));
}

/** Computes the first stage of a radix-5 DFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_5_first_stage_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load five complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 3, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data4 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 4, 0));

    // Compute DFT N = 5
    DFT_5(data0, data1, data2, data3, data4);

    // Store five complex output values
    vstore2(data0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(data1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 1, 0));
    vstore2(data2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2, 0));
    vstore2(data3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 3, 0));
    vstore2(data4, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 4, 0));
}

/** Computes the first stage of a radix-7 DFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_7_first_stage_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load seven complex input values
    VEC_DATA_TYPE(DATA_TYPE, 8)
    data0 = vload8(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 4)
    data1 = vload4(0, (__global DATA_TYPE *)tensor3D_offset(&input, 4, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 6, 0, 0));

    // Compute DFT N = 7
    DFT_7(data0.s01, data0.s23, data0.s45, data0.s67, data1.s01, data1.s23, data2.s01);

    // Store seven complex output values
    vstore8(data0, 0, (__global DATA_TYPE *)output.ptr);
    vstore4(data1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 4, 0, 0));
    vstore2(data2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 6, 0, 0));
}

/** Computes the first stage of a radix-7 DFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_7_first_stage_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load seven complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 3, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data4 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 4, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data5 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 5, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data6 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 6, 0));

    // Compute DFT N = 7
    DFT_7(data0, data1, data2, data3, data4, data5, data6);

    // Store seven complex output values
    vstore2(data0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(data1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 1, 0));
    vstore2(data2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2, 0));
    vstore2(data3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 3, 0));
    vstore2(data4, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 4, 0));
    vstore2(data5, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 5, 0));
    vstore2(data6, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 6, 0));
}

/** Computes the first stage of a radix-8 DFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_8_first_stage_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load eight complex input values
    VEC_DATA_TYPE(DATA_TYPE, 16)
    data = vload16(0, (__global DATA_TYPE *)input.ptr);

    // Compute DFT N = 8
    DFT_8(data.s01, data.s23, data.s45, data.s67, data.s89, data.sAB, data.sCD, data.sEF);

    // Store eight complex output values
    vstore16(data, 0, (__global DATA_TYPE *)output.ptr);
}

/** Computes the first stage of a radix-8 DFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void fft_radix_8_first_stage_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load eight complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 1, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 3, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data4 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 4, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data5 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 5, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data6 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 6, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    data7 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 7, 0));

    // Compute DFT N = 8
    DFT_8(data0, data1, data2, data3, data4, data5, data6, data7);

    // Store eight complex output values
    vstore2(data0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(data1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 1, 0));
    vstore2(data2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2, 0));
    vstore2(data3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 3, 0));
    vstore2(data4, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 4, 0));
    vstore2(data5, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 5, 0));
    vstore2(data6, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 6, 0));
    vstore2(data7, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 7, 0));
}

/** Computes a stage of a radix-2 FFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_2_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-2
    uint kx = get_global_id(0);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += n * input.stride_x + get_global_id(1) * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += n * output.stride_x + get_global_id(1) * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load two complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, Nx, 0, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);

    // Compute DFT N = 2
    DFT_2(c0, c1);

    // Store two complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, Nx, 0, 0));
}

/** Computes a stage of a radix-2 FFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_2_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-2
    uint kx = get_global_id(1);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += get_global_id(0) * input.stride_x + n * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += get_global_id(0) * output.stride_x + n * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load two complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, Nx, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);

    // Compute DFT N = 2
    DFT_2(c0, c1);

    // Store two complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, Nx, 0));
}

/** Computes a stage of a radix-3 FFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_3_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-3
    uint kx = get_global_id(0);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += n * input.stride_x + get_global_id(1) * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += n * output.stride_x + get_global_id(1) * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load three complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 2 * Nx, 0, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);

    // Compute DFT N = 3
    DFT_3(c0, c1, c2);

    // Store three complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, Nx, 0, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 2 * Nx, 0, 0));
}

/** Computes a stage of a radix-3 FFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_3_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-3
    uint kx = get_global_id(1);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += get_global_id(0) * input.stride_x + n * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += get_global_id(0) * output.stride_x + n * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load three complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2 * Nx, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);

    // Compute DFT N = 3
    DFT_3(c0, c1, c2);

    // Store three complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, Nx, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2 * Nx, 0));
}

/** Computes a stage of a radix-4 FFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_4_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-4
    uint kx = get_global_id(0);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += n * input.stride_x + get_global_id(1) * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += n * output.stride_x + get_global_id(1) * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load four complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 2 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 3 * Nx, 0, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);
    TWIDDLE_FACTOR_MULTIPLICATION(3 * phi, c3);

    // Compute DFT N = 4
    DFT_4(c0, c1, c2, c3);

    // Store four complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, Nx, 0, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 2 * Nx, 0, 0));
    vstore2(c3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 3 * Nx, 0, 0));
}

/** Computes a stage of a radix-4 FFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_4_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-4
    uint kx = get_global_id(1);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += get_global_id(0) * input.stride_x + n * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += get_global_id(0) * output.stride_x + n * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load four complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 3 * Nx, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);
    TWIDDLE_FACTOR_MULTIPLICATION(3 * phi, c3);

    // Compute DFT N = 4
    DFT_4(c0, c1, c2, c3);

    // Store four complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, Nx, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2 * Nx, 0));
    vstore2(c3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 3 * Nx, 0));
}

/** Computes a stage of a radix-5 FFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_5_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-5
    uint kx = get_global_id(0);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += n * input.stride_x + get_global_id(1) * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += n * output.stride_x + get_global_id(1) * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load five complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 2 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 3 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c4 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 4 * Nx, 0, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);
    TWIDDLE_FACTOR_MULTIPLICATION(3 * phi, c3);
    TWIDDLE_FACTOR_MULTIPLICATION(4 * phi, c4);

    // Compute DFT N = 5
    DFT_5(c0, c1, c2, c3, c4);

    // Store five complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, Nx, 0, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 2 * Nx, 0, 0));
    vstore2(c3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 3 * Nx, 0, 0));
    vstore2(c4, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 4 * Nx, 0, 0));
}

/** Computes a stage of a radix-5 FFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_5_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-5
    uint kx = get_global_id(1);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += get_global_id(0) * input.stride_x + n * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += get_global_id(0) * output.stride_x + n * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load five complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 3 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c4 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 4 * Nx, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);
    TWIDDLE_FACTOR_MULTIPLICATION(3 * phi, c3);
    TWIDDLE_FACTOR_MULTIPLICATION(4 * phi, c4);

    // Compute DFT N = 5
    DFT_5(c0, c1, c2, c3, c4);

    // Store five complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, Nx, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2 * Nx, 0));
    vstore2(c3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 3 * Nx, 0));
    vstore2(c4, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 4 * Nx, 0));
}

/** Computes a stage of a radix-7 FFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_7_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-7
    uint kx = get_global_id(0);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += n * input.stride_x + get_global_id(1) * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += n * output.stride_x + get_global_id(1) * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load seven complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 2 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 3 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c4 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 4 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c5 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 5 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c6 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 6 * Nx, 0, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);
    TWIDDLE_FACTOR_MULTIPLICATION(3 * phi, c3);
    TWIDDLE_FACTOR_MULTIPLICATION(4 * phi, c4);
    TWIDDLE_FACTOR_MULTIPLICATION(5 * phi, c5);
    TWIDDLE_FACTOR_MULTIPLICATION(6 * phi, c6);

    // Compute DFT N = 7
    DFT_7(c0, c1, c2, c3, c4, c5, c6);

    // Store seven complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, Nx, 0, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 2 * Nx, 0, 0));
    vstore2(c3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 3 * Nx, 0, 0));
    vstore2(c4, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 4 * Nx, 0, 0));
    vstore2(c5, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 5 * Nx, 0, 0));
    vstore2(c6, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 6 * Nx, 0, 0));
}

/** Computes a stage of a radix-7 FFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_7_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-7
    uint kx = get_global_id(1);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += get_global_id(0) * input.stride_x + n * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += get_global_id(0) * output.stride_x + n * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load seven complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 3 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c4 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 4 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c5 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 5 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c6 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 6 * Nx, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);
    TWIDDLE_FACTOR_MULTIPLICATION(3 * phi, c3);
    TWIDDLE_FACTOR_MULTIPLICATION(4 * phi, c4);
    TWIDDLE_FACTOR_MULTIPLICATION(5 * phi, c5);
    TWIDDLE_FACTOR_MULTIPLICATION(6 * phi, c6);

    // Compute DFT N = 7
    DFT_7(c0, c1, c2, c3, c4, c5, c6);

    // Store seven complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, Nx, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2 * Nx, 0));
    vstore2(c3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 3 * Nx, 0));
    vstore2(c4, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 4 * Nx, 0));
    vstore2(c5, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 5 * Nx, 0));
    vstore2(c6, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 6 * Nx, 0));
}

/** Computes a stage of a radix-8 FFT on axis 0.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_8_axis_0(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-8
    uint kx = get_global_id(0);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += n * input.stride_x + get_global_id(1) * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += n * output.stride_x + get_global_id(1) * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load eight complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 2 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 3 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c4 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 4 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c5 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 5 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c6 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 6 * Nx, 0, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c7 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 7 * Nx, 0, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);
    TWIDDLE_FACTOR_MULTIPLICATION(3 * phi, c3);
    TWIDDLE_FACTOR_MULTIPLICATION(4 * phi, c4);
    TWIDDLE_FACTOR_MULTIPLICATION(5 * phi, c5);
    TWIDDLE_FACTOR_MULTIPLICATION(6 * phi, c6);
    TWIDDLE_FACTOR_MULTIPLICATION(7 * phi, c7);

    // Compute DFT N = 8
    DFT_8(c0, c1, c2, c3, c4, c5, c6, c7);

    // Store eight complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, Nx, 0, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 2 * Nx, 0, 0));
    vstore2(c3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 3 * Nx, 0, 0));
    vstore2(c4, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 4 * Nx, 0, 0));
    vstore2(c5, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 5 * Nx, 0, 0));
    vstore2(c6, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 6 * Nx, 0, 0));
    vstore2(c7, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 7 * Nx, 0, 0));
}

/** Computes a stage of a radix-8 FFT on axis 1.
 *
 * @note In order to perform the FFT function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @param[in,out] input_ptr                            Pointer to the source tensor. Supported data types: F16/f32
 * @param[in,out] input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in,out] input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in,out] input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in,out] input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in,out] input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in,out] input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in,out] input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out]    output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]     output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]     output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]     output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]     output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 * @param[in]     Nx                                   The butterfly span. Products of radix order of previous radix's stage
 * @param[in]     Ni                                   Nx * Ny.
 * @param[in]     exp_const                            Exponent constant
 */
__kernel void fft_radix_8_axis_1(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
    ,
    uint Nx, uint Ni, float exp_const)
{
    // Each work-item computes a single radix-8
    uint kx = get_global_id(1);

    // Compute nx
    uint nx = kx % Nx;

    // Compute n index
    uint n = nx + (kx / Nx) * Ni;

    // Get tensor pointers
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    input.ptr += get_global_id(0) * input.stride_x + n * input.stride_y + get_global_id(2) * input.stride_z;
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);
    output.ptr += get_global_id(0) * output.stride_x + n * output.stride_y + get_global_id(2) * output.stride_z;
#endif /* IN_PLACE */

    // Load eight complex input values
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c0 = vload2(0, (__global DATA_TYPE *)input.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c1 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c2 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 2 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c3 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 3 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c4 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 4 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c5 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 5 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c6 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 6 * Nx, 0));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    c7 = vload2(0, (__global DATA_TYPE *)tensor3D_offset(&input, 0, 7 * Nx, 0));

    // Compute phi
    DATA_TYPE phi = (DATA_TYPE)nx * (DATA_TYPE)exp_const;

    // Multiply by twiddle factor
    TWIDDLE_FACTOR_MULTIPLICATION(phi, c1);
    TWIDDLE_FACTOR_MULTIPLICATION(2 * phi, c2);
    TWIDDLE_FACTOR_MULTIPLICATION(3 * phi, c3);
    TWIDDLE_FACTOR_MULTIPLICATION(4 * phi, c4);
    TWIDDLE_FACTOR_MULTIPLICATION(5 * phi, c5);
    TWIDDLE_FACTOR_MULTIPLICATION(6 * phi, c6);
    TWIDDLE_FACTOR_MULTIPLICATION(7 * phi, c7);

    // Compute DFT N = 8
    DFT_8(c0, c1, c2, c3, c4, c5, c6, c7);

    // Store eight complex output values
    vstore2(c0, 0, (__global DATA_TYPE *)output.ptr);
    vstore2(c1, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, Nx, 0));
    vstore2(c2, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 2 * Nx, 0));
    vstore2(c3, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 3 * Nx, 0));
    vstore2(c4, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 4 * Nx, 0));
    vstore2(c5, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 5 * Nx, 0));
    vstore2(c6, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 6 * Nx, 0));
    vstore2(c7, 0, (__global DATA_TYPE *)tensor3D_offset(&output, 0, 7 * Nx, 0));
}
#endif // defined(DATA_TYPE)