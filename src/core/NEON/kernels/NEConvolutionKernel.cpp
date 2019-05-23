/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEConvolutionKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <arm_neon.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <tuple>

namespace arm_compute
{
namespace
{
const uint16x8_t max_int16 = vdupq_n_u16(INT16_MAX);

inline void store_results(const int32x4_t &out, const int32x4_t &out2, int16_t *output)
{
    const int16x8_t s16results = vcombine_s16(vqmovn_s32(out),
                                              vqmovn_s32(out2));
    vst1q_s16(output, s16results);
}

inline void store_results(const int32x4_t &out, const int32x4_t &out2, uint8_t *output)
{
    const uint8x8_t u8results = vqmovn_u16(vcombine_u16(vqmovun_s32(out),
                                                        vqmovun_s32(out2)));
    vst1_u8(output, u8results);
}

inline void store_results(const uint32x4_t &out, const uint32x4_t &out2, int16_t *output)
{
    const uint16x8_t u16results = vcombine_u16(vqmovn_u32(out), vqmovn_u32(out2));
    const int16x8_t  s16results = vreinterpretq_s16_u16(vminq_u16(u16results, max_int16));
    vst1q_s16(output, s16results);
}

inline void store_results(const uint32x4_t &out, const uint32x4_t &out2, uint8_t *output)
{
    const uint8x8_t u8results = vqmovn_u16(vcombine_u16(vqmovn_u32(out),
                                                        vqmovn_u32(out2)));
    vst1_u8(output, u8results);
}

inline void store_results(const int16x8_t &out, const int16x8_t &out2, int16_t *output)
{
    vst1q_s16(output, out);
    vst1q_s16(output + 8, out2);
}

inline void store_results(const int16x8_t &out, const int16x8_t &out2, uint8_t *output)
{
    const uint8x16_t u8results = vcombine_u8(vqmovun_s16(out),
                                             vqmovun_s16(out2));
    vst1q_u8(output, u8results);
}

inline void store_results(const uint16x8_t &out, const uint16x8_t &out2, uint8_t *output)
{
    const uint8x16_t u8results = vcombine_u8(vqmovn_u16(out),
                                             vqmovn_u16(out2));
    vst1q_u8(output, u8results);
}

inline void store_results(const uint16x8_t &out, const uint16x8_t &out2, int16_t *output)
{
    vst1q_s16(output, vreinterpretq_s16_u16(vminq_u16(out, max_int16)));
    vst1q_s16(output + 8, vreinterpretq_s16_u16(vminq_u16(out2, max_int16)));
}

inline void convolve_row3x1_unrolled(int32x4_t &out, int32x4_t &out2, const uint8x16_t &row_data, const int16x4_t &mat0, const int16x4_t &mat1, const int16x4_t &mat2)
{
    // Convert to s16 and split in blocks of 4 values:
    const int16x8_t s16_tmp0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(row_data)));
    const int16x8_t s16_tmp1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(row_data)));

    const int16x4x3_t row =
    {
        {
            vget_low_s16(s16_tmp0),
            vget_high_s16(s16_tmp0),
            vget_low_s16(s16_tmp1)
        }
    };

    // Calculate row left value for pixels [0,3]
    out = vmlal_s16(out, row.val[0], mat0);
    // Calculate row middle value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 1), mat1);
    // Calculate row right value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 2), mat2);

    // Calculate row left value for pixels [4,7]
    out2 = vmlal_s16(out2, row.val[1], mat0);
    // Calculate row middle value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 1), mat1);
    // Calculate row right value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 2), mat2);
}

inline void convolve_row3x1(int32x4_t &out, int32x4_t &out2, const uint8x16_t &row_data, const int16_t *convolution)
{
    const int16x4_t mat0 = vld1_dup_s16(convolution);
    const int16x4_t mat1 = vld1_dup_s16(convolution + 1);
    const int16x4_t mat2 = vld1_dup_s16(convolution + 2);

    convolve_row3x1_unrolled(out, out2, row_data, mat0, mat1, mat2);
}

inline void convolve_row5x1(int32x4_t &out, int32x4_t &out2, const uint8x16_t &row_data, const int16_t *convolution)
{
    const int16x4_t mat0 = vld1_dup_s16(convolution);
    const int16x4_t mat1 = vld1_dup_s16(convolution + 1);
    const int16x4_t mat2 = vld1_dup_s16(convolution + 2);
    const int16x4_t mat3 = vld1_dup_s16(convolution + 3);
    const int16x4_t mat4 = vld1_dup_s16(convolution + 4);

    // Convert to s16 and split in blocks of 4 values:
    const int16x8_t s16_tmp0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(row_data)));
    const int16x8_t s16_tmp1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(row_data)));

    const int16x4x3_t row =
    {
        {
            vget_low_s16(s16_tmp0),
            vget_high_s16(s16_tmp0),
            vget_low_s16(s16_tmp1)
        }
    };

    // Calculate row left 2 value for pixels [0,3]
    out = vmlal_s16(out, row.val[0], mat0);
    // Calculate row left 1 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 1), mat1);
    // Calculate row middle value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 2), mat2);
    // Calculate row right +1 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 3), mat3);
    // Calculate row right +2 value for pixels [0,3]
    out = vmlal_s16(out, row.val[1], mat4);

    // Calculate row left 2 value for pixels [4,7]
    out2 = vmlal_s16(out2, row.val[1], mat0);
    // Calculate row left 1 value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 1), mat1);
    // Calculate row middle value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 2), mat2);
    // Calculate row right +1 value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 3), mat3);
    // Calculate row right +2 value for pixels [4,7]
    out2 = vmlal_s16(out2, row.val[2], mat4);
}

inline void convolve_row7x1(int32x4_t &out, int32x4_t &out2, const uint8x16_t &row_data, const int16_t *convolution)
{
    const int16x4_t mat0 = vld1_dup_s16(convolution);
    const int16x4_t mat1 = vld1_dup_s16(convolution + 1);
    const int16x4_t mat2 = vld1_dup_s16(convolution + 2);
    const int16x4_t mat3 = vld1_dup_s16(convolution + 3);
    const int16x4_t mat4 = vld1_dup_s16(convolution + 4);
    const int16x4_t mat5 = vld1_dup_s16(convolution + 5);
    const int16x4_t mat6 = vld1_dup_s16(convolution + 6);

    // Convert to s16 and split in blocks of 4 values:
    const int16x8_t s16_tmp0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(row_data)));
    const int16x8_t s16_tmp1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(row_data)));

    const int16x4x4_t row =
    {
        {
            vget_low_s16(s16_tmp0),
            vget_high_s16(s16_tmp0),
            vget_low_s16(s16_tmp1),
            vget_high_s16(s16_tmp1)
        }
    };

    // Calculate row left 3 value for pixels [0,3]
    out = vmlal_s16(out, row.val[0], mat0);
    // Calculate row left 2 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 1), mat1);
    // Calculate row left 1 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 2), mat2);
    // Calculate row middle value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 3), mat3);
    // Calculate row right +1 value for pixels [0,3]
    out = vmlal_s16(out, row.val[1], mat4);
    // Calculate row right +2 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[1], row.val[2], 1), mat5);
    // Calculate row right +3 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[1], row.val[2], 2), mat6);

    // Calculate row left 3 value for pixels [4,7]
    out2 = vmlal_s16(out2, row.val[1], mat0);
    // Calculate row left 2 value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 1), mat1);
    // Calculate row left 1 value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 2), mat2);
    // Calculate row middle value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 3), mat3);
    // Calculate row right +1 value for pixels [4,7]
    out2 = vmlal_s16(out2, row.val[2], mat4);
    // Calculate row right +2 value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[2], row.val[3], 1), mat5);
    // Calculate row right +3 value for pixels [4,7]
    out2 = vmlal_s16(out2, vext_s16(row.val[2], row.val[3], 2), mat6);
}

inline void convolve_row9x1(int32x4_t &out, int32x4_t &out2, const uint8x16_t &row_data, const int16_t *convolution)
{
    const int16x4_t mat0 = vld1_dup_s16(convolution);
    const int16x4_t mat1 = vld1_dup_s16(convolution + 1);
    const int16x4_t mat2 = vld1_dup_s16(convolution + 2);
    const int16x4_t mat3 = vld1_dup_s16(convolution + 3);
    const int16x4_t mat4 = vld1_dup_s16(convolution + 4);
    const int16x4_t mat5 = vld1_dup_s16(convolution + 5);
    const int16x4_t mat6 = vld1_dup_s16(convolution + 6);
    const int16x4_t mat7 = vld1_dup_s16(convolution + 7);
    const int16x4_t mat8 = vld1_dup_s16(convolution + 8);

    // Convert to s16 and split in blocks of 4 values:
    const int16x8_t s16_tmp0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(row_data)));
    const int16x8_t s16_tmp1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(row_data)));

    const int16x4x4_t row =
    {
        {
            vget_low_s16(s16_tmp0),
            vget_high_s16(s16_tmp0),
            vget_low_s16(s16_tmp1),
            vget_high_s16(s16_tmp1)
        }
    };

    // Calculate row left 4 value for pixels [0,3]
    out = vmlal_s16(out, row.val[0], mat0);
    // Calculate row left 3 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 1), mat1);
    // Calculate row left 2 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 2), mat2);
    // Calculate row left 1 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[0], row.val[1], 3), mat3);
    // Calculate row middle value for pixels [0,3]
    out = vmlal_s16(out, row.val[1], mat4);
    // Calculate row right +1 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[1], row.val[2], 1), mat5);
    // Calculate row right +2 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[1], row.val[2], 2), mat6);
    // Calculate row right +3 value for pixels [0,3]
    out = vmlal_s16(out, vext_s16(row.val[1], row.val[2], 3), mat7);
    // Calculate row right +4 value for pixels [0,3]
    out = vmlal_s16(out, row.val[2], mat8);

    // Calculate row left 4 value for pixels [0,3]
    out2 = vmlal_s16(out2, row.val[1], mat0);
    // Calculate row left 3 value for pixels [0,3]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 1), mat1);
    // Calculate row left 2 value for pixels [0,3]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 2), mat2);
    // Calculate row left 1 value for pixels [0,3]
    out2 = vmlal_s16(out2, vext_s16(row.val[1], row.val[2], 3), mat3);
    // Calculate row middle value for pixels [0,3]
    out2 = vmlal_s16(out2, row.val[2], mat4);
    // Calculate row right +1 value for pixels [0,3]
    out2 = vmlal_s16(out2, vext_s16(row.val[2], row.val[3], 1), mat5);
    // Calculate row right +2 value for pixels [0,3]
    out2 = vmlal_s16(out2, vext_s16(row.val[2], row.val[3], 2), mat6);
    // Calculate row right +3 value for pixels [0,3]
    out2 = vmlal_s16(out2, vext_s16(row.val[2], row.val[3], 3), mat7);
    // Calculate row right +4 value for pixels [0,3]
    out2 = vmlal_s16(out2, row.val[3], mat8);
}
} // namespace

/****************************************************************************************\
 *                                    Square Convolution                                *
\****************************************************************************************/

template <unsigned int matrix_size>
NEConvolutionKernel<matrix_size>::NEConvolutionKernel()
    : INESimpleKernel(), _scale(0), _convolution{ {} }
{
}

template <unsigned int matrix_size>
BorderSize             NEConvolutionKernel<matrix_size>::border_size() const
{
    return BorderSize{ matrix_size / 2 };
}

template <unsigned int matrix_size>
void NEConvolutionKernel<matrix_size>::configure(const ITensor *input, ITensor *output, const int16_t *conv, uint32_t scale, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, conv);

    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);

    _input  = input;
    _output = output;

    std::copy_n(conv, _convolution.size(), _convolution.begin());

    if(scale == 0)
    {
        _scale = calculate_matrix_scale(_convolution.data(), matrix_size);
    }
    else
    {
        _scale = scale;
    }

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, matrix_size),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

template <>
template <typename OutputType>
void NEConvolutionKernel<3>::convolution(const Window &win)
{
    static_assert(sizeof(OutputType) == sizeof(uint8_t) || sizeof(OutputType) == sizeof(int16_t), "The output buffer can only be u8 or s16");
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    Iterator input(_input, win);
    Iterator output(_output, win);

    // Load the matrix's coefficients into NEON registers:
    const int16x4_t   mat00     = vld1_dup_s16(_convolution.data());
    const int16x4_t   mat01     = vld1_dup_s16(_convolution.data() + 1);
    const int16x4_t   mat02     = vld1_dup_s16(_convolution.data() + 2);
    const int16x4_t   mat10     = vld1_dup_s16(_convolution.data() + 3);
    const int16x4_t   mat11     = vld1_dup_s16(_convolution.data() + 4);
    const int16x4_t   mat12     = vld1_dup_s16(_convolution.data() + 5);
    const int16x4_t   mat20     = vld1_dup_s16(_convolution.data() + 6);
    const int16x4_t   mat21     = vld1_dup_s16(_convolution.data() + 7);
    const int16x4_t   mat22     = vld1_dup_s16(_convolution.data() + 8);
    const float32x4_t scale_val = vdupq_n_f32(1.0f / _scale);

    const unsigned char *input_top_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-1, -1));
    const unsigned char *input_mid_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-1, 0));
    const unsigned char *input_low_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-1, 1));

    execute_window_loop(win, [&](const Coordinates &)
    {
        int32x4_t out  = vdupq_n_s32(0);
        int32x4_t out2 = vdupq_n_s32(0);

        // Load 16 bytes from the top row:
        const uint8x16_t top_data = vld1q_u8(input_top_ptr + input.offset());
        convolve_row3x1_unrolled(out, out2, top_data, mat00, mat01, mat02);

        // Load 16 bytes from the middle row:
        const uint8x16_t mid_data = vld1q_u8(input_mid_ptr + input.offset());
        convolve_row3x1_unrolled(out, out2, mid_data, mat10, mat11, mat12);

        // Load 16 bytes from the middle row:
        const uint8x16_t low_data = vld1q_u8(input_low_ptr + input.offset());
        convolve_row3x1_unrolled(out, out2, low_data, mat20, mat21, mat22);

        // Apply scale
        if(_scale != 1)
        {
            // Convert to F32, scale and convert back to S32
            out  = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out), scale_val));
            out2 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out2), scale_val));
        }

        // Clamp and store as U8 or S16:
        store_results(out, out2, reinterpret_cast<OutputType *>(output.ptr()));
    },
    input, output);
}

template <>
template <typename OutputType>
void NEConvolutionKernel<5>::convolution(const Window &win)
{
    static_assert(sizeof(OutputType) == sizeof(uint8_t) || sizeof(OutputType) == sizeof(int16_t), "The output buffer can only be u8 or s16");
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    Iterator input(_input, win);
    Iterator output(_output, win);

    const float32x4_t scale_val = vdupq_n_f32(1.0f / _scale);

    const unsigned char *input_top2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-2, -2));
    const unsigned char *input_top1_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-2, -1));
    const unsigned char *input_mid_ptr  = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-2, 0));
    const unsigned char *input_low1_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-2, 1));
    const unsigned char *input_low2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-2, 2));

    execute_window_loop(win, [&](const Coordinates &)
    {
        int32x4_t out  = vdupq_n_s32(0);
        int32x4_t out2 = vdupq_n_s32(0);

        // Load 16 bytes from the top2 row:
        const uint8x16_t data_t2 = vld1q_u8(input_top2_ptr + input.offset());
        convolve_row5x1(out, out2, data_t2, _convolution.data());

        // Load 16 bytes from the top1 row:
        const uint8x16_t data_t1 = vld1q_u8(input_top1_ptr + input.offset());
        convolve_row5x1(out, out2, data_t1, _convolution.data() + 5);

        // Load 16 bytes from the middle row:
        const uint8x16_t data_m = vld1q_u8(input_mid_ptr + input.offset());
        convolve_row5x1(out, out2, data_m, _convolution.data() + 10);

        // Load 16 bytes from the low1 row:
        const uint8x16_t data_b1 = vld1q_u8(input_low1_ptr + input.offset());
        convolve_row5x1(out, out2, data_b1, _convolution.data() + 15);

        // Load 16 bytes from the low2 row:
        const uint8x16_t data_b2 = vld1q_u8(input_low2_ptr + input.offset());
        convolve_row5x1(out, out2, data_b2, _convolution.data() + 20);

        // Apply scale
        if(_scale != 1)
        {
            // Convert to F32, scale and convert back to S32
            out  = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out), scale_val));
            out2 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out2), scale_val));
        }

        // Clamp and store as U8 or S16:
        store_results(out, out2, reinterpret_cast<OutputType *>(output.ptr()));
    },
    input, output);
}

template <>
template <typename OutputType>
void NEConvolutionKernel<7>::convolution(const Window &win)
{
    static_assert(sizeof(OutputType) == sizeof(uint8_t) || sizeof(OutputType) == sizeof(int16_t), "The output buffer can only be u8 or s16");
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    Iterator input(_input, win);
    Iterator output(_output, win);

    const float32x4_t scale_val = vdupq_n_f32(1.0f / _scale);

    const unsigned char *input_top3_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-3, -3));
    const unsigned char *input_top2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-3, -2));
    const unsigned char *input_top1_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-3, -1));
    const unsigned char *input_mid_ptr  = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-3, 0));
    const unsigned char *input_low1_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-3, 1));
    const unsigned char *input_low2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-3, 2));
    const unsigned char *input_low3_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-3, 3));

    execute_window_loop(win, [&](const Coordinates &)
    {
        int32x4_t out  = vdupq_n_s32(0);
        int32x4_t out2 = vdupq_n_s32(0);

        // Load 16 bytes from the top3 row:
        const uint8x16_t data_t3 = vld1q_u8(input_top3_ptr + input.offset());
        convolve_row7x1(out, out2, data_t3, _convolution.data());

        // Load 16 bytes from the top2 row:
        const uint8x16_t data_t2 = vld1q_u8(input_top2_ptr + input.offset());
        convolve_row7x1(out, out2, data_t2, _convolution.data() + 7);

        // Load 16 bytes from the top1 row:
        const uint8x16_t data_t1 = vld1q_u8(input_top1_ptr + input.offset());
        convolve_row7x1(out, out2, data_t1, _convolution.data() + 14);

        // Load 16 bytes from the middle row:
        const uint8x16_t data_m = vld1q_u8(input_mid_ptr + input.offset());
        convolve_row7x1(out, out2, data_m, _convolution.data() + 21);

        // Load 16 bytes from the low1 row:
        const uint8x16_t data_b1 = vld1q_u8(input_low1_ptr + input.offset());
        convolve_row7x1(out, out2, data_b1, _convolution.data() + 28);

        // Load 16 bytes from the low2 row:
        const uint8x16_t data_b2 = vld1q_u8(input_low2_ptr + input.offset());
        convolve_row7x1(out, out2, data_b2, _convolution.data() + 35);

        // Load 16 bytes from the low3 row:
        const uint8x16_t data_b3 = vld1q_u8(input_low3_ptr + input.offset());
        convolve_row7x1(out, out2, data_b3, _convolution.data() + 42);

        // Apply scale
        if(_scale != 1)
        {
            // Convert to F32, scale and convert back to S32
            out  = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out), scale_val));
            out2 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out2), scale_val));
        }

        // Clamp and store as U8 or S16:
        store_results(out, out2, reinterpret_cast<OutputType *>(output.ptr()));
    },
    input, output);
}

template <>
template <typename OutputType>
void NEConvolutionKernel<9>::convolution(const Window &win)
{
    static_assert(sizeof(OutputType) == sizeof(uint8_t) || sizeof(OutputType) == sizeof(int16_t), "The output buffer can only be u8 or s16");
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    Iterator input(_input, win);
    Iterator output(_output, win);

    const float32x4_t scale_val = vdupq_n_f32(1.0f / _scale);

    const unsigned char *input_top4_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-4, -4));
    const unsigned char *input_top3_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-4, -3));
    const unsigned char *input_top2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-4, -2));
    const unsigned char *input_top1_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-4, -1));
    const unsigned char *input_mid_ptr  = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-4, 0));
    const unsigned char *input_low1_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-4, 1));
    const unsigned char *input_low2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-4, 2));
    const unsigned char *input_low3_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-4, 3));
    const unsigned char *input_low4_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-4, 4));

    execute_window_loop(win, [&](const Coordinates &)
    {
        int32x4_t out  = vdupq_n_s32(0);
        int32x4_t out2 = vdupq_n_s32(0);

        // Load 16 bytes from the top4 row:
        const uint8x16_t data_t4 = vld1q_u8(input_top4_ptr + input.offset());
        convolve_row9x1(out, out2, data_t4, _convolution.data());

        // Load 16 bytes from the top3 row:
        const uint8x16_t data_t3 = vld1q_u8(input_top3_ptr + input.offset());
        convolve_row9x1(out, out2, data_t3, _convolution.data() + 9);

        // Load 16 bytes from the top2 row:
        const uint8x16_t data_t2 = vld1q_u8(input_top2_ptr + input.offset());
        convolve_row9x1(out, out2, data_t2, _convolution.data() + 18);

        // Load 16 bytes from the top1 row:
        const uint8x16_t data_t1 = vld1q_u8(input_top1_ptr + input.offset());
        convolve_row9x1(out, out2, data_t1, _convolution.data() + 27);

        // Load 16 bytes from the middle row:
        const uint8x16_t data_m = vld1q_u8(input_mid_ptr + input.offset());
        convolve_row9x1(out, out2, data_m, _convolution.data() + 36);

        // Load 16 bytes from the low1 row:
        const uint8x16_t data_b1 = vld1q_u8(input_low1_ptr + input.offset());
        convolve_row9x1(out, out2, data_b1, _convolution.data() + 45);

        // Load 16 bytes from the low2 row:
        const uint8x16_t data_b2 = vld1q_u8(input_low2_ptr + input.offset());
        convolve_row9x1(out, out2, data_b2, _convolution.data() + 54);

        // Load 16 bytes from the low3 row:
        const uint8x16_t data_b3 = vld1q_u8(input_low3_ptr + input.offset());
        convolve_row9x1(out, out2, data_b3, _convolution.data() + 63);

        // Load 16 bytes from the low4 row:
        const uint8x16_t data_b4 = vld1q_u8(input_low4_ptr + input.offset());
        convolve_row9x1(out, out2, data_b4, _convolution.data() + 72);

        // Apply scale
        if(_scale != 1)
        {
            // Convert to F32, scale and convert back to S32
            out  = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out), scale_val));
            out2 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out2), scale_val));
        }

        // Clamp and store as U8 or S16:
        store_results(out, out2, reinterpret_cast<OutputType *>(output.ptr()));
    },
    input, output);
}

template <unsigned int matrix_size>
void NEConvolutionKernel<matrix_size>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_output->info()->data_type())
    {
        case DataType::U8:
            convolution<uint8_t>(window);
            break;
        case DataType::S16:
            convolution<int16_t>(window);
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported Data type!");
            break;
    }
}

template class arm_compute::NEConvolutionKernel<3>;
template class arm_compute::NEConvolutionKernel<5>;
template class arm_compute::NEConvolutionKernel<7>;
template class arm_compute::NEConvolutionKernel<9>;

/****************************************************************************************\
 *                              Separable Square Convolution                            *
\****************************************************************************************/

template <unsigned int matrix_size>
NESeparableConvolutionHorKernel<matrix_size>::NESeparableConvolutionHorKernel()
    : _conv_row{ { 0 } }, _border_size(0)
{
}

template <unsigned int matrix_size>
BorderSize             NESeparableConvolutionHorKernel<matrix_size>::border_size() const
{
    return _border_size;
}

template <unsigned int matrix_size>
void NESeparableConvolutionHorKernel<matrix_size>::configure(const ITensor *input, ITensor *output, const int16_t *conv_row, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, conv_row);

    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U16, DataType::S16, DataType::S32);

    _input  = input;
    _output = output;
    std::copy_n(conv_row, _conv_row.size(), _conv_row.begin());
    _border_size = BorderSize(border_undefined ? 0 : matrix_size / 2, matrix_size / 2);

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;

    Window                 win = calculate_max_window_horizontal(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), -border_size().left, num_elems_read_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

template <unsigned int matrix_size>
void NESeparableConvolutionHorKernel<matrix_size>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    switch(_output->info()->data_type())
    {
        case DataType::U16:
            convolve<uint16_t>(window);
            break;
        case DataType::S16:
            convolve<int16_t>(window);
            break;
        case DataType::S32:
            convolve<int32_t>(window);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported intermediate data type!");
            break;
    }
}

template <>
template <>
inline void NESeparableConvolutionHorKernel<5>::convolve<uint16_t>(const Window &window)
{
    Window win_in(window);
    win_in.shift(Window::DimX, -2);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        const uint16x8x2_t data_u16 =
        {
            {
                vmovl_u8(vget_low_u8(data)),
                vmovl_u8(vget_high_u8(data))
            }
        };

        uint16x8_t out = vmulq_n_u16(data_u16.val[0], _conv_row[0]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 1), _conv_row[1]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 2), _conv_row[2]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 3), _conv_row[3]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 4), _conv_row[4]);

        vst1q_u16(reinterpret_cast<uint16_t *>(output.ptr()), out);
    },
    input, output);
}

template <>
template <>
inline void NESeparableConvolutionHorKernel<5>::convolve<int16_t>(const Window &window)
{
    Window win_in(window);
    win_in.shift(Window::DimX, -2);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        const int16x8x2_t data_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
            }
        };

        int16x8_t out = vmulq_n_s16(data_s16.val[0], _conv_row[0]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 1), _conv_row[1]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 2), _conv_row[2]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 3), _conv_row[3]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 4), _conv_row[4]);

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), out);
    },
    input, output);
}

template <>
template <>
void NESeparableConvolutionHorKernel<5>::convolve<int32_t>(const Window &window)
{
    Window win_in(window);
    win_in.shift(Window::DimX, -2);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        const int16x8x2_t data_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
            }
        };

        const int16x8_t data_s16_l1 = vextq_s16(data_s16.val[0], data_s16.val[1], 1);
        const int16x8_t data_s16_m  = vextq_s16(data_s16.val[0], data_s16.val[1], 2);
        const int16x8_t data_s16_r1 = vextq_s16(data_s16.val[0], data_s16.val[1], 3);
        const int16x8_t data_s16_r2 = vextq_s16(data_s16.val[0], data_s16.val[1], 4);

        int32x4_t out_low = vmull_n_s16(vget_low_s16(data_s16.val[0]), _conv_row[0]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_l1), _conv_row[1]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_m), _conv_row[2]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_r1), _conv_row[3]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_r2), _conv_row[4]);

        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()), out_low);

        int32x4_t out_high = vmull_n_s16(vget_high_s16(data_s16.val[0]), _conv_row[0]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_l1), _conv_row[1]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_m), _conv_row[2]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_r1), _conv_row[3]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_r2), _conv_row[4]);

        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()) + 4, out_high);
    },
    input, output);
}

template <>
template <>
inline void NESeparableConvolutionHorKernel<7>::convolve<uint16_t>(const Window &window)
{
    Window win_in(window);
    win_in.shift(Window::DimX, -3);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        const uint16x8x2_t data_u16 =
        {
            {
                vmovl_u8(vget_low_u8(data)),
                vmovl_u8(vget_high_u8(data))
            }
        };

        uint16x8_t out = vmulq_n_u16(data_u16.val[0], _conv_row[0]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 1), _conv_row[1]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 2), _conv_row[2]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 3), _conv_row[3]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 4), _conv_row[4]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 5), _conv_row[5]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 6), _conv_row[6]);

        vst1q_u16(reinterpret_cast<uint16_t *>(output.ptr()), out);
    },
    input, output);
}

template <>
template <>
inline void NESeparableConvolutionHorKernel<7>::convolve<int16_t>(const Window &window)
{
    Window win_in(window);
    win_in.shift(Window::DimX, -3);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        const int16x8x2_t data_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
            }
        };

        int16x8_t out = vmulq_n_s16(data_s16.val[0], _conv_row[0]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 1), _conv_row[1]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 2), _conv_row[2]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 3), _conv_row[3]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 4), _conv_row[4]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 5), _conv_row[5]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 6), _conv_row[6]);

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), out);
    },
    input, output);
}

template <>
template <>
void NESeparableConvolutionHorKernel<7>::convolve<int32_t>(const Window &window)
{
    Window win_in(window);
    win_in.shift(Window::DimX, -3);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        const int16x8x2_t data_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
            }
        };

        const int16x8_t data_s16_l2 = vextq_s16(data_s16.val[0], data_s16.val[1], 1);
        const int16x8_t data_s16_l1 = vextq_s16(data_s16.val[0], data_s16.val[1], 2);
        const int16x8_t data_s16_m  = vextq_s16(data_s16.val[0], data_s16.val[1], 3);
        const int16x8_t data_s16_r1 = vextq_s16(data_s16.val[0], data_s16.val[1], 4);
        const int16x8_t data_s16_r2 = vextq_s16(data_s16.val[0], data_s16.val[1], 5);
        const int16x8_t data_s16_r3 = vextq_s16(data_s16.val[0], data_s16.val[1], 6);

        int32x4_t out_low = vmull_n_s16(vget_low_s16(data_s16.val[0]), _conv_row[0]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_l2), _conv_row[1]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_l1), _conv_row[2]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_m), _conv_row[3]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_r1), _conv_row[4]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_r2), _conv_row[5]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_r3), _conv_row[6]);

        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()), out_low);

        int32x4_t out_high = vmull_n_s16(vget_high_s16(data_s16.val[0]), _conv_row[0]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_l2), _conv_row[1]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_l1), _conv_row[2]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_m), _conv_row[3]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_r1), _conv_row[4]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_r2), _conv_row[5]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_r3), _conv_row[6]);

        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()) + 4, out_high);
    },
    input, output);
}

template <>
template <>
inline void NESeparableConvolutionHorKernel<9>::convolve<uint16_t>(const Window &window)
{
    Window win_in(window);
    win_in.shift(Window::DimX, -4);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        const uint16x8x2_t data_u16 =
        {
            {
                vmovl_u8(vget_low_u8(data)),
                vmovl_u8(vget_high_u8(data))
            }
        };

        uint16x8_t out = vmulq_n_u16(data_u16.val[0], _conv_row[0]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 1), _conv_row[1]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 2), _conv_row[2]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 3), _conv_row[3]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 4), _conv_row[4]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 5), _conv_row[5]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 6), _conv_row[6]);
        out            = vmlaq_n_u16(out, vextq_u16(data_u16.val[0], data_u16.val[1], 7), _conv_row[7]);
        out            = vmlaq_n_u16(out, data_u16.val[1], _conv_row[8]);

        vst1q_u16(reinterpret_cast<uint16_t *>(output.ptr()), out);
    },
    input, output);
}

template <>
template <>
inline void NESeparableConvolutionHorKernel<9>::convolve<int16_t>(const Window &window)
{
    Window win_in(window);
    win_in.shift(Window::DimX, -4);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        const int16x8x2_t data_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
            }
        };

        int16x8_t out = vmulq_n_s16(data_s16.val[0], _conv_row[0]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 1), _conv_row[1]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 2), _conv_row[2]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 3), _conv_row[3]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 4), _conv_row[4]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 5), _conv_row[5]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 6), _conv_row[6]);
        out           = vmlaq_n_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 7), _conv_row[7]);
        out           = vmlaq_n_s16(out, data_s16.val[1], _conv_row[8]);

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), out);
    },
    input, output);
}

template <>
template <>
void NESeparableConvolutionHorKernel<9>::convolve<int32_t>(const Window &window)
{
    Window win_in(window);
    win_in.shift(Window::DimX, -4);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t data = vld1q_u8(input.ptr());

        const int16x8x2_t data_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
            }
        };

        const int16x8_t data_s16_l3 = vextq_s16(data_s16.val[0], data_s16.val[1], 1);
        const int16x8_t data_s16_l2 = vextq_s16(data_s16.val[0], data_s16.val[1], 2);
        const int16x8_t data_s16_l1 = vextq_s16(data_s16.val[0], data_s16.val[1], 3);
        const int16x8_t data_s16_m  = vextq_s16(data_s16.val[0], data_s16.val[1], 4);
        const int16x8_t data_s16_r1 = vextq_s16(data_s16.val[0], data_s16.val[1], 5);
        const int16x8_t data_s16_r2 = vextq_s16(data_s16.val[0], data_s16.val[1], 6);
        const int16x8_t data_s16_r3 = vextq_s16(data_s16.val[0], data_s16.val[1], 7);

        int32x4_t out_low = vmull_n_s16(vget_low_s16(data_s16.val[0]), _conv_row[0]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_l3), _conv_row[1]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_l2), _conv_row[2]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_l1), _conv_row[3]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_m), _conv_row[4]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_r1), _conv_row[5]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_r2), _conv_row[6]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16_r3), _conv_row[7]);
        out_low           = vmlal_n_s16(out_low, vget_low_s16(data_s16.val[1]), _conv_row[8]);

        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()), out_low);

        int32x4_t out_high = vmull_n_s16(vget_high_s16(data_s16.val[0]), _conv_row[0]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_l3), _conv_row[1]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_l2), _conv_row[2]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_l1), _conv_row[3]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_m), _conv_row[4]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_r1), _conv_row[5]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_r2), _conv_row[6]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16_r3), _conv_row[7]);
        out_high           = vmlal_n_s16(out_high, vget_high_s16(data_s16.val[1]), _conv_row[8]);

        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()) + 4, out_high);
    },
    input, output);
}

template class arm_compute::NESeparableConvolutionHorKernel<5>;
template class arm_compute::NESeparableConvolutionHorKernel<7>;
template class arm_compute::NESeparableConvolutionHorKernel<9>;

template <unsigned int matrix_size>
NESeparableConvolutionVertKernel<matrix_size>::NESeparableConvolutionVertKernel()
    : _conv_col{ { 0 } }, _scale(0)
{
}

template <unsigned int matrix_size>
BorderSize             NESeparableConvolutionVertKernel<matrix_size>::border_size() const
{
    return BorderSize{ matrix_size / 2, 0 };
}

template <unsigned int matrix_size>
void NESeparableConvolutionVertKernel<matrix_size>::configure(const ITensor *input, ITensor *output, const int16_t *conv_col, uint32_t scale, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, conv_col);

    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U16, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON(scale == 0);

    _input  = input;
    _output = output;
    std::copy_n(conv_col, _conv_col.size(), _conv_col.begin());
    _scale = scale;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 16;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), 0, -border_size().top, num_elems_read_per_iteration, matrix_size),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

template <unsigned int matrix_size>
void NESeparableConvolutionVertKernel<matrix_size>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_input->info()->data_type())
    {
        case DataType::U16:
            switch(_output->info()->data_type())
            {
                case DataType::U8:
                    convolution_u16<uint8_t>(window);
                    break;
                case DataType::S16:
                    convolution_u16<int16_t>(window);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
            break;
        case DataType::S16:
            switch(_output->info()->data_type())
            {
                case DataType::U8:
                    convolution_s16<uint8_t>(window);
                    break;
                case DataType::S16:
                    convolution_s16<int16_t>(window);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
            break;
        case DataType::S32:
            switch(_output->info()->data_type())
            {
                case DataType::U8:
                    convolution_s32<uint8_t>(window);
                    break;
                case DataType::S16:
                    convolution_s32<int16_t>(window);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported intermediate data type!");
            break;
    }
}

template <unsigned int matrix_size>
template <typename OutputType>
void NESeparableConvolutionVertKernel<matrix_size>::convolution_u16(const Window &win)
{
    static_assert(sizeof(OutputType) == sizeof(uint8_t) || sizeof(OutputType) == sizeof(int16_t), "The output buffer can only be u8 or s16");

    Window win_in(win);
    win_in.set_dimension_step(Window::DimX, 8);

    Iterator in(_input, win_in);
    Iterator out(_output, win);

    std::array<unsigned char *, matrix_size> input_ptrs{ {} };
    const float32x4_t oneoverscale = vdupq_n_f32(1.0f / _scale);
    const int         k_half       = matrix_size / 2;

    // Set row pointers
    for(int i = -k_half; i <= k_half; ++i)
    {
        input_ptrs[k_half + i] = _input->ptr_to_element(Coordinates(0, i));
    }

    execute_window_loop(win, [&](const Coordinates &)
    {
        uint16x8_t out0 = vdupq_n_u16(0);
        uint16x8_t out1 = vdupq_n_u16(0);

        // First half
        for(unsigned int r = 0; r < matrix_size; ++r)
        {
            const uint16x8_t data = vld1q_u16(reinterpret_cast<const uint16_t *>(input_ptrs[r] + in.offset()));
            out0                  = vmlaq_n_u16(out0, data, _conv_col[r]);
        }

        in.increment(Window::DimX);

        // Second half
        for(unsigned int r = 0; r < matrix_size; ++r)
        {
            const uint16x8_t data = vld1q_u16(reinterpret_cast<const uint16_t *>(input_ptrs[r] + in.offset()));
            out1                  = vmlaq_n_u16(out1, data, _conv_col[r]);
        }

        //scale the result if needed
        if(_scale != 1)
        {
            float32x4_t out0_f32_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(out0)));
            float32x4_t out0_f32_low  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(out0)));
            out0_f32_high             = vmulq_f32(out0_f32_high, oneoverscale);
            out0_f32_low              = vmulq_f32(out0_f32_low, oneoverscale);
            store_results(vcvtq_u32_f32(out0_f32_low), vcvtq_u32_f32(out0_f32_high), reinterpret_cast<OutputType *>(out.ptr()));

            float32x4_t out1_f32_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(out1)));
            float32x4_t out1_f32_low  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(out1)));
            out1_f32_high             = vmulq_f32(out1_f32_high, oneoverscale);
            out1_f32_low              = vmulq_f32(out1_f32_low, oneoverscale);
            store_results(vcvtq_u32_f32(out1_f32_low), vcvtq_u32_f32(out1_f32_high), reinterpret_cast<OutputType *>(out.ptr()) + 8);
        }
        else
        {
            store_results(out0, out1, reinterpret_cast<OutputType *>(out.ptr()));
        }
    },
    in, out);
}

template <unsigned int matrix_size>
template <typename OutputType>
void NESeparableConvolutionVertKernel<matrix_size>::convolution_s16(const Window &win)
{
    static_assert(sizeof(OutputType) == sizeof(uint8_t) || sizeof(OutputType) == sizeof(int16_t), "The output buffer can only be u8 or s16");

    Window win_in(win);
    win_in.set_dimension_step(Window::DimX, 8);

    Iterator in(_input, win_in);
    Iterator out(_output, win);

    std::array<unsigned char *, matrix_size> input_ptrs{ {} };
    const float32x4_t oneoverscale = vdupq_n_f32(1.0f / _scale);
    const int         k_half       = matrix_size / 2;

    // Set row pointers
    for(int i = -k_half; i <= k_half; ++i)
    {
        input_ptrs[k_half + i] = _input->ptr_to_element(Coordinates(0, i));
    }

    execute_window_loop(win, [&](const Coordinates &)
    {
        int16x8_t out0 = vdupq_n_s16(0);
        int16x8_t out1 = vdupq_n_s16(0);

        // First half
        for(unsigned int r = 0; r < matrix_size; ++r)
        {
            const int16x8_t data = vld1q_s16(reinterpret_cast<const int16_t *>(input_ptrs[r] + in.offset()));
            out0                 = vmlaq_n_s16(out0, data, _conv_col[r]);
        }

        in.increment(Window::DimX);

        // Second half
        for(unsigned int r = 0; r < matrix_size; ++r)
        {
            const int16x8_t data = vld1q_s16(reinterpret_cast<const int16_t *>(input_ptrs[r] + in.offset()));
            out1                 = vmlaq_n_s16(out1, data, _conv_col[r]);
        }

        //scale the result if needed
        if(_scale != 1)
        {
            float32x4_t out0_f32_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(out0)));
            float32x4_t out0_f32_low  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(out0)));
            out0_f32_high             = vmulq_f32(out0_f32_high, oneoverscale);
            out0_f32_low              = vmulq_f32(out0_f32_low, oneoverscale);
            store_results(vcvtq_s32_f32(out0_f32_low), vcvtq_s32_f32(out0_f32_high), reinterpret_cast<OutputType *>(out.ptr()));

            float32x4_t out1_f32_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(out1)));
            float32x4_t out1_f32_low  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(out1)));
            out1_f32_high             = vmulq_f32(out1_f32_high, oneoverscale);
            out1_f32_low              = vmulq_f32(out1_f32_low, oneoverscale);
            store_results(vcvtq_s32_f32(out1_f32_low), vcvtq_s32_f32(out1_f32_high), reinterpret_cast<OutputType *>(out.ptr()) + 8);
        }
        else
        {
            store_results(out0, out1, reinterpret_cast<OutputType *>(out.ptr()));
        }
    },
    in, out);
}

template <unsigned int matrix_size>
template <typename OutputType>
void NESeparableConvolutionVertKernel<matrix_size>::convolution_s32(const Window &win)
{
    static_assert(sizeof(OutputType) == sizeof(uint8_t) || sizeof(OutputType) == sizeof(int16_t), "The output buffer can only be u8 or s16");

    Window win_in(win);
    win_in.set_dimension_step(Window::DimX, 8);

    Iterator in(_input, win_in);
    Iterator out(_output, win);

    std::array<unsigned char *, matrix_size> input_ptrs{ {} };
    const float32x4_t oneoverscale = vdupq_n_f32(1.0f / _scale);
    const int         k_half       = matrix_size / 2;

    // Set row pointers
    for(int i = -k_half; i <= k_half; ++i)
    {
        input_ptrs[k_half + i] = _input->ptr_to_element(Coordinates(0, i));
    }

    const int32x4_t zero = vdupq_n_s32(0);

    execute_window_loop(win, [&](const Coordinates &)
    {
        int32x4x2_t out0 =
        {
            {
                zero,
                zero
            }
        };

        int32x4x2_t out1 =
        {
            {
                zero,
                zero
            }
        };

        // First half
        for(unsigned int r = 0; r < matrix_size; ++r)
        {
            const int32x4x2_t data = vld2q_s32(reinterpret_cast<const int32_t *>(input_ptrs[r] + in.offset()));
            out0.val[0]            = vmlaq_n_s32(out0.val[0], data.val[0], _conv_col[r]);
            out0.val[1]            = vmlaq_n_s32(out0.val[1], data.val[1], _conv_col[r]);
        }

        in.increment(Window::DimX);

        // Second half
        for(unsigned int r = 0; r < matrix_size; ++r)
        {
            const int32x4x2_t data = vld2q_s32(reinterpret_cast<const int32_t *>(input_ptrs[r] + in.offset()));
            out1.val[0]            = vmlaq_n_s32(out1.val[0], data.val[0], _conv_col[r]);
            out1.val[1]            = vmlaq_n_s32(out1.val[1], data.val[1], _conv_col[r]);
        }

        //scale the result if needed
        if(_scale != 1)
        {
            float32x4_t out0_f32_odd  = vcvtq_f32_s32(out0.val[0]);
            float32x4_t out0_f32_even = vcvtq_f32_s32(out0.val[1]);
            out0_f32_odd              = vmulq_f32(out0_f32_odd, oneoverscale);
            out0_f32_even             = vmulq_f32(out0_f32_even, oneoverscale);
            out0.val[0]               = vcvtq_s32_f32(out0_f32_odd);
            out0.val[1]               = vcvtq_s32_f32(out0_f32_even);

            float32x4_t out1_f32_odd  = vcvtq_f32_s32(out1.val[0]);
            float32x4_t out1_f32_even = vcvtq_f32_s32(out1.val[1]);
            out1_f32_odd              = vmulq_f32(out1_f32_odd, oneoverscale);
            out1_f32_even             = vmulq_f32(out1_f32_even, oneoverscale);
            out1.val[0]               = vcvtq_s32_f32(out1_f32_odd);
            out1.val[1]               = vcvtq_s32_f32(out1_f32_even);
        }

        const int32x4x2_t out0_s32 = vzipq_s32(out0.val[0], out0.val[1]);
        store_results(out0_s32.val[0], out0_s32.val[1], reinterpret_cast<OutputType *>(out.ptr()));

        const int32x4x2_t out1_s32 = vzipq_s32(out1.val[0], out1.val[1]);
        store_results(out1_s32.val[0], out1_s32.val[1], reinterpret_cast<OutputType *>(out.ptr()) + 8);
    },
    in, out);
}

template class arm_compute::NESeparableConvolutionVertKernel<5>;
template class arm_compute::NESeparableConvolutionVertKernel<7>;
template class arm_compute::NESeparableConvolutionVertKernel<9>;

/****************************************************************************************\
 *                                 Rectangle Convolution                                *
\****************************************************************************************/

NEConvolutionRectangleKernel::NEConvolutionRectangleKernel()
    : _input(nullptr), _output(nullptr), _scale(0), _convolution(), _border_size(), _func_idx(0)
{
}

BorderSize NEConvolutionRectangleKernel::border_size() const
{
    return _border_size;
}

void NEConvolutionRectangleKernel::configure(const ITensor *input, ITensor *output, const int16_t *conv, uint32_t width, uint32_t height, uint32_t scale, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, conv);

    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON(width != 3 && width != 5 && width != 7 && width != 9);
    ARM_COMPUTE_ERROR_ON(height != 3 && height != 5 && height != 7 && height != 9);
    ARM_COMPUTE_ERROR_ON(0 == scale);

    _input       = input;
    _output      = output;
    _scale       = scale;
    _border_size = BorderSize(height / 2, width / 2);

    // Setup the convolution matrix
    const uint32_t nr_elements = width * height;
    _convolution.resize(nr_elements);
    std::copy_n(conv, nr_elements, _convolution.begin());

    // Set function index to help choose appropriate function in run()
    _func_idx = get_index(height) * 4 + get_index(width);
    ARM_COMPUTE_ERROR_ON(_func_idx > (_nr_supported_sizes * _nr_supported_sizes));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, _border_size);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), -_border_size.left, -_border_size.top, num_elems_read_per_iteration, height),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, _border_size);

    INEKernel::configure(win);
}

void NEConvolutionRectangleKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    using ConvolutionRectangleFunction = void (NEConvolutionRectangleKernel::*)(const Window & window);

    // uint8_t function table
    static const std::array<ConvolutionRectangleFunction, 16> func_table_u8 =
    {
        {
            &NEConvolutionRectangleKernel::convolution<uint8_t, 3, 3>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 3, 5>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 3, 7>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 3, 9>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 5, 3>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 5, 5>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 5, 7>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 5, 9>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 7, 3>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 7, 5>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 7, 7>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 7, 9>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 9, 3>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 9, 5>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 9, 7>,
            &NEConvolutionRectangleKernel::convolution<uint8_t, 9, 9>
        }
    };
    // int16_t function table
    static const std::array<ConvolutionRectangleFunction, 16> func_table_s16 =
    {
        {
            &NEConvolutionRectangleKernel::convolution<int16_t, 3, 3>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 3, 5>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 3, 7>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 3, 9>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 5, 3>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 5, 5>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 5, 7>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 5, 9>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 7, 3>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 7, 5>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 7, 7>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 7, 9>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 9, 3>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 9, 5>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 9, 7>,
            &NEConvolutionRectangleKernel::convolution<int16_t, 9, 9>
        }
    };

    // Run appropriate function
    switch(_output->info()->data_type())
    {
        case DataType::U8:
            ARM_COMPUTE_ERROR_ON(_func_idx >= func_table_u8.size());
            (this->*func_table_u8[_func_idx])(window);
            break;
        case DataType::S16:
            ARM_COMPUTE_ERROR_ON(_func_idx >= func_table_s16.size());
            (this->*func_table_s16[_func_idx])(window);
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}

unsigned int NEConvolutionRectangleKernel::get_index(uint32_t val)
{
    switch(val)
    {
        case 3:
            return 0;
        case 5:
            return 1;
        case 7:
            return 2;
        case 9:
            return 3;
        default:
            ARM_COMPUTE_ERROR("Not supported dimension size");
            return 0;
    }
}

template <typename OutputType, unsigned int rows, unsigned int cols>
void NEConvolutionRectangleKernel::convolution(const Window &win)
{
    static_assert(sizeof(OutputType) == sizeof(uint8_t) || sizeof(OutputType) == sizeof(int16_t), "The output buffer can only be u8 or s16");
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    Iterator input(_input, win);
    Iterator output(_output, win);

    std::array<unsigned char *, rows> input_ptrs{ {} };
    const int16_t    *conv       = _convolution.data();
    const float32x4_t scale_val  = vdupq_n_f32(1.0f / _scale);
    const int         k_row_half = rows / 2;
    const int         k_col_half = cols / 2;

    // Set row pointers
    for(int i = -k_row_half; i <= k_row_half; ++i)
    {
        input_ptrs[k_row_half + i] = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(-k_col_half, i));
    }

    execute_window_loop(win, [&](const Coordinates &)
    {
        int32x4_t out  = vdupq_n_s32(0);
        int32x4_t out2 = vdupq_n_s32(0);

        // Perform appropriate convolution
        for(unsigned int r = 0; r < rows; ++r)
        {
            const uint8x16_t data = vld1q_u8(input_ptrs[r] + input.offset());
            if(3 == cols)
            {
                convolve_row3x1(out, out2, data, conv + r * cols);
            }
            else if(5 == cols)
            {
                convolve_row5x1(out, out2, data, conv + r * cols);
            }
            else if(7 == cols)
            {
                convolve_row7x1(out, out2, data, conv + r * cols);
            }
            else if(9 == cols)
            {
                convolve_row9x1(out, out2, data, conv + r * cols);
            }
            else
            {
                ARM_COMPUTE_ERROR("Unsupported number of columns");
            }
        }

        // Apply scale
        if(_scale != 1)
        {
            // Convert to F32, scale and convert back to S32
            out  = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out), scale_val));
            out2 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(out2), scale_val));
        }

        // Clamp and store as U8 or S16:
        store_results(out, out2, reinterpret_cast<OutputType *>(output.ptr()));
    },
    input, output);
}
} // namespace arm_compute
