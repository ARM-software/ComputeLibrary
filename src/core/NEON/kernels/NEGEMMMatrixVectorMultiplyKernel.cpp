/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixVectorMultiplyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;

template <typename I0, typename I1, typename O>
void NEGEMMMatrixVectorMultiplyKernel::matrix_vector_multiply(const Window &window_in, const Window &window_w, const Window &window_out)
{
    ARM_COMPUTE_ERROR("Unsupported data types");
    ARM_COMPUTE_UNUSED(window_in);
    ARM_COMPUTE_UNUSED(window_w);
    ARM_COMPUTE_UNUSED(window_out);
}

namespace arm_compute
{
template <>
void NEGEMMMatrixVectorMultiplyKernel::matrix_vector_multiply<float, float, float>(const Window &window_in,
                                                                                   const Window &window_w,
                                                                                   const Window &window_out)
{
    Iterator in(_input0, window_in);
    Iterator in2(_input1, window_w);
    Iterator out(_output, window_out);

    const int input_w          = _input0->info()->dimension(0);
    const int input_h          = _input0->info()->dimension(1);
    const int input_stride_x   = _input0->info()->strides_in_bytes().x();
    const int weights_stride_x = _input1->info()->strides_in_bytes().x();
    const int weights_stride_y = _input1->info()->strides_in_bytes().y();
    const int output_stride_x  = _output->info()->strides_in_bytes().x();

    execute_window_loop(window_in, [&](const Coordinates & id)
    {
        // Get pointers
        const uint8_t *const input_ptr   = in.ptr();
        const uint8_t *const weights_ptr = in2.ptr() + id.z() * weights_stride_y;
        auto                 output_ptr  = reinterpret_cast<float *>(out.ptr() + (id.y() + id.z() * input_h) * output_stride_x);

        float32x4_t row_dot = vdupq_n_f32(0.f);
        for(int i = 0; i < input_w; i += 4)
        {
            const auto input   = vld1q_f32(reinterpret_cast<const float *>(input_ptr + i * input_stride_x));
            const auto weights = vld1q_f32(reinterpret_cast<const float *>(weights_ptr + i * weights_stride_x));
            row_dot            = vaddq_f32(row_dot, vmulq_f32(input, weights));
        }

        auto temp = vadd_f32(vget_high_f32(row_dot), vget_low_f32(row_dot));
        temp      = vpadd_f32(temp, temp);

        *output_ptr = vget_lane_f32(temp, 0);
    },
    in, in2, out);
}

template <>
void NEGEMMMatrixVectorMultiplyKernel::matrix_vector_multiply<uint8_t, uint8_t, int32_t>(const Window &window_in,
                                                                                         const Window &window_w,
                                                                                         const Window &window_out)
{
    Iterator in(_input0, window_in);
    Iterator in2(_input1, window_w);
    Iterator out(_output, window_out);

    const int input_offset   = -_input0->info()->quantization_info().offset;
    const int weights_offset = -_input1->info()->quantization_info().offset;

    const int input_w          = _input0->info()->dimension(0);
    const int input_h          = _input0->info()->dimension(1);
    const int input_stride_x   = _input0->info()->strides_in_bytes().x();
    const int weights_stride_x = _input1->info()->strides_in_bytes().x();
    const int weights_stride_y = _input1->info()->strides_in_bytes().y();
    const int output_stride_x  = _output->info()->strides_in_bytes().x();
    const int read_step        = 16 / _input0->info()->element_size();

    const int32x4_t v_input_offset   = vdupq_n_s32(input_offset);
    const int32x4_t v_weights_offset = vdupq_n_s32(weights_offset);

    execute_window_loop(window_in, [&](const Coordinates & id)
    {
        // Get pointers
        const uint8_t *const input_ptr   = in.ptr();
        const uint8_t *const weights_ptr = in2.ptr() + id.z() * weights_stride_y;
        auto                 output_ptr  = reinterpret_cast<int32_t *>(out.ptr() + (id.y() + id.z() * input_h) * output_stride_x);

        int32x4_t row_dot = vdupq_n_s32(0);
        for(int i = 0; i < input_w; i += read_step)
        {
            // Read values
            const auto input   = vld1q_u8(reinterpret_cast<const uint8_t *>(input_ptr + i * input_stride_x));
            const auto weights = vld1q_u8(reinterpret_cast<const uint8_t *>(weights_ptr + i * weights_stride_x));

            // Add offsets
            const int32x4x4_t input_s32 =
            {
                {
                    vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vget_low_u8(input))))),
                    vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vget_low_u8(input))))),
                    vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vget_high_u8(input))))),
                    vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vget_high_u8(input)))))
                }
            };
            const int32x4x4_t weights_s32 =
            {
                {
                    vaddw_s16(v_weights_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vget_low_u8(weights))))),
                    vaddw_s16(v_weights_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vget_low_u8(weights))))),
                    vaddw_s16(v_weights_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vget_high_u8(weights))))),
                    vaddw_s16(v_weights_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vget_high_u8(weights)))))
                }
            };

            // Dot
            row_dot = vaddq_s32(row_dot, vmulq_s32(input_s32.val[0], weights_s32.val[0]));
            row_dot = vaddq_s32(row_dot, vmulq_s32(input_s32.val[1], weights_s32.val[1]));
            row_dot = vaddq_s32(row_dot, vmulq_s32(input_s32.val[2], weights_s32.val[2]));
            row_dot = vaddq_s32(row_dot, vmulq_s32(input_s32.val[3], weights_s32.val[3]));
        }

        // Reduction
        auto temp = vadd_s32(vget_high_s32(row_dot), vget_low_s32(row_dot));
        temp      = vpadd_s32(temp, temp);

        *output_ptr = vget_lane_s32(temp, 0);
    },
    in, in2, out);
}
} //namespace arm_compute

NEGEMMMatrixVectorMultiplyKernel::NEGEMMMatrixVectorMultiplyKernel()
    : _func(nullptr), _input0(nullptr), _input1(nullptr), _output(nullptr), _border_size(0)
{
}

BorderSize NEGEMMMatrixVectorMultiplyKernel::border_size() const
{
    return _border_size;
}

void NEGEMMMatrixVectorMultiplyKernel::configure(const ITensor *input0, const ITensor *input1, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::QASYMM8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input0, input1, output);
    ARM_COMPUTE_ERROR_ON(is_data_type_quantized_asymmetric(input0->info()->data_type()) && (output->info()->data_type() != DataType::S32));
    ARM_COMPUTE_ERROR_ON(input0->info()->dimension(2) != input1->info()->dimension(1));

    _input0 = input0;
    _input1 = input1;
    _output = output;

    // Set appropriate function to run
    switch(input0->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = &NEGEMMMatrixVectorMultiplyKernel::matrix_vector_multiply<uint8_t, uint8_t, int32_t>;
            break;
        case DataType::F32:
            _func = &NEGEMMMatrixVectorMultiplyKernel::matrix_vector_multiply<float, float, float>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }

    // Configure kernel window
    const unsigned int num_elems_read_per_iteration = 16 / _input0->info()->element_size();

    const unsigned int border_x = ceil_to_multiple(input0->info()->dimension(0), num_elems_read_per_iteration) - input0->info()->dimension(0);
    _border_size                = BorderSize(0, border_x);

    Window win = calculate_max_window(*input0->info(), Steps(num_elems_read_per_iteration));

    AccessWindowHorizontal input0_access(input0->info(), 0, num_elems_read_per_iteration);
    AccessWindowHorizontal input1_access(input1->info(), 0, num_elems_read_per_iteration);
    AccessWindowStatic     output_access(output->info(), 0, 0, output->info()->dimension(0), output->info()->dimension(1));

    update_window_and_padding(win, input0_access, input1_access, output_access);

    _output->info()->set_valid_region(ValidRegion(Coordinates(), _output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGEMMMatrixVectorMultiplyKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window window_slice = window.first_slice_window_3D();

    Window window_in(window);
    Window window_weights(window_slice);
    Window window_out(window);

    // Setup input0 slice
    window_in.set(Window::DimX, Window::Dimension(0, _input0->info()->dimension(0), _input0->info()->dimension(0)));
    window_in.set(Window::DimY, Window::Dimension(0, _input0->info()->dimension(1), 1));
    window_in.set(Window::DimZ, Window::Dimension(0, _input0->info()->dimension(2), 1));

    // Setup input1 and output slice. Their dimensions are increased in the kernel.
    window_weights.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_weights.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_weights.set(Window::DimZ, Window::Dimension(0, 0, 0));

    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    if(_func != nullptr)
    {
        (this->*_func)(window_in, window_weights, window_out);
    }
}
