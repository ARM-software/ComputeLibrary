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
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixVectorMultiplyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
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

namespace
{
Status validate_arguments(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(output, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(input0->data_type()) && (output->data_type() != DataType::S32));
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_float(input0->data_type()) && (output->data_type() != input0->data_type()));

    ARM_COMPUTE_RETURN_ERROR_ON(input0->num_dimensions() == input1->num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(2) != input1->dimension(1));
    ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(DataLayoutDimension::HEIGHT) != output->dimension(DataLayoutDimension::HEIGHT));
    ARM_COMPUTE_RETURN_ERROR_ON(input1->dimension(DataLayoutDimension::WIDTH) != output->dimension(DataLayoutDimension::WIDTH));

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input0, ITensorInfo *input1, ITensorInfo *output)
{
    const unsigned int num_elems_read_per_iteration = 16 / input0->element_size();

    Window win = calculate_max_window(*input0, Steps(num_elems_read_per_iteration));

    AccessWindowHorizontal input0_access(input0, 0, num_elems_read_per_iteration);
    AccessWindowHorizontal input1_access(input1, 0, num_elems_read_per_iteration);
    AccessWindowStatic     output_access(output, 0, 0, output->dimension(0), output->dimension(1));

    bool window_changed = update_window_and_padding(win, input0_access, input1_access, output_access);

    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

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
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
void NEGEMMMatrixVectorMultiplyKernel::matrix_vector_multiply<half, half, half>(const Window &window_in,
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
        auto                 output_ptr  = reinterpret_cast<__fp16 *>(out.ptr() + (id.y() + id.z() * input_h) * output_stride_x);

        float16x8_t row_dot = vdupq_n_f16(0.f);
        for(int i = 0; i < input_w; i += 8)
        {
            const auto input   = vld1q_f16(reinterpret_cast<const __fp16 *>(input_ptr + i * input_stride_x));
            const auto weights = vld1q_f16(reinterpret_cast<const __fp16 *>(weights_ptr + i * weights_stride_x));
            row_dot            = vaddq_f16(row_dot, vmulq_f16(input, weights));
        }

        auto temp = vadd_f16(vget_high_f16(row_dot), vget_low_f16(row_dot));
        temp      = vpadd_f16(temp, temp);
        temp      = vpadd_f16(temp, temp);

        *output_ptr = vget_lane_f16(temp, 0);
    },
    in, in2, out);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

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
    ARM_COMPUTE_ERROR_ON_NULLPTR(input0, input1, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input0->info(), input1->info(), output->info()));

    _input0 = input0;
    _input1 = input1;
    _output = output;

    // Set appropriate function to run
    switch(input0->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = &NEGEMMMatrixVectorMultiplyKernel::matrix_vector_multiply<uint8_t, uint8_t, int32_t>;
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &NEGEMMMatrixVectorMultiplyKernel::matrix_vector_multiply<half, half, half>;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
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

    auto win_config = validate_and_configure_window(input0->info(), input1->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEGEMMMatrixVectorMultiplyKernel::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input0, input1, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input0, input1, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input0->clone().get(), input1->clone().get(), output->clone().get()).first);
    return Status{};
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
