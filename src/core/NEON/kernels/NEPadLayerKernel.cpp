/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEPadLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &paddings, const PaddingMode mode)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(mode != PaddingMode::CONSTANT, "Only constant padding mode is supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(paddings.size() > 4, "Padding list bigger than 4 dimensions");
    if(output->total_size() != 0)
    {
        const TensorShape expected_output_shape = arm_compute::misc::shape_calculator::compute_padded_shape(input->tensor_shape(), paddings);
        const TensorInfo  expected_output_info  = input->clone()->set_tensor_shape(expected_output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &expected_output_info);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }
    return Status{};
}
} // namespace

template <typename T>
void NEPadLayerKernel::run_pad_constant(const Window &window)
{
    Window output_window{ window };
    output_window.set(Window::DimX, Window::Dimension(0, 1, 1));

    const size_t element_size = _input->info()->element_size();
    Iterator     output_it(_output, output_window);
    execute_window_loop(output_window, [&](const Coordinates & id)
    {
        Coordinates idin{ id };
        for(size_t dim = _padding.size() - 1; dim > 0; --dim)
        {
            idin[dim] -= _padding[dim].first;
            if(idin[dim] < 0 || static_cast<int>(_input->info()->dimension(dim)) - 1 < idin[dim])
            {
                std::fill_n(reinterpret_cast<T *>(output_it.ptr()), _output->info()->dimension(0), _constant_value.get<T>());
                return;
            }
        }
        T *input_it_ptr  = reinterpret_cast<T *>(_input->ptr_to_element(idin));
        T *output_it_ptr = reinterpret_cast<T *>(output_it.ptr());
        std::fill_n(output_it_ptr, _padding[0].first, _constant_value.get<T>());
        memcpy(output_it_ptr + _padding[0].first, input_it_ptr, _input->info()->dimension(0) * element_size);
        std::fill_n(output_it_ptr + _padding[0].first + _input->info()->dimension(0), _padding[0].second, _constant_value.get<T>());
    },
    output_it);
}

void NEPadLayerKernel::run_pad_constant_uint8_3Dinput_3Dpad(const Window &window)
{
    ARM_COMPUTE_UNUSED(window);

    const size_t start_plane = window.z().start();
    const size_t end_plane   = window.z().end();

    size_t start_plane_input = start_plane;
    if(_padding.size() > 2)
    {
        start_plane_input = (start_plane < _padding[2].first) ? 0 : start_plane - _padding[2].first;
    }
    const int output_plane_size = _output->info()->dimension(0) * _output->info()->dimension(1);
    const int input_plane_size  = _input->info()->dimension(0) * _input->info()->dimension(1);

    const int pad_y_elems_top = (_padding.size() > 1 ? _padding[1].first : 0) * _output->info()->dimension(0);
    const int pad_y_elems_bot = (_padding.size() > 1 ? _padding[1].second : 0) * _output->info()->dimension(0);

    const size_t jump_to_next_row_input  = _input->info()->dimension(0);
    const size_t jump_to_next_row_output = _padding[0].first + _padding[0].second;

    uint8_t       *output_row_ptr = _output->buffer() + _output->info()->offset_first_element_in_bytes() + start_plane * output_plane_size;
    const uint8_t *input_it_ptr   = _input->buffer() + _input->info()->offset_first_element_in_bytes() + start_plane_input * input_plane_size;
    const auto     pad_value      = _constant_value.get<uint8_t>();

    for(size_t z_i = start_plane; z_i < end_plane; ++z_i)
    {
        if(_padding.size() > 2 && z_i < _padding[2].first)
        {
            memset(output_row_ptr, pad_value, output_plane_size);
            output_row_ptr += output_plane_size;
        }
        else if(_padding.size() > 2 && z_i > (_input->info()->dimension(2) + _padding[2].first - 1))
        {
            memset(output_row_ptr, pad_value, output_plane_size);
            output_row_ptr += output_plane_size;
        }
        else
        {
            memset(output_row_ptr, pad_value, pad_y_elems_top);
            output_row_ptr += pad_y_elems_top;
            size_t y_i = _input->info()->dimension(1);
            // Basic loop unrolling
            for(; y_i > 3; y_i -= 4)
            {
                memset(output_row_ptr, pad_value, _padding[0].first);
                output_row_ptr += _padding[0].first;

                memcpy(output_row_ptr, input_it_ptr, _input->info()->dimension(0));
                output_row_ptr += _input->info()->dimension(0);
                input_it_ptr += jump_to_next_row_input;

                memset(output_row_ptr, pad_value, _padding[0].second + _padding[0].first);
                output_row_ptr += jump_to_next_row_output;

                memcpy(output_row_ptr, input_it_ptr, _input->info()->dimension(0));
                output_row_ptr += _input->info()->dimension(0);
                input_it_ptr += jump_to_next_row_input;

                memset(output_row_ptr, pad_value, _padding[0].second + _padding[0].first);
                output_row_ptr += jump_to_next_row_output;

                memcpy(output_row_ptr, input_it_ptr, _input->info()->dimension(0));
                output_row_ptr += _input->info()->dimension(0);
                input_it_ptr += jump_to_next_row_input;

                memset(output_row_ptr, pad_value, _padding[0].second + _padding[0].first);
                output_row_ptr += jump_to_next_row_output;

                memcpy(output_row_ptr, input_it_ptr, _input->info()->dimension(0));
                output_row_ptr += _input->info()->dimension(0);
                input_it_ptr += jump_to_next_row_input;

                memset(output_row_ptr, pad_value, _padding[0].second);
                output_row_ptr += _padding[0].second;
            }
            for(; y_i > 0; --y_i)
            {
                memset(output_row_ptr, pad_value, _padding[0].first);
                output_row_ptr += _padding[0].first;

                memcpy(output_row_ptr, input_it_ptr, _input->info()->dimension(0));
                output_row_ptr += _input->info()->dimension(0);
                input_it_ptr += _input->info()->dimension(0);

                memset(output_row_ptr, pad_value, _padding[0].second);
                output_row_ptr += _padding[0].second;
            }
            memset(output_row_ptr, pad_value, pad_y_elems_bot);
            output_row_ptr += pad_y_elems_bot;
        }
    }
}

NEPadLayerKernel::NEPadLayerKernel()
    : _func(), _input(nullptr), _output(nullptr), _padding(), _constant_value(), _mode()
{
}

void NEPadLayerKernel::configure(ITensor *input, ITensor *output, const PaddingList &padding, const PixelValue constant_value, const PaddingMode mode)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    // Auto-init
    const TensorShape expected_output_shape = arm_compute::misc::shape_calculator::compute_padded_shape(input->info()->tensor_shape(), padding);
    const TensorInfo  expected_output_info  = input->info()->clone()->set_tensor_shape(expected_output_shape);
    auto_init_if_empty(*output->info(), expected_output_info);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), padding, mode));

    _input          = input;
    _output         = output;
    _padding        = padding;
    _constant_value = constant_value;
    _mode           = mode;

    if(_mode == PaddingMode::CONSTANT)
    {
        switch(_input->info()->element_size())
        {
            case 1:
                if(_input->info()->num_dimensions() == 3 &&                           // Is 3D
                   padding.size() <= 3 &&                                             // Has 3D padding
                   !_input->info()->has_padding() && !_output->info()->has_padding()) // Input & Output have no padding
                {
                    _func = &NEPadLayerKernel::run_pad_constant_uint8_3Dinput_3Dpad;
                }
                else
                {
                    _func = &NEPadLayerKernel::run_pad_constant<uint8_t>;
                }
                break;
            case 2:
                _func = &NEPadLayerKernel::run_pad_constant<uint16_t>;
                break;
            case 4:
                _func = &NEPadLayerKernel::run_pad_constant<uint32_t>;
                break;
            default:
                ARM_COMPUTE_ERROR("Element size not supported");
                break;
        }
    }
    else
    {
        ARM_COMPUTE_ERROR("Padding mode not supported");
    }

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());

    // The NEPad doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    ICPPKernel::configure(win);
}

Status NEPadLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, const PixelValue constant_value, const PaddingMode mode)
{
    ARM_COMPUTE_UNUSED(constant_value);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, padding, mode));
    return Status{};
}

void NEPadLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_func != nullptr)
    {
        (this->*_func)(window);
    }
}
} // namespace arm_compute
