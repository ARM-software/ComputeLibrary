/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEL2NormalizeLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include <arm_neon.h>
#include <cmath>

namespace arm_compute
{
namespace
{
constexpr int max_input_tensor_dim = 3;

template <typename T, int S>
void l2_normalize_X(const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window)
{
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    const int  window_step_x  = 16 / data_size_from_type(in->info()->data_type());
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input_it(in, win_collapsed);
    Iterator sum_it(sum, win_collapsed);
    Iterator output_it(out, win_collapsed);

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto in_ptr  = reinterpret_cast<const T *>(input_it.ptr());
        const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());

        const T    sum_value      = *reinterpret_cast<const T *>(sum_it.ptr());
        const T    norm_value     = static_cast<T>(1.f) / std::sqrt(std::max(sum_value, static_cast<T>(epsilon)));
        const auto vec_norm_value = wrapper::vdup_n(norm_value, ExactTagType{});

        // Compute elements over vector steps
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            wrapper::vstore(out_ptr + x, wrapper::vmul(wrapper::vloadq(in_ptr + x), vec_norm_value));
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            out_ptr[x] = in_ptr[x] * norm_value;
        }
    },
    input_it, sum_it, output_it);
}

template <typename T, int S>
void l2_normalize_YZ(const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window, size_t axis)
{
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    const int  window_step_x  = 16 / data_size_from_type(in->info()->data_type());
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Window window_sum(win);
    window_sum.set(axis, Window::Dimension(0, 0, 0));

    Iterator input_it(in, win);
    Iterator sum_it(sum, window_sum);
    Iterator output_it(out, win);

    const auto vec_eps = wrapper::vdup_n(static_cast<T>(epsilon), ExactTagType{});

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto in_ptr  = reinterpret_cast<const T *>(input_it.ptr());
        const auto sum_ptr = reinterpret_cast<const T *>(sum_it.ptr());
        const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());

        // Compute elements over vector steps
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vec_norm_value = wrapper::vinvsqrt(wrapper::vmax(wrapper::vloadq(sum_ptr + x), vec_eps));
            wrapper::vstore(out_ptr + x, wrapper::vmul(wrapper::vloadq(in_ptr + x), vec_norm_value));
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            const T norm_value = static_cast<T>(1.f) / std::sqrt(std::max(sum_ptr[x], static_cast<T>(epsilon)));
            out_ptr[x]         = in_ptr[x] * norm_value;
        }
    },
    input_it, sum_it, output_it);
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, int axis, float epsilon)
{
    ARM_COMPUTE_UNUSED(epsilon);

    const uint32_t actual_axis = wrap_around(axis, max_input_tensor_dim);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, sum);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(actual_axis > 2, "Actual axis greater than 2 is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(actual_axis >= TensorShape::num_max_dimensions, "Actual normalization axis greater than max number of dimensions");

    // Reduce shape on axis
    TensorShape sum_shape = input->tensor_shape();
    sum_shape.set(actual_axis, 1);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(sum->tensor_shape(), sum_shape);

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(input->tensor_shape(), output->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    Window win = calculate_max_window(*input, Steps());

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, input->data_type());

    // NEL2NormalizeLayerKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

    return std::make_tuple(Status{}, win);
}
} // namespace

NEL2NormalizeLayerKernel::NEL2NormalizeLayerKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr), _actual_axis(0), _epsilon(1e-12)
{
}

void NEL2NormalizeLayerKernel::configure(const ITensor *input, const ITensor *sum, ITensor *output, int axis, float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), sum->info(), output->info(), axis, epsilon));

    _input       = input;
    _sum         = sum;
    _output      = output;
    _actual_axis = wrap_around(axis, max_input_tensor_dim);
    _epsilon     = epsilon;

    // Configure kernel window
    auto win_config = validate_and_configure_window(_input->info(), _output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEL2NormalizeLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, int axis, float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, sum, output, axis, epsilon));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get())));

    return Status{};
}

void NEL2NormalizeLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_actual_axis > 2)
    {
        ARM_COMPUTE_ERROR("Unsupported normalization axis");
    }

    switch(_input->info()->data_type())
    {
        case DataType::F32:
            (_actual_axis == Window::DimX) ? l2_normalize_X<float, 4>(_input, _sum, _output, _epsilon, window) : l2_normalize_YZ<float, 4>(_input, _sum, _output, _epsilon, window, _actual_axis);
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            (_actual_axis == Window::DimX) ? l2_normalize_X<float16_t, 8>(_input, _sum, _output, _epsilon, window) : l2_normalize_YZ<float16_t, 8>(_input, _sum, _output, _epsilon, window, _actual_axis);
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}
} // namespace arm_compute
