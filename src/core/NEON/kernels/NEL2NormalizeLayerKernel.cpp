/*
 * Copyright (c) 2017-2019 ARM Limited.
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
template <typename T, int S>
void l2_normalize_X(const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window)
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    Window window_sum(window);
    window_sum.set(Window::DimX, Window::Dimension(0, 0, 0));

    Window in_slice  = window.first_slice_window_1D();
    Window sum_slice = window_sum.first_slice_window_1D();

    do
    {
        Iterator input_it(in, in_slice);
        Iterator sum_it(sum, sum_slice);
        Iterator output_it(out, in_slice);

        const auto sum_value           = *reinterpret_cast<const T *>(sum_it.ptr());
        const auto vec_normalize_value = wrapper::vdup_n(static_cast<T>(1.f / std::sqrt(std::max(sum_value, static_cast<T>(epsilon)))), ExactTagType{});

        execute_window_loop(in_slice, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const T *>(input_it.ptr());
            const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());

            wrapper::vstore(out_ptr, wrapper::vmul(wrapper::vloadq(in_ptr), vec_normalize_value));
        },
        input_it, output_it);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(sum_slice));
}

template <typename T, int S>
void l2_normalize_Y(const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window)
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    Window window_sum(window);
    window_sum.set(Window::DimY, Window::Dimension(0, 0, 0));

    Window in_slice  = window.first_slice_window_2D();
    Window sum_slice = window_sum.first_slice_window_2D();

    do
    {
        Iterator input_it(in, in_slice);
        Iterator sum_it(sum, sum_slice);
        Iterator output_it(out, in_slice);

        auto eps = wrapper::vdup_n(static_cast<T>(epsilon), ExactTagType{});

        execute_window_loop(in_slice, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const T *>(input_it.ptr());
            const auto sum_ptr = reinterpret_cast<const T *>(sum_it.ptr());
            const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());

            const auto vec_normalize_value = wrapper::vinvsqrt(wrapper::vmax(wrapper::vloadq(sum_ptr), eps));
            wrapper::vstore(out_ptr, wrapper::vmul(wrapper::vloadq(in_ptr), vec_normalize_value));
        },
        input_it, sum_it, output_it);
    }
    while(window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(sum_slice));
}

template <typename T, int S>
void l2_normalize_Z(const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window)
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    Window window_sum(window);
    window_sum.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Window in_slice  = window.first_slice_window_3D();
    Window sum_slice = window_sum.first_slice_window_3D();

    do
    {
        Iterator input_it(in, in_slice);
        Iterator sum_it(sum, sum_slice);
        Iterator output_it(out, in_slice);

        auto eps = wrapper::vdup_n(static_cast<T>(epsilon), ExactTagType{});

        execute_window_loop(in_slice, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const T *>(input_it.ptr());
            const auto sum_ptr = reinterpret_cast<const T *>(sum_it.ptr());
            const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());

            const auto vec_normalize_value = wrapper::vinvsqrt(wrapper::vmax(wrapper::vloadq(sum_ptr), eps));
            wrapper::vstore(out_ptr, wrapper::vmul(wrapper::vloadq(in_ptr), vec_normalize_value));
        },
        input_it, sum_it, output_it);
    }
    while(window.slide_window_slice_3D(in_slice) && window.slide_window_slice_3D(sum_slice));
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, unsigned int axis, float epsilon)
{
    ARM_COMPUTE_UNUSED(epsilon);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, sum);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 2, "Axis greater than 2 is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Normalization axis greater than max number of dimensions");

    // Reduce shape on axis
    TensorShape sum_shape = input->tensor_shape();
    sum_shape.set(axis, 1);
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

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *sum, ITensorInfo *output, unsigned int axis)
{
    const unsigned int num_elems_processed_per_iteration     = 16 / data_size_from_type(input->data_type());
    const unsigned int num_elems_processed_per_iteration_sum = (axis == 0) ? 1 : num_elems_processed_per_iteration;

    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, input->data_type());

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal sum_access(sum, 0, num_elems_processed_per_iteration_sum);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, sum_access, output_access);
    output_access.set_valid_region(win, input->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};

    return std::make_tuple(err, win);
}
} // namespace

NEL2NormalizeLayerKernel::NEL2NormalizeLayerKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr), _axis(0), _epsilon(1e-12)
{
}

void NEL2NormalizeLayerKernel::configure(const ITensor *input, const ITensor *sum, ITensor *output, unsigned int axis, float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), sum->info(), output->info(), axis, epsilon));

    _input   = input;
    _sum     = sum;
    _output  = output;
    _axis    = axis;
    _epsilon = epsilon;

    // Configure kernel window
    auto win_config = validate_and_configure_window(_input->info(), _sum->info(), _output->info(), axis);
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEL2NormalizeLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, unsigned int axis, float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, sum, output, axis, epsilon));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), sum->clone().get(), output->clone().get(), axis)));

    return Status{};
}

void NEL2NormalizeLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_axis)
    {
        case 0:
            switch(_input->info()->data_type())
            {
                case DataType::F32:
                    l2_normalize_X<float, 4>(_input, _sum, _output, _epsilon, window);
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    l2_normalize_X<float16_t, 8>(_input, _sum, _output, _epsilon, window);
                    break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                default:
                    ARM_COMPUTE_ERROR("Not implemented");
            }
            break;
        case 1:
            switch(_input->info()->data_type())
            {
                case DataType::F32:
                    l2_normalize_Y<float, 4>(_input, _sum, _output, _epsilon, window);
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    l2_normalize_Y<float16_t, 8>(_input, _sum, _output, _epsilon, window);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not implemented");
            }
            break;
        case 2:
            switch(_input->info()->data_type())
            {
                case DataType::F32:
                    l2_normalize_Z<float, 4>(_input, _sum, _output, _epsilon, window);
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    l2_normalize_Z<float16_t, 8>(_input, _sum, _output, _epsilon, window);
                    break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                default:
                    ARM_COMPUTE_ERROR("Not implemented");
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported normalization axis");
    }
}
} // namespace arm_compute
