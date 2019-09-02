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
#include "arm_compute/core/NEON/kernels/NEMeanStdDevNormalizationKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 2, "Input tensor cannot have more than 2 dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }
    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    if(output != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output, *input);
    }

    // This kernel doesn't need padding. A left-over for loop on dimension X, we cannot have any read or write out of memory
    // For this reason num_elems_processed_per_iteration is set to 1
    Window win = calculate_max_window(*input, Steps());
    if(output != nullptr)
    {
        output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));
    }

    return std::make_pair(Status{}, win);
}
} // namespace

template <typename ScalarType, int size>
void NEMeanStdDevNormalizationKernel::mean_stddev_normalization(const Window &window)
{
    using ExactTagType = typename wrapper::traits::neon_vector<ScalarType, size>::tag_type;

    // Set build options
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x  = size;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Iterator input(_input, win);
    Iterator output(_output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        int  x       = window_start_x;
        auto in_ptr  = reinterpret_cast<const ScalarType *>(input.ptr());
        auto out_ptr = reinterpret_cast<ScalarType *>(output.ptr());

        auto sum_vec    = wrapper::vdup_n(static_cast<ScalarType>(0.f), ExactTagType{});
        auto sum_sq_vec = wrapper::vdup_n(static_cast<ScalarType>(0.f), ExactTagType{});

        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            auto data  = wrapper::vloadq(in_ptr + x);
            sum_vec    = wrapper::vadd(sum_vec, data);
            sum_sq_vec = wrapper::vadd(sum_sq_vec, wrapper::vmul(data, data));
        }

        auto sum_carry_res    = wrapper::vpadd(wrapper::vgethigh(sum_vec), wrapper::vgetlow(sum_vec));
        auto sum_sq_carry_res = wrapper::vpadd(wrapper::vgethigh(sum_sq_vec), wrapper::vgetlow(sum_sq_vec));
        for(int i = 0; i < size / 4; ++i)
        {
            sum_carry_res    = wrapper::vpadd(sum_carry_res, sum_carry_res);
            sum_sq_carry_res = wrapper::vpadd(sum_sq_carry_res, sum_sq_carry_res);
        }

        auto sum    = wrapper::vgetlane(sum_carry_res, 0);
        auto sum_sq = wrapper::vgetlane(sum_sq_carry_res, 0);

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            ScalarType data = *(in_ptr + x);
            sum += data;
            sum_sq += data * data;
        }

        ScalarType mean       = sum / _input->info()->dimension(0);
        ScalarType var        = (sum_sq / _input->info()->dimension(0)) - (mean * mean);
        ScalarType stddev_inv = 1.f / sqrt(var + _epsilon);

        auto mean_vec       = wrapper::vdup_n(mean, ExactTagType{});
        auto stddev_inv_vec = wrapper::vdup_n(stddev_inv, ExactTagType{});
        for(x = window_start_x; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            auto data = wrapper::vloadq(in_ptr + x);
            auto res  = wrapper::vmul(wrapper::vsub(data, mean_vec), stddev_inv_vec);
            // Store results
            wrapper::vstore(out_ptr + x, res);
        }
        for(; x < window_end_x; ++x)
        {
            *(out_ptr + x) = (*(in_ptr + x) - mean) * stddev_inv;
        }
    },
    input, output);
}

NEMeanStdDevNormalizationKernel::NEMeanStdDevNormalizationKernel()
    : _input(nullptr), _output(nullptr), _epsilon(1e-8f), _func(nullptr)
{
}

void NEMeanStdDevNormalizationKernel::configure(ITensor *input, ITensor *output, float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    ARM_COMPUTE_ERROR_THROW_ON(NEMeanStdDevNormalizationKernel::validate(input->info(), (output != nullptr) ? output->info() : nullptr, epsilon));

    _input   = input;
    _output  = (output == nullptr) ? input : output;
    _epsilon = epsilon;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (output == nullptr) ? nullptr : output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICPPKernel::configure(win_config.second);

    // Configure function to run based on different data types
    const DataType data_type = input->info()->data_type();
    switch(data_type)
    {
        case DataType::F32:
            _func = &NEMeanStdDevNormalizationKernel::mean_stddev_normalization<float, 4>;
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &NEMeanStdDevNormalizationKernel::mean_stddev_normalization<float16_t, 8>;
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        default:
            ARM_COMPUTE_ERROR("Not Supported");
            break;
    }
}

Status NEMeanStdDevNormalizationKernel::validate(const ITensorInfo *input, const ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, epsilon));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (output != nullptr) ? output->clone().get() : nullptr).first);
    return Status{};
}

void NEMeanStdDevNormalizationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
} // namespace arm_compute
