/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEThresholdKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ThresholdKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);

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
    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, *input->clone());

    // NEThresholdKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

    return std::make_pair(Status{}, win);
}
} // namespace

NEThresholdKernel::NEThresholdKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _info()
{
}

void NEThresholdKernel::configure(const ITensor *input, ITensor *output, const ThresholdKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), info));

    _input  = input;
    _output = output;
    _info   = info;

    switch(_info.type)
    {
        case ThresholdType::BINARY:
            _func = &NEThresholdKernel::run_binary;
            break;
        case ThresholdType::RANGE:
            _func = &NEThresholdKernel::run_range;
            break;
        default:
            ARM_COMPUTE_ERROR("Thresholding type not recognized");
            break;
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICPPKernel::configure(win_config.second);
}

Status NEThresholdKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ThresholdKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

    return Status{};
}

inline void NEThresholdKernel::run_binary(const Window &window)
{
    /** Neon vector tag type. */
    using Type         = uint8_t;
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<Type, wrapper::traits::BitWidth::W128>;

    const int  window_step_x  = 16 / sizeof(Type);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    const uint8_t threshold   = _info.threshold;
    const uint8_t true_value  = _info.true_value;
    const uint8_t false_value = _info.false_value;

    const auto vthreshold   = wrapper::vdup_n(threshold, ExactTagType{});
    const auto vtrue_value  = wrapper::vdup_n(true_value, ExactTagType{});
    const auto vfalse_value = wrapper::vdup_n(false_value, ExactTagType{});

    Iterator input(_input, win_collapsed);
    Iterator output(_output, win_collapsed);

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const Type *>(input.ptr());
        const auto output_ptr = reinterpret_cast<Type *>(output.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vdata = wrapper::vloadq(input_ptr + x);
            const auto vmask = wrapper::vcgt(vdata, vthreshold);
            wrapper::vstore(output_ptr + x, wrapper::vbsl(vmask, vtrue_value, vfalse_value));
        }

        for(; x < window_end_x; ++x)
        {
            const Type data   = *(reinterpret_cast<const Type *>(input_ptr + x));
            *(output_ptr + x) = (data > threshold) ? true_value : false_value;
        }
    },
    input, output);
}

inline void NEThresholdKernel::run_range(const Window &window)
{
    /** Neon vector tag type. */
    using Type         = uint8_t;
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<Type, wrapper::traits::BitWidth::W128>;

    const int  window_step_x  = 16 / sizeof(Type);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    const uint8_t lower_threshold = _info.threshold;
    const uint8_t upper_threshold = _info.upper;
    const uint8_t true_value      = _info.true_value;
    const uint8_t false_value     = _info.false_value;

    const auto vlower_threshold = wrapper::vdup_n(lower_threshold, ExactTagType{});
    const auto vupper_threshold = wrapper::vdup_n(upper_threshold, ExactTagType{});
    const auto vtrue_value      = wrapper::vdup_n(true_value, ExactTagType{});
    const auto vfalse_value     = wrapper::vdup_n(false_value, ExactTagType{});

    Iterator input(_input, win_collapsed);
    Iterator output(_output, win_collapsed);

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const Type *>(input.ptr());
        const auto output_ptr = reinterpret_cast<Type *>(output.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vdata = wrapper::vloadq(input_ptr + x);
            auto       vmask = wrapper::vcle(vdata, vupper_threshold);
            vmask            = wrapper::vand(wrapper::vcge(vdata, vlower_threshold), vmask);
            wrapper::vstore(output_ptr + x, wrapper::vbsl(vmask, vtrue_value, vfalse_value));
        }

        for(; x < window_end_x; ++x)
        {
            const Type data   = *(reinterpret_cast<const Type *>(input_ptr + x));
            *(output_ptr + x) = (data <= upper_threshold && data >= lower_threshold) ? true_value : false_value;
        }
    },
    input, output);
}

void NEThresholdKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
} // namespace arm_compute
