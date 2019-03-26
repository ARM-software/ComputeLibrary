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
#include "arm_compute/core/NEON/kernels/NEFFTScaleKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
void scale_complex(float *c_in, float *c_out, bool is_conjugate, float scale)
{
    const auto a = wrapper::vload(c_in);
    auto       b = wrapper::vdiv(a, float32x2_t{ scale, scale });
    if(is_conjugate)
    {
        const float img_part = wrapper::vgetlane(b, 1);
        b                    = wrapper::vsetlane(-img_part, b, 1);
    }

    wrapper::vstore(c_out, b);
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 2, DataType::F32);

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_channels() != 1 && output->num_channels() != 2);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());

    if(output != nullptr)
    {
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output, *input->clone());

        // NEFFTScaleKernel doesn't need padding so update_window_and_padding() can be skipped
        Coordinates coord;
        coord.set_num_dimensions(output->num_dimensions());
        output->set_valid_region(ValidRegion(coord, output->tensor_shape()));
    }

    return std::make_pair(Status{}, win);
}
} // namespace

NEFFTScaleKernel::NEFFTScaleKernel()
    : _input(nullptr), _output(nullptr), _scale(), _run_in_place(false), _is_conj(false)
{
}

void NEFFTScaleKernel::configure(ITensor *input, ITensor *output, const FFTScaleKernelInfo &config)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (output != nullptr) ? output->info() : nullptr));

    _input        = input;
    _output       = output;
    _run_in_place = (output == nullptr) || (output == input);
    _is_conj      = config.conjugate;
    _scale        = config.scale;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), _run_in_place ? nullptr : output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEFFTScaleKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const FFTScaleKernelInfo &config)
{
    ARM_COMPUTE_UNUSED(config);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEFFTScaleKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_UNUSED(info);

    Window input_window = window;
    input_window.set(Window::DimX, 0);

    Iterator in(_input, input_window);
    Iterator out(_run_in_place ? _input : _output, input_window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        scale_complex(reinterpret_cast<float *>(in.ptr()), reinterpret_cast<float *>(out.ptr()), _is_conj, _scale);
    },
    in, out);
}
} // namespace arm_compute
