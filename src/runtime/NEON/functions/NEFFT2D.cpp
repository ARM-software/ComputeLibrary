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
#include "arm_compute/runtime/NEON/functions/NEFFT2D.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/Scheduler.h"

namespace arm_compute
{
NEFFT2D::NEFFT2D(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _first_pass_func(memory_manager), _second_pass_func(memory_manager), _first_pass_tensor()
{
}

void NEFFT2D::configure(const ITensor *input, ITensor *output, const FFT2DInfo &config)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEFFT2D::validate(input->info(), output->info(), config));

    // Setup first pass
    FFT1DInfo first_pass_config;
    first_pass_config.axis      = config.axes.first;
    first_pass_config.direction = config.direction;
    _memory_group.manage(&_first_pass_tensor);
    _first_pass_func.configure(input, &_first_pass_tensor, first_pass_config);

    // Setup second pass
    FFT1DInfo second_pass_config;
    second_pass_config.axis      = config.axes.second;
    second_pass_config.direction = config.direction;
    _second_pass_func.configure(&_first_pass_tensor, output, second_pass_config);
    _first_pass_tensor.allocator()->allocate();
}

Status NEFFT2D::validate(const ITensorInfo *input, const ITensorInfo *output, const FFT2DInfo &config)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    // Create intermediate tensor info
    TensorInfo first_pass_tensor(input->clone()->set_is_resizable(true).reset_padding().set_num_channels(2));

    // Validate first pass
    FFT1DInfo first_pass_config;
    first_pass_config.axis      = config.axes.first;
    first_pass_config.direction = config.direction;
    ARM_COMPUTE_RETURN_ON_ERROR(NEFFT1D::validate(input, &first_pass_tensor, first_pass_config));

    // Validate second pass
    FFT1DInfo second_pass_config;
    second_pass_config.axis      = config.axes.second;
    second_pass_config.direction = config.direction;
    ARM_COMPUTE_RETURN_ON_ERROR(NEFFT1D::validate(&first_pass_tensor, output, second_pass_config));

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

void NEFFT2D::run()
{
    _memory_group.acquire();

    _first_pass_func.run();
    _second_pass_func.run();

    _memory_group.release();
}
} // namespace arm_compute
