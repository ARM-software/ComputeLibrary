/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLHOGGradient.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLHOGGradient::CLHOGGradient(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _derivative(), _mag_phase(), _gx(), _gy()
{
}

void CLHOGGradient::configure(ICLTensor *input, ICLTensor *output_magnitude, ICLTensor *output_phase, PhaseType phase_type, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_magnitude, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_phase, 1, DataType::U8);

    const TensorShape &shape_img = input->info()->tensor_shape();

    // Allocate image memory
    TensorInfo info(shape_img, Format::S16);
    _gx.allocator()->init(info);
    _gy.allocator()->init(info);

    // Manage intermediate buffers
    _memory_group.manage(&_gx);
    _memory_group.manage(&_gy);

    // Initialise derivate kernel
    _derivative.configure(input, &_gx, &_gy, border_mode, constant_border_value);

    // Initialise magnitude/phase kernel
    if(PhaseType::UNSIGNED == phase_type)
    {
        _mag_phase.configure(&_gx, &_gy, output_magnitude, output_phase, MagnitudeType::L2NORM, PhaseType::UNSIGNED);
    }
    else
    {
        _mag_phase.configure(&_gx, &_gy, output_magnitude, output_phase, MagnitudeType::L2NORM, PhaseType::SIGNED);
    }

    // Allocate intermediate tensors
    _gx.allocator()->allocate();
    _gy.allocator()->allocate();
}

void CLHOGGradient::run()
{
    _memory_group.acquire();

    // Run derivative
    _derivative.run();

    // Run magnitude/phase kernel
    CLScheduler::get().enqueue(_mag_phase);

    _memory_group.release();
}