/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEHOGGradient.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/kernels/NEMagnitudePhaseKernel.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

using namespace arm_compute;

NEHOGGradient::NEHOGGradient()
    : _derivative(), _mag_phase(nullptr), _gx(), _gy()
{
}

void NEHOGGradient::configure(ITensor *input, ITensor *output_magnitude, ITensor *output_phase, PhaseType phase_type, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_magnitude, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_phase, 1, DataType::U8);

    const TensorShape &shape_img = input->info()->tensor_shape();

    // Allocate image memory
    TensorInfo info(shape_img, Format::S16);
    info.auto_padding();
    _gx.allocator()->init(info);
    _gx.allocator()->allocate();
    _gy.allocator()->init(info);
    _gy.allocator()->allocate();

    // Initialise derivate kernel
    _derivative.configure(input, &_gx, &_gy, border_mode, constant_border_value);

    // Initialise magnitude/phase kernel
    if(PhaseType::UNSIGNED == phase_type)
    {
        auto k = arm_compute::cpp14::make_unique<NEMagnitudePhaseKernel<MagnitudeType::L2NORM, PhaseType::UNSIGNED>>();
        k->configure(&_gx, &_gy, output_magnitude, output_phase);
        _mag_phase = std::move(k);
    }
    else
    {
        auto k = arm_compute::cpp14::make_unique<NEMagnitudePhaseKernel<MagnitudeType::L2NORM, PhaseType::SIGNED>>();
        k->configure(&_gx, &_gy, output_magnitude, output_phase);
        _mag_phase = std::move(k);
    }
}

void NEHOGGradient::run()
{
    // Run derivative
    _derivative.run();

    // Run magnitude/phase kernel
    NEScheduler::get().multithread(_mag_phase.get());
}
