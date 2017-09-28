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
#include "arm_compute/runtime/NEON/functions/NEWarpAffine.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/NEON/kernels/NEWarpKernel.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <utility>

using namespace arm_compute;

void NEWarpAffine::configure(ITensor *input, ITensor *output, const float *matrix, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == matrix);

    switch(policy)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
        {
            auto k = arm_compute::support::cpp14::make_unique<NEWarpAffineKernel<InterpolationPolicy::NEAREST_NEIGHBOR>>();
            k->configure(input, output, matrix, border_mode, constant_border_value);
            _kernel = std::move(k);
            break;
        }
        case InterpolationPolicy::BILINEAR:
        {
            auto k = arm_compute::support::cpp14::make_unique<NEWarpAffineKernel<InterpolationPolicy::BILINEAR>>();
            k->configure(input, output, matrix, border_mode, constant_border_value);
            _kernel = std::move(k);
            break;
        }
        case InterpolationPolicy::AREA:
        default:
            ARM_COMPUTE_ERROR("Interpolation type not supported");
    }

    _border_handler.configure(input, _kernel->border_size(), border_mode, constant_border_value);
}
