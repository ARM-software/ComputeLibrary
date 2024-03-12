/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEStackLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/NEON/kernels/NEStackLayerKernel.h"

namespace arm_compute
{
NEStackLayer::~NEStackLayer() = default;

NEStackLayer::NEStackLayer() // NOLINT
    : _stack_kernel(std::make_unique<NEStackLayerKernel>()), _is_prepared(false)
{
}

void NEStackLayer::configure(const std::vector<ITensor *> &input, int axis, ITensor *output)
{
    ARM_COMPUTE_LOG_PARAMS(input, axis, output);

    // Wrap around negative values
    const unsigned int axis_u = wrap_around(axis, static_cast<int>(input[0]->info()->num_dimensions() + 1));

    _stack_kernel->configure(input, axis_u, output);
}

Status NEStackLayer::validate(const std::vector<ITensorInfo *> &input, int axis, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(input.empty());

    // Wrap around negative values
    const size_t       rank   = input[0]->num_dimensions();
    const unsigned int axis_u = wrap_around(axis, static_cast<int>(rank + 1));

    // Validate Kernel
    ARM_COMPUTE_RETURN_ON_ERROR(NEStackLayerKernel::validate(input, axis_u, output));

    return Status{};
}

void NEStackLayer::run()
{
    if (!_is_prepared)
    {
        _stack_kernel->prepare();
        _is_prepared = true;
    }

    NEScheduler::get().schedule(_stack_kernel.get(), _stack_kernel->get_split_dimension());
}
} // namespace arm_compute
