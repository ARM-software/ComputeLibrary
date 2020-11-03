/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "src/core/NEON/kernels/NEStackLayerKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
NEStackLayer::~NEStackLayer() = default;

NEStackLayer::NEStackLayer() // NOLINT
    : _input(),
      _stack_kernels(),
      _num_inputs(0)
{
}

void NEStackLayer::configure(const std::vector<ITensor *> &input, int axis, ITensor *output)
{
    _num_inputs = input.size();
    _stack_kernels.resize(_num_inputs);

    // Wrap around negative values
    const unsigned int axis_u = wrap_around(axis, static_cast<int>(input[0]->info()->num_dimensions() + 1));

    for(unsigned int i = 0; i < _num_inputs; i++)
    {
        _stack_kernels[i] = arm_compute::support::cpp14::make_unique<NEStackLayerKernel>();
        _stack_kernels[i]->configure(input[i], axis_u, i, _num_inputs, output);
    }
}

Status NEStackLayer::validate(const std::vector<ITensorInfo *> &input, int axis, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(input.empty());

    // Wrap around negative values
    const size_t       rank   = input[0]->num_dimensions();
    const unsigned int axis_u = wrap_around(axis, static_cast<int>(rank + 1));

    const unsigned int num_inputs = input.size();

    for(unsigned int i = 0; i < num_inputs; i++)
    {
        // All the tensors must have the same rank
        ARM_COMPUTE_RETURN_ERROR_ON(input[i]->num_dimensions() != rank);
        // Validate Kernel
        ARM_COMPUTE_RETURN_ON_ERROR(NEStackLayerKernel::validate(input[i], axis_u, i, num_inputs, output));
    }

    return Status{};
}

void NEStackLayer::run()
{
    for(unsigned i = 0; i < _num_inputs; i++)
    {
        NEScheduler::get().schedule(_stack_kernels[i].get(), Window::DimY);
    }
}
} // namespace arm_compute
