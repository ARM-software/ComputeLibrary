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
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
namespace
{
/** Define dimension to split the window
 *
 * @param[in] axis Reduction axis
 *
 * @return The dimension to split the window
 */
size_t reduction_window_split_dimension(unsigned int axis)
{
    switch(axis)
    {
        case 0:
            return Window::DimY;
        case 1:
        case 2:
        case 3:
            return Window::DimX;
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction axis");
    }
}
} // namespace

NEReductionOperation::NEReductionOperation()
    : _reduction_kernel(), _fill_border_kernel(), _window_split(0), _reduction_axis()
{
}

Status NEReductionOperation::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ON_ERROR(NEReductionOperationKernel::validate(input, output, axis, op));

    return Status{};
}

void NEReductionOperation::configure(ITensor *input, ITensor *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);

    // Configure reduction kernel
    _reduction_kernel.configure(input, output, axis, op);
    _window_split   = reduction_window_split_dimension(axis);
    _reduction_axis = axis;

    if(axis == 0)
    {
        // Configure fill border kernel
        const BorderSize fill_border_size = _reduction_kernel.border_size();
        const PixelValue pixelValue       = (op == ReductionOperation::PROD) ? PixelValue(1, input->info()->data_type(), input->info()->quantization_info()) : PixelValue(0, input->info()->data_type());
        _fill_border_kernel.configure(input, fill_border_size, BorderMode::CONSTANT, pixelValue);
    }
}

void NEReductionOperation::run()
{
    if(_reduction_axis == 0)
    {
        NEScheduler::get().schedule(&_fill_border_kernel, Window::DimY);
    }
    NEScheduler::get().schedule(&_reduction_kernel, _window_split);
}
} // namespace arm_compute
