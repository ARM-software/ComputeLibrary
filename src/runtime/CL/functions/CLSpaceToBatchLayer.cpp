/*
 * Copyright (c) 2018 ARM Limited.
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

#include "arm_compute/runtime/CL/functions/CLSpaceToBatchLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
CLSpaceToBatchLayer::CLSpaceToBatchLayer()
    : _space_to_batch_kernel(), _output(nullptr), _has_padding(false)
{
}

void CLSpaceToBatchLayer::configure(const ICLTensor *input, const ICLTensor *block_shape, const ICLTensor *paddings, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    if(input->info()->tensor_shape().total_size() != output->info()->tensor_shape().total_size())
    {
        _has_padding = true;
    }

    _output = output;
    _space_to_batch_kernel.configure(input, block_shape, paddings, output);
}

void CLSpaceToBatchLayer::configure(const ICLTensor *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    if(input->info()->tensor_shape().total_size() != output->info()->tensor_shape().total_size())
    {
        _has_padding = true;
    }

    _output = output;
    _space_to_batch_kernel.configure(input, block_shape_x, block_shape_y, padding_left, padding_right, output);
}

Status CLSpaceToBatchLayer::validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *paddings, const ITensorInfo *output)
{
    return CLSpaceToBatchLayerKernel::validate(input, block_shape, paddings, output);
}

Status CLSpaceToBatchLayer::validate(const ITensorInfo *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right,
                                     const ITensorInfo *output)
{
    return CLSpaceToBatchLayerKernel::validate(input, block_shape_x, block_shape_y, padding_left, padding_right, output);
}

void CLSpaceToBatchLayer::run()
{
    // Zero out output only if we have paddings
    // TODO(micspy01): replace with memset once ready
    if(_has_padding)
    {
        _output->map(CLScheduler::get().queue(), true);
        if(is_data_type_quantized_asymmetric(_output->info()->data_type()))
        {
            const uint8_t quantized_zero = _output->info()->quantization_info().offset;
            std::fill_n(_output->buffer(), _output->info()->total_size(), quantized_zero);
        }
        else
        {
            memset(_output->buffer(), 0, _output->info()->total_size());
        }
        _output->unmap(CLScheduler::get().queue());
    }

    CLScheduler::get().enqueue(_space_to_batch_kernel, true);
}
} // namespace arm_compute
