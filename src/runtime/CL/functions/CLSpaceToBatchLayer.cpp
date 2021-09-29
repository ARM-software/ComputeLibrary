/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/core/CL/kernels/CLSpaceToBatchLayerKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
CLSpaceToBatchLayer::CLSpaceToBatchLayer()
    : _space_to_batch_kernel(std::make_unique<CLSpaceToBatchLayerKernel>()),
      _fill(),
      _has_padding(false)
{
}

CLSpaceToBatchLayer::~CLSpaceToBatchLayer() = default;

void CLSpaceToBatchLayer::configure(const ICLTensor *input, const ICLTensor *block_shape, const ICLTensor *paddings, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, block_shape, paddings, output);
}

void CLSpaceToBatchLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *block_shape, const ICLTensor *paddings, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, block_shape, paddings, output);
    ARM_COMPUTE_LOG_PARAMS(input, block_shape, paddings, output);

    if(input->info()->tensor_shape().total_size() != output->info()->tensor_shape().total_size())
    {
        _has_padding = true;
        _fill.configure(compile_context, output, PixelValue(0, input->info()->data_type(), input->info()->quantization_info()));
    }
    _space_to_batch_kernel->configure(compile_context, input, block_shape, paddings, output);
}

void CLSpaceToBatchLayer::configure(const ICLTensor *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, block_shape_x, block_shape_y, padding_left, padding_right, output);
}

void CLSpaceToBatchLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left,
                                    const Size2D &padding_right, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_LOG_PARAMS(input, block_shape_x, block_shape_y, padding_left, padding_right, output);

    if(input->info()->tensor_shape().total_size() != output->info()->tensor_shape().total_size())
    {
        _has_padding = true;
        _fill.configure(compile_context, output, PixelValue(0, input->info()->data_type(), input->info()->quantization_info()));
    }
    _space_to_batch_kernel->configure(compile_context, input, block_shape_x, block_shape_y, padding_left, padding_right, output);
}

Status CLSpaceToBatchLayer::validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *paddings, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(CLFill::validate(output, PixelValue(0, input->data_type(), input->quantization_info())));
    ARM_COMPUTE_RETURN_ON_ERROR(CLSpaceToBatchLayerKernel::validate(input, block_shape, paddings, output));

    return Status{};
}

Status CLSpaceToBatchLayer::validate(const ITensorInfo *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right,
                                     const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(CLFill::validate(output, PixelValue(0, input->data_type(), input->quantization_info())));
    ARM_COMPUTE_RETURN_ON_ERROR(CLSpaceToBatchLayerKernel::validate(input, block_shape_x, block_shape_y, padding_left, padding_right, output));

    return Status{};
}

void CLSpaceToBatchLayer::run()
{
    // Zero out output only if we have paddings
    if(_has_padding)
    {
        //CLScheduler::get().enqueue(*_fill, true);
        _fill.run();
    }
    CLScheduler::get().enqueue(*_space_to_batch_kernel, true);
}
} // namespace arm_compute
