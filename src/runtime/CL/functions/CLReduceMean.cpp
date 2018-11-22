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
#include "arm_compute/runtime/CL/functions/CLReduceMean.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLReductionOperationKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/helpers/tensor_transform.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
CLReduceMean::CLReduceMean(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _reduction_kernels(), _reduced_outs(), _reshape(), _reduction_ops(), _keep_dims()
{
}
void CLReduceMean::configure(ICLTensor *input, const Coordinates &reduction_axis, bool keep_dims, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _reduction_ops     = reduction_axis.num_dimensions();
    _reduction_kernels = arm_compute::support::cpp14::make_unique<CLReductionOperation[]>(_reduction_ops);
    _reduced_outs      = arm_compute::support::cpp14::make_unique<CLTensor[]>(_reduction_ops - (keep_dims ? 1 : 0));
    _keep_dims         = keep_dims;

    // Perform reduction for every axis
    for(unsigned int i = 0; i < _reduction_ops; ++i)
    {
        TensorShape out_shape = i == 0 ? input->info()->tensor_shape() : (_reduced_outs.get() + i - 1)->info()->tensor_shape();
        out_shape.set(reduction_axis[i], 1);
        auto in = (i == 0) ? input : (_reduced_outs.get() + i - 1);

        if(i == _reduction_ops - 1 && keep_dims)
        {
            _reduction_kernels[i].configure(in, output, reduction_axis[i], ReductionOperation::MEAN_SUM);
        }
        else
        {
            _reduced_outs[i].allocator()->init(TensorInfo(out_shape, input->info()->num_channels(), input->info()->data_type(), input->info()->quantization_info()));
            _memory_group.manage(_reduced_outs.get() + i);
            _reduction_kernels[i].configure(in, _reduced_outs.get() + i, reduction_axis[i], ReductionOperation::MEAN_SUM);
        }
    }

    // Allocate intermediate tensors
    for(unsigned int i = 0; i < _reduction_ops - (keep_dims ? 1 : 0); ++i)
    {
        _reduced_outs[i].allocator()->allocate();
    }

    // Configure reshape layer if we want to drop the dimensions
    if(!keep_dims)
    {
        TensorShape out_shape = input->info()->tensor_shape();

        // We have to sort the reduction axis vectors in order for remove_dimension
        // to work properly
        Coordinates axis_copy = reduction_axis;
        std::sort(axis_copy.begin(), axis_copy.begin() + _reduction_ops);
        for(unsigned int i = 0; i < _reduction_ops; ++i)
        {
            out_shape.remove_dimension(axis_copy[i] - i);
        }
        auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(out_shape));
        _reshape.configure(_reduced_outs.get() + _reduction_ops - 1, output);
    }
}

Status CLReduceMean::validate(const ITensorInfo *input, const Coordinates &reduction_axis, bool keep_dims, const ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(keep_dims);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON(reduction_axis.num_dimensions() > input->num_dimensions());

    for(unsigned int i = 0; i < reduction_axis.num_dimensions(); ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(reduction_axis[i] > 3);
        ARM_COMPUTE_RETURN_ERROR_ON(static_cast<unsigned int>(reduction_axis[i]) > input->num_dimensions() - 1);
        if(output->total_size() > 0 && keep_dims)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(reduction_axis[i]) != 1);
        }

        ARM_COMPUTE_RETURN_ON_ERROR(CLReductionOperation::validate(input, output, reduction_axis[i], ReductionOperation::MEAN_SUM));
    }

    return Status{};
}

void CLReduceMean::run()
{
    _memory_group.acquire();

    for(unsigned int i = 0; i < _reduction_ops; ++i)
    {
        _reduction_kernels[i].run();
    }

    if(!_keep_dims)
    {
        _reshape.run();
    }
    _memory_group.release();
}
} // namespace arm_compute
