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
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLReductionOperationKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/Tensor.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
unsigned int calculate_number_of_stages(const ITensorInfo *input, unsigned int axis)
{
    // We need only 1 stage for all axis except x-axis and x-axis for QASYMM8.
    if(axis != 0 || (axis == 0 && is_data_type_quantized(input->data_type())))
    {
        return 1;
    }
    // Calculate number of WGs. 16 elements per thread, 8 threads per WG
    const unsigned int num_of_wg = ceil(input->dimension(0) / 128.f);

    // Calculate number of stages. First stage performs op and the rest reduction sum
    // depending on the size of the input. Last stage should have only 1 WG.
    const unsigned int num_of_stages = num_of_wg / 128 + 2;

    return num_of_stages;
}
} // namespace

CLReductionOperation::CLReductionOperation(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _results_vector(), _reduction_kernels_vector(), _border_handlers_vector(), _num_of_stages(), _reduction_axis(), _is_serial()
{
}

Status CLReductionOperation::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    const unsigned int num_of_stages = calculate_number_of_stages(input, axis);
    bool               is_serial     = is_data_type_quantized(input->data_type()) || axis != 0;
    if(is_serial)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLReductionOperationKernel::validate(input, output, axis, op));
    }
    else
    {
        // Create temporary tensor infos
        std::vector<TensorInfo> sums_vector(num_of_stages - 1);

        // Create intermediate tensor info
        TensorShape shape{ input->tensor_shape() };

        for(unsigned int i = 0; i < num_of_stages - 1; i++)
        {
            shape.set(0, ceil(shape.x() / 128.f));
            sums_vector[i].set_data_type(input->data_type());
            sums_vector[i].set_tensor_shape(shape);
            sums_vector[i].set_num_channels(input->num_channels());
        }

        ReductionOperation first_kernel_op;
        ReductionOperation intermediate_kernel_op;
        ReductionOperation last_kernel_op;
        switch(op)
        {
            case ReductionOperation::SUM:
            case ReductionOperation::MEAN_SUM:
                first_kernel_op        = ReductionOperation::SUM;
                intermediate_kernel_op = ReductionOperation::SUM;
                last_kernel_op         = op;
                break;
            case ReductionOperation::SUM_SQUARE:
                first_kernel_op        = ReductionOperation::SUM_SQUARE;
                intermediate_kernel_op = ReductionOperation::SUM;
                last_kernel_op         = ReductionOperation::SUM;
                break;
            case ReductionOperation::PROD:
                first_kernel_op        = ReductionOperation::PROD;
                intermediate_kernel_op = ReductionOperation::PROD;
                last_kernel_op         = ReductionOperation::PROD;
                break;
            case ReductionOperation::MIN:
                first_kernel_op        = ReductionOperation::MIN;
                intermediate_kernel_op = ReductionOperation::MIN;
                last_kernel_op         = ReductionOperation::MIN;
                break;
            case ReductionOperation::MAX:
                first_kernel_op        = ReductionOperation::MAX;
                intermediate_kernel_op = ReductionOperation::MAX;
                last_kernel_op         = ReductionOperation::MAX;
                break;
            default:
                ARM_COMPUTE_ERROR("Not supported");
        }

        // Validate ReductionOperation only on first kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLReductionOperationKernel::validate(input, &sums_vector[0], axis, first_kernel_op));

        // Validate ReductionOperation on intermediate stages
        for(unsigned int i = 1; i < num_of_stages - 1; ++i)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLReductionOperationKernel::validate(&sums_vector[i - 1], &sums_vector[i], axis, intermediate_kernel_op));
        }

        // Validate ReductionOperation on the last stage
        const unsigned int last_stage = num_of_stages - 1;
        ARM_COMPUTE_RETURN_ON_ERROR(CLReductionOperationKernel::validate(&sums_vector[last_stage - 1], output, axis, last_kernel_op, input->dimension(0)));
    }

    return Status{};
}

void CLReductionOperation::configure(ICLTensor *input, ICLTensor *output, unsigned int axis, ReductionOperation op)
{
    _num_of_stages  = calculate_number_of_stages(input->info(), axis);
    _reduction_axis = axis;
    _is_serial      = is_data_type_quantized(input->info()->data_type()) || axis != 0;

    // Configure reduction operation kernels
    _reduction_kernels_vector.resize(_num_of_stages);

    // Create temporary tensors
    if(_is_serial)
    {
        _reduction_kernels_vector[0].configure(input, output, axis, op, 0);
    }
    else
    {
        _border_handlers_vector.resize(_num_of_stages);
        _results_vector.resize(_num_of_stages - 1);
        TensorShape shape{ input->info()->tensor_shape() };
        for(unsigned int i = 0; i < _num_of_stages - 1; i++)
        {
            shape.set(0, ceil(shape.x() / 128.f));
            _results_vector[i].allocator()->init(input->info()->clone()->set_tensor_shape(shape));
        }

        // Apply ReductionOperation only on first kernel
        _memory_group.manage(&_results_vector[0]);

        ReductionOperation first_kernel_op;
        ReductionOperation intermediate_kernel_op;
        ReductionOperation last_kernel_op;
        PixelValue         pixelValue;
        switch(op)
        {
            case ReductionOperation::SUM:
            case ReductionOperation::MEAN_SUM:
                first_kernel_op        = ReductionOperation::SUM;
                intermediate_kernel_op = ReductionOperation::SUM;
                last_kernel_op         = op;
                pixelValue             = PixelValue();
                break;
            case ReductionOperation::SUM_SQUARE:
                first_kernel_op        = ReductionOperation::SUM_SQUARE;
                intermediate_kernel_op = ReductionOperation::SUM;
                last_kernel_op         = ReductionOperation::SUM;
                pixelValue             = PixelValue();
                break;
            case ReductionOperation::PROD:
                first_kernel_op        = ReductionOperation::PROD;
                intermediate_kernel_op = ReductionOperation::PROD;
                last_kernel_op         = ReductionOperation::PROD;
                pixelValue             = PixelValue(1, input->info()->data_type());
                break;
            case ReductionOperation::MIN:
                first_kernel_op        = ReductionOperation::MIN;
                intermediate_kernel_op = ReductionOperation::MIN;
                last_kernel_op         = ReductionOperation::MIN;
                switch(input->info()->data_type())
                {
                    case DataType::F32:
                    {
                        pixelValue = PixelValue(std::numeric_limits<float>::max());
                        break;
                    }
                    case DataType::F16:
                    {
                        pixelValue = PixelValue(static_cast<half>(65504.0f));
                        break;
                    }
                    case DataType::QASYMM8:
                    {
                        pixelValue = PixelValue(255, input->info()->data_type(), input->info()->quantization_info());
                        break;
                    }
                    default:
                    {
                        ARM_COMPUTE_ERROR("Unsupported DataType");
                    }
                }
                break;
            case ReductionOperation::MAX:
                first_kernel_op        = ReductionOperation::MAX;
                intermediate_kernel_op = ReductionOperation::MAX;
                last_kernel_op         = ReductionOperation::MAX;
                switch(input->info()->data_type())
                {
                    case DataType::F32:
                    {
                        pixelValue = PixelValue(-std::numeric_limits<float>::max());
                        break;
                    }
                    case DataType::F16:
                    {
                        pixelValue = PixelValue(static_cast<half>(-65504.0f));
                        break;
                    }
                    case DataType::QASYMM8:
                    {
                        pixelValue = PixelValue(0, input->info()->data_type(), input->info()->quantization_info());
                        break;
                    }
                    default:
                    {
                        ARM_COMPUTE_ERROR("Unsupported DataType");
                    }
                }
                break;
            default:
                ARM_COMPUTE_ERROR("Not supported");
        }

        _reduction_kernels_vector[0].configure(input, &_results_vector[0], axis, first_kernel_op);
        _border_handlers_vector[0].configure(input, _reduction_kernels_vector[0].border_size(), BorderMode::CONSTANT, pixelValue);

        // Apply ReductionOperation on intermediate stages
        for(unsigned int i = 1; i < _num_of_stages - 1; ++i)
        {
            _memory_group.manage(&_results_vector[i]);
            _reduction_kernels_vector[i].configure(&_results_vector[i - 1], &_results_vector[i], axis, intermediate_kernel_op);
            _border_handlers_vector[i].configure(&_results_vector[i - 1], _reduction_kernels_vector[i].border_size(), BorderMode::CONSTANT, pixelValue);
            _results_vector[i - 1].allocator()->allocate();
        }

        // Apply ReductionOperation on the last stage
        const unsigned int last_stage  = _num_of_stages - 1;
        const unsigned int input_width = input->info()->dimension(0);
        _reduction_kernels_vector[last_stage].configure(&_results_vector[last_stage - 1], output, axis, last_kernel_op, input_width);
        _border_handlers_vector[last_stage].configure(&_results_vector[last_stage - 1], _reduction_kernels_vector[last_stage].border_size(), BorderMode::CONSTANT, pixelValue);
        _results_vector[last_stage - 1].allocator()->allocate();
    }
}

void CLReductionOperation::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_is_serial)
    {
        CLScheduler::get().enqueue(_reduction_kernels_vector[0], false);
    }
    else
    {
        for(unsigned int i = 0; i < _num_of_stages; ++i)
        {
            CLScheduler::get().enqueue(_border_handlers_vector[i], false);
            CLScheduler::get().enqueue(_reduction_kernels_vector[i], false);
        }
    }
}
