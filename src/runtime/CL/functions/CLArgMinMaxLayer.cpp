/*
 * Copyright (c) 2018-2020 ARM Limited.
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

#include "arm_compute/runtime/CL/functions/CLArgMinMaxLayer.h"

#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Utils.h"

namespace arm_compute
{
CLArgMinMaxLayer::CLArgMinMaxLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _results_vector(), _not_reshaped_output(), _reduction_kernels_vector(), _reshape(), _num_of_stages(), _reduction_axis()
{
}

Status CLArgMinMaxLayer::validate(const ITensorInfo *input, int axis, const ITensorInfo *output, const ReductionOperation &op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(op != ReductionOperation::ARG_IDX_MAX && op != ReductionOperation::ARG_IDX_MIN, "Invalid reduction operation");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= static_cast<int>(TensorShape::num_max_dimensions), "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");
    const unsigned int num_of_stages = calculate_number_of_stages_only_x_axis(input->dimension(0), axis);

    DataType   output_data_type = DataType::S32;
    TensorInfo not_reshaped_output;
    const auto input_num_channles = input->num_channels();
    const auto input_qinfo        = input->quantization_info();

    if(output->total_size() != 0)
    {
        output_data_type                       = output->data_type();
        const TensorInfo expected_output_shape = output->clone()->set_tensor_shape(arm_compute::misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis, false));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&expected_output_shape, output);
    }

    auto shape_before_reshape = input->tensor_shape();
    shape_before_reshape.set(axis, 1);
    auto initialize_tensorinfo = [](TensorInfo & ti, TensorShape shape, DataType data_type, int num_channels, QuantizationInfo qinfo)
    {
        ti.set_data_type(data_type).set_tensor_shape(shape).set_num_channels(num_channels).set_quantization_info(qinfo);
    };

    initialize_tensorinfo(not_reshaped_output, shape_before_reshape, output_data_type, input_num_channles, input_qinfo);

    if(num_of_stages == 1)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLArgMinMaxLayerKernel::validate(input, nullptr, &not_reshaped_output, axis, op));
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

        // Validate ReductionOperation only on first kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLArgMinMaxLayerKernel::validate(input, nullptr, &sums_vector[0], axis, op));

        // Validate ReductionOperation on intermediate stages
        for(unsigned int i = 1; i < num_of_stages - 1; ++i)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLArgMinMaxLayerKernel::validate(input, &sums_vector[i - 1], &sums_vector[i], axis, op));
        }

        // Validate ReductionOperation on the last stage
        const unsigned int last_stage = num_of_stages - 1;
        ARM_COMPUTE_RETURN_ON_ERROR(CLArgMinMaxLayerKernel::validate(input, &sums_vector[last_stage - 1], &not_reshaped_output, axis, op));
    }
    ARM_COMPUTE_RETURN_ON_ERROR(CLReshapeLayer::validate(&not_reshaped_output, output));
    return Status{};
}

void CLArgMinMaxLayer::configure(const ICLTensor *input, int axis, ICLTensor *output, const ReductionOperation &op)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, axis, output, op);
}

void CLArgMinMaxLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, int axis, ICLTensor *output, const ReductionOperation &op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    _num_of_stages  = calculate_number_of_stages_only_x_axis(input->info()->dimension(0), axis);
    _reduction_axis = axis;

    const TensorShape output_shape     = arm_compute::misc::shape_calculator::compute_reduced_shape(input->info()->tensor_shape(), axis, false);
    DataType          output_data_type = (output->info()->data_type() == DataType::UNKNOWN) ? DataType::S32 : output->info()->data_type();
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape).set_data_type(output_data_type).reset_padding().set_is_resizable(true));

    // Configure reduction operation kernels
    _reduction_kernels_vector.resize(_num_of_stages);

    _memory_group.manage(&_not_reshaped_output);
    // Create temporary tensors
    if(_num_of_stages == 1)
    {
        _reduction_kernels_vector[0].configure(compile_context, input, nullptr, &_not_reshaped_output, axis, op);
    }
    else
    {
        _results_vector.resize(_num_of_stages - 1);
        TensorShape shape{ input->info()->tensor_shape() };
        for(unsigned int i = 0; i < _num_of_stages - 1; i++)
        {
            shape.set(0, ceil(shape.x() / 128.f));
            _results_vector[i].allocator()->init(input->info()->clone()->set_tensor_shape(shape).set_data_type(output_data_type));
        }

        // Apply ReductionOperation only on first kernel
        _memory_group.manage(&_results_vector[0]);
        _reduction_kernels_vector[0].configure(compile_context, input, nullptr, &_results_vector[0], axis, op);

        // Apply ReductionOperation on intermediate stages
        for(unsigned int i = 1; i < _num_of_stages - 1; ++i)
        {
            _memory_group.manage(&_results_vector[i]);
            _reduction_kernels_vector[i].configure(compile_context, input, &_results_vector[i - 1], &_results_vector[i], axis, op);
            _results_vector[i - 1].allocator()->allocate();
        }

        // Apply ReductionOperation on the last stage
        const unsigned int last_stage = _num_of_stages - 1;
        _reduction_kernels_vector[last_stage].configure(compile_context, input, &_results_vector[last_stage - 1], &_not_reshaped_output, axis, op);
        _results_vector[last_stage - 1].allocator()->allocate();
    }
    _reshape.configure(compile_context, &_not_reshaped_output, output);
    _not_reshaped_output.allocator()->allocate();
}

void CLArgMinMaxLayer::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    for(unsigned int i = 0; i < _num_of_stages; ++i)
    {
        CLScheduler::get().enqueue(_reduction_kernels_vector[i], false);
    }
    _reshape.run();
}
} // namespace arm_compute