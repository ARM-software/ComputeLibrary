/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/kernels/CLReductionOperationKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/runtime/Utils.h"

namespace arm_compute
{
CLReductionOperation::CLReductionOperation(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _unreshaped_output(),
      _reduction_kernel(),
      _reshape(),
      _reduction_axis(),
      _is_reshape_required(false)
{
}

CLReductionOperation::~CLReductionOperation() = default;

Status CLReductionOperation::validate(
    const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op, bool keep_dims)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions,
                                    "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");

    const bool is_reshape_required = !keep_dims;

    if (is_reshape_required && output->total_size() != 0)
    {
        const TensorInfo expected_output_shape = output->clone()->set_tensor_shape(
            arm_compute::misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis, keep_dims));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&expected_output_shape, output);
    }

    auto *output_internal = output;

    TensorInfo output_before_reshape;
    const auto input_shape        = input->tensor_shape();
    const auto input_num_channles = input->num_channels();
    const auto input_qinfo        = input->quantization_info();
    const auto output_data_type   = output->data_type();

    auto initialize_tensorinfo = [](TensorInfo &ti, TensorShape shape, DataType data_type, int num_channels,
                                    QuantizationInfo qinfo) {
        ti.set_data_type(data_type).set_tensor_shape(shape).set_num_channels(num_channels).set_quantization_info(qinfo);
    };

    if (is_reshape_required)
    {
        auto shape_before_reshape = input_shape;
        shape_before_reshape.set(axis, 1);
        initialize_tensorinfo(output_before_reshape, shape_before_reshape, output_data_type, input_num_channles,
                              input_qinfo);
        output_internal = &output_before_reshape;
    }

    ARM_COMPUTE_RETURN_ON_ERROR(CLReductionOperationKernel::validate(input, output_internal, axis, op));

    if (is_reshape_required)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLReshapeLayer::validate(output_internal, output));
    }

    return Status{};
}

ICLTensor *CLReductionOperation::configure_intermediate_result_vector(ICLTensor *input, ICLTensor *output)
{
    if (!_is_reshape_required)
    {
        return output;
    }

    auto shape = input->info()->tensor_shape();
    shape.set(_reduction_axis, 1);
    _unreshaped_output.allocator()->init(input->info()->clone()->set_tensor_shape(shape));
    return &_unreshaped_output;
}

void CLReductionOperation::configure(
    ICLTensor *input, ICLTensor *output, unsigned int axis, ReductionOperation op, bool keep_dims)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, axis, op, keep_dims);
}

void CLReductionOperation::configure(const CLCompileContext &compile_context,
                                     ICLTensor              *input,
                                     ICLTensor              *output,
                                     unsigned int            axis,
                                     ReductionOperation      op,
                                     bool                    keep_dims)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_LOG_PARAMS(input, output, axis, op, keep_dims);
    _reduction_axis      = axis;
    _is_reshape_required = !keep_dims;

    auto *output_internal = configure_intermediate_result_vector(input, output);

    if (_is_reshape_required)
    {
        const TensorShape output_shape =
            arm_compute::misc::shape_calculator::compute_reduced_shape(input->info()->tensor_shape(), axis, false);
        const auto output_data_type = input->info()->data_type();
        auto_init_if_empty(*output->info(), input->info()
                                                ->clone()
                                                ->set_tensor_shape(output_shape)
                                                .set_data_type(output_data_type)
                                                .reset_padding()
                                                .set_is_resizable(true));

        _memory_group.manage(&_unreshaped_output);
    }

    _reduction_kernel = std::make_unique<CLReductionOperationKernel>();
    _reduction_kernel->configure(compile_context, input, output_internal, axis, op);

    if (_is_reshape_required)
    {
        _reshape.configure(compile_context, &_unreshaped_output, output);
        _unreshaped_output.allocator()->allocate();
    }
}

void CLReductionOperation::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    CLScheduler::get().enqueue(*_reduction_kernel, false);

    if (_is_reshape_required)
    {
        _reshape.run();
    }
}
} // namespace arm_compute
