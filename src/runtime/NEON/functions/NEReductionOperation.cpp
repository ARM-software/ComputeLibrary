/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEReductionOperationKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "support/MemorySupport.h"

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

NEReductionOperation::~NEReductionOperation() = default;

NEReductionOperation::NEReductionOperation(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _reduction_kernel(), _reshape(), _output_internal(), _window_split(0), _reduction_axis(), _is_reshape_required(false)
{
}

Status NEReductionOperation::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op, bool keep_dims)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");

    const auto is_reshape_required = !keep_dims;

    auto *output_internal = output;

    TensorInfo info_before_reshape;

    if(is_reshape_required)
    {
        const TensorInfo expected_output_shape = output->clone()->set_tensor_shape(arm_compute::misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis, keep_dims));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&expected_output_shape, output);

        auto shape_before_reshape = input->tensor_shape();
        shape_before_reshape.set(axis, 1);

        const auto input_num_channles = input->num_channels();
        const auto input_qinfo        = input->quantization_info();
        const auto is_arg_min_max     = (op == ReductionOperation::ARG_IDX_MAX) || (op == ReductionOperation::ARG_IDX_MIN);
        const auto output_data_type   = is_arg_min_max ? DataType::S32 : output->data_type();

        info_before_reshape.set_data_type(output_data_type).set_tensor_shape(shape_before_reshape).set_num_channels(input_num_channles).set_quantization_info(input_qinfo);

        output_internal = &info_before_reshape;
    }

    ARM_COMPUTE_RETURN_ON_ERROR(NEReductionOperationKernel::validate(input, output_internal, axis, op));

    if(is_reshape_required)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEReshapeLayer::validate(output_internal, output));
    }

    return Status{};
}

void NEReductionOperation::configure(ITensor *input, ITensor *output, unsigned int axis, ReductionOperation op, bool keep_dims)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _is_reshape_required = !keep_dims;

    auto      *output_internal = output;
    const auto is_arg_min_max  = (op == ReductionOperation::ARG_IDX_MAX) || (op == ReductionOperation::ARG_IDX_MIN);

    if(_is_reshape_required)
    {
        const auto output_internal_shape = arm_compute::misc::shape_calculator::compute_reduced_shape(input->info()->tensor_shape(), axis);
        const auto output_external_shape = arm_compute::misc::shape_calculator::compute_reduced_shape(input->info()->tensor_shape(), axis, false);
        const auto output_data_type      = is_arg_min_max ? DataType::S32 : input->info()->data_type();
        const auto num_channels          = input->info()->num_channels();
        const auto qinfo                 = input->info()->quantization_info();

        _output_internal.allocator()->init(input->info()->clone()->set_data_type(output_data_type).set_tensor_shape(output_internal_shape).reset_padding().set_is_resizable(true).set_num_channels(
                                               num_channels).set_quantization_info(qinfo));
        _memory_group.manage(&_output_internal);
        output_internal = &_output_internal;
        auto_init_if_empty(*output->info(), input->info()->clone()->set_data_type(output_data_type).set_tensor_shape(output_external_shape).reset_padding().set_is_resizable(true));
    }

    ARM_COMPUTE_ERROR_THROW_ON(NEReductionOperation::validate(input->info(), output->info(), axis, op, keep_dims));

    // Configure reduction kernel
    _reduction_kernel = arm_compute::support::cpp14::make_unique<NEReductionOperationKernel>();
    _reduction_kernel->configure(input, output_internal, axis, op);
    _window_split   = reduction_window_split_dimension(axis);
    _reduction_axis = axis;

    if(_is_reshape_required)
    {
        _reshape.configure(output_internal, output);
        _output_internal.allocator()->allocate();
    }
}

void NEReductionOperation::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);
    NEScheduler::get().schedule(_reduction_kernel.get(), _window_split);
    if(_is_reshape_required)
    {
        _reshape.run();
    }
}
} // namespace arm_compute
