/*
 * Copyright (c) 2017-2020 ARM Limited.
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

NEReductionOperation::NEReductionOperation(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _reduction_kernel(), _fill_border_kernel(), _reshape_kernel(), _output_internal(), _window_split(0), _reduction_axis(), _is_reshape_required(false)
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
        ARM_COMPUTE_RETURN_ON_ERROR(NEReshapeLayerKernel::validate(output_internal, output));
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
    _reduction_kernel.configure(input, output_internal, axis, op);
    _window_split   = reduction_window_split_dimension(axis);
    _reduction_axis = axis;

    if(axis == 0)
    {
        // Configure fill border kernel
        const BorderSize fill_border_size = _reduction_kernel.border_size();
        PixelValue       pixelValue;
        switch(op)
        {
            case ReductionOperation::PROD:
            {
                pixelValue = PixelValue(1, input->info()->data_type(), input->info()->quantization_info());
                break;
            }
            case ReductionOperation::MIN:
            {
                pixelValue = std::get<1>(get_min_max(input->info()->data_type()));
                break;
            }
            case ReductionOperation::MAX:
            {
                pixelValue = std::get<0>(get_min_max(input->info()->data_type()));
                break;
            }
            case ReductionOperation::ARG_IDX_MAX:
            case ReductionOperation::ARG_IDX_MIN:
            {
                pixelValue = PixelValue(0, input->info()->data_type(), input->info()->quantization_info());
                break;
            }
            case ReductionOperation::MEAN_SUM:
            case ReductionOperation::SUM_SQUARE:
            case ReductionOperation::SUM:
            {
                pixelValue = PixelValue(static_cast<uint32_t>(0));
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Reduction Operation unsupported");
        }
        _fill_border_kernel.configure(input, fill_border_size, (is_arg_min_max ? BorderMode::REPLICATE : BorderMode::CONSTANT), pixelValue);
    }

    if(_is_reshape_required)
    {
        _reshape_kernel.configure(output_internal, output);
        _output_internal.allocator()->allocate();
    }
}

void NEReductionOperation::run()
{
    if(_reduction_axis == 0)
    {
        NEScheduler::get().schedule(&_fill_border_kernel, Window::DimY);
    }
    NEScheduler::get().schedule(&_reduction_kernel, _window_split);
    if(_is_reshape_required)
    {
        NEScheduler::get().schedule(&_reshape_kernel, Window::DimY);
    }
}
} // namespace arm_compute
