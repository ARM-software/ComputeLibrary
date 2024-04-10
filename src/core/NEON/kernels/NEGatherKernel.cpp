/*
 * Copyright (c) 2019-2023 Arm Limited.
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
#include "src/core/NEON/kernels/NEGatherKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, int axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, indices, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

    if (axis < 0)
    {
        axis += input->num_dimensions();
    }

    ARM_COMPUTE_RETURN_ERROR_ON(0 > axis || axis >= static_cast<int32_t>(input->num_dimensions()));
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() + indices->num_dimensions() - 1 >
                                Coordinates::num_max_dimensions);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);

    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
        TensorShape output_shape = arm_compute::misc::shape_calculator::compute_gather_shape(
            input->tensor_shape(), indices->tensor_shape(), axis);
        ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
    }

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32, DataType::S32);

    return Status{};
}
} // namespace

NEGatherKernel::NEGatherKernel()
    : _input{}, _indices{}, _axis{}, _output{}, _func{}, _src_it_strides{}, _idx_it_strides{}
{
}

template <typename TIndex>
void NEGatherKernel::gather_common(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    auto dst_win = window;

    const auto src_info = _input->info();
    const auto idx_info = _indices->info();
    const auto dst_info = _output->info();

    const auto num_dims     = dst_info->num_dimensions();
    const auto chunk_stride = src_info->strides_in_bytes()[_axis];

    const auto window_start_x = window.x().start();
    const auto window_end_x   = window.x().end();
    auto       window_size_x  = src_info->element_size();

    const auto idx_limit = static_cast<TIndex>(src_info->tensor_shape()[_axis]);

    if (_axis != 0)
    {
        dst_win.set(0, Window::Dimension(window_start_x, window_start_x + 1, 1));
        window_size_x *= window_end_x - window_start_x;
    }

    // Compute source and index tensors window based on the output window.
    auto   src_win = dst_win;
    Window idx_win;

    for (size_t i = 0; i < idx_info->num_dimensions(); ++i)
    {
        src_win.set(_axis + i, Window::Dimension(0, 1, 1));
        idx_win.set(_axis + i, window[_axis + i]);
    }

    // Use the custom strides to access all three tensors using the same loop.
    Iterator src_it(num_dims, _src_it_strides, _input->buffer(), src_info->offset_first_element_in_bytes(), src_win);
    Iterator idx_it(num_dims, _idx_it_strides, _indices->buffer(), idx_info->offset_first_element_in_bytes(), idx_win);
    Iterator dst_it(num_dims, dst_info->strides_in_bytes(), _output->buffer(),
                    dst_info->offset_first_element_in_bytes(), dst_win);

    execute_window_loop(
        dst_win,
        [&](const Coordinates &)
        {
            const auto idx = *reinterpret_cast<const TIndex *>(idx_it.ptr());

            if (idx >= 0 && idx < idx_limit)
            {
                const auto src_ptr = src_it.ptr() + idx * chunk_stride;

                std::copy_n(src_ptr, window_size_x, dst_it.ptr());
            }
            else
            {
                std::fill_n(dst_it.ptr(), window_size_x, 0);
            }
        },
        src_it, idx_it, dst_it);
}

void NEGatherKernel::configure(const ITensor *input, const ITensor *indices, ITensor *output, int axis)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, indices);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), indices->info(), output->info(), axis));

    _input   = input;
    _indices = indices;
    _output  = output;
    _axis    = axis;

    if (_axis < 0)
    {
        _axis += input->info()->num_dimensions();
    }
    ARM_COMPUTE_ERROR_ON(0 > _axis || _axis >= static_cast<int32_t>(input->info()->num_dimensions()));

    switch (_indices->info()->data_type())
    {
        case DataType::U32:
            _func = &NEGatherKernel::gather_common<uint32_t>;
            break;
        case DataType::S32:
            _func = &NEGatherKernel::gather_common<int32_t>;
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    // Output auto initialization if not yet initialized
    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_gather_shape(
        input->info()->tensor_shape(), indices->info()->tensor_shape(), _axis);
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Create window
    Window win = calculate_max_window(*output->info(), Steps());

    INEKernel::configure(win);

    // Create input and indices strides that have the same number of dimensions as the output tensor.
    // These will be used to iterate lock-step through all tensors (input, indices and output).
    size_t dim_no = 0;

    const auto  input_info    = input->info();
    const auto &input_strides = input_info->strides_in_bytes();

    const auto  indices_info     = indices->info();
    const auto &indices_strides  = indices_info->strides_in_bytes();
    const auto  indices_num_dims = indices_info->num_dimensions();

    for (; dim_no < static_cast<size_t>(_axis); ++dim_no)
    {
        _src_it_strides[dim_no] = input_strides[dim_no];
    }

    for (; dim_no < static_cast<size_t>(_axis) + indices_num_dims; ++dim_no)
    {
        _idx_it_strides[dim_no] = indices_strides[dim_no - _axis];
    }

    for (; dim_no < Coordinates::num_max_dimensions; ++dim_no)
    {
        _src_it_strides[dim_no] = input_strides[dim_no - indices_num_dims + 1];
    }
}

Status
NEGatherKernel::validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, int axis)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, indices, output, axis));
    return Status{};
}

void NEGatherKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window, info);
}

} // namespace arm_compute
