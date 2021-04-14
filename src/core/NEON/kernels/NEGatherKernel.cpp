/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace
{
/** Validate the indices
 *
 * Validate that indices are not negative
 *
 * @param[in] indices Indices tensor info.
 */
template <typename U>
void validate_indices(const ITensor *indices)
{
    for(size_t i = 0; i < indices->info()->tensor_shape()[0]; ++i)
    {
        ARM_COMPUTE_ERROR_ON(*(reinterpret_cast<U *>(indices->ptr_to_element(Coordinates(i)))) < 0);
    }
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, int axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, indices, output);
    ARM_COMPUTE_RETURN_ERROR_ON(indices->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

    if(axis < 0)
    {
        axis += input->num_dimensions();
    }

    ARM_COMPUTE_RETURN_ERROR_ON(0 > axis || axis >= static_cast<int32_t>(input->num_dimensions()));
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
        TensorShape output_shape = arm_compute::misc::shape_calculator::compute_gather_shape(input->tensor_shape(), indices->tensor_shape(), axis);
        ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
    }

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32, DataType::S32);

    return Status{};
}
} // namespace

NEGatherKernel::NEGatherKernel()
    : _input{}, _indices{}, _axis{}, _output{}, _func{}
{
}

template <typename U>
inline void NEGatherKernel::gather_0_axis(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    // Validate that the indices are not negative
    validate_indices<U>(_indices);

    Iterator output_it(_output, window);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        Coordinates gather_id(id);

        auto new_index = *(reinterpret_cast<U *>(_indices->ptr_to_element(Coordinates(id[0]))));
        gather_id.set(0, new_index);

        std::copy_n(_input->ptr_to_element(gather_id), _output->info()->element_size(), output_it.ptr());
    },
    output_it);
}

template <typename U>
void NEGatherKernel::gather_n_axis(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    // Validate that the indices are not negative
    validate_indices<U>(_indices);

    Window output_window{ window };
    output_window.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator output_it(_output, output_window);
    execute_window_loop(output_window, [&](const Coordinates & id)
    {
        Coordinates gather_id(id);

        auto new_index = *(reinterpret_cast<U *>(_indices->ptr_to_element(Coordinates(id[_axis]))));
        gather_id.set(_axis, new_index);

        std::copy_n(_input->ptr_to_element(gather_id), _input->info()->dimension(0) * _output->info()->element_size(), output_it.ptr());
    },
    output_it);
}

void NEGatherKernel::configure(const ITensor *input, const ITensor *indices, ITensor *output, int axis)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, indices);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), indices->info(), output->info(), axis));

    _input   = input;
    _indices = indices;
    _output  = output;
    _axis    = axis;

    if(_axis < 0)
    {
        _axis += input->info()->num_dimensions();
    }
    ARM_COMPUTE_ERROR_ON(0 > _axis || _axis >= static_cast<int32_t>(input->info()->num_dimensions()));

    if(0 == _axis)
    {
        switch(_indices->info()->data_type())
        {
            case DataType::U32:
                _func = &NEGatherKernel::gather_0_axis<uint32_t>;
                break;
            case DataType::S32:
                _func = &NEGatherKernel::gather_0_axis<int32_t>;
                break;
            default:
                ARM_COMPUTE_ERROR("Not supported");
                break;
        }
    }
    else
    {
        switch(_indices->info()->data_type())
        {
            case DataType::U32:
                _func = &NEGatherKernel::gather_n_axis<uint32_t>;
                break;
            case DataType::S32:
                _func = &NEGatherKernel::gather_n_axis<int32_t>;
                break;
            default:
                ARM_COMPUTE_ERROR("Not supported");
                break;
        }
    }
    // Output auto initialization if not yet initialized
    TensorShape output_shape = arm_compute::misc::shape_calculator::compute_gather_shape(input->info()->tensor_shape(), indices->info()->tensor_shape(), _axis);
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Create window
    Window win = calculate_max_window(*output->info(), Steps());

    INEKernel::configure(win);
}

Status NEGatherKernel::validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, int axis)
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
