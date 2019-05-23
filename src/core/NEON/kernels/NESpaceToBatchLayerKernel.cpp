/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NESpaceToBatchLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include <arm_neon.h>
#include <cstdint>

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *block_info, const ITensorInfo *padddings, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, block_info, padddings, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(block_info, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(block_info->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(padddings->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(padddings->tensor_shape()[1] != block_info->tensor_shape()[0]);

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        const DataLayout data_layout = input->data_layout();
        const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_channel] != output->tensor_shape()[idx_channel]);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
Status validate_arguments_static(const ITensorInfo *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right,
                                 const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(block_shape_x < 1 || block_shape_y < 1);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        const DataLayout data_layout = input->data_layout();
        const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
        const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
        const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
        const int        idx_batch   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);
        ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape()[idx_width] < padding_left.x() + padding_right.y());
        ARM_COMPUTE_RETURN_ERROR_ON((input->tensor_shape()[idx_width] + padding_left.x() + padding_right.x()) % block_shape_x != 0);
        ARM_COMPUTE_RETURN_ERROR_ON((input->tensor_shape()[idx_height] + padding_left.y() + padding_right.y()) % block_shape_y != 0);
        ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_channel] != output->tensor_shape()[idx_channel]);
        ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape()[idx_batch] % (block_shape_x * block_shape_y) != 0);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

NESpaceToBatchLayerKernel::NESpaceToBatchLayerKernel()
    : _input(nullptr), _block_shape(nullptr), _paddings(nullptr), _output(nullptr), _padding_left(), _block_shape_x(), _block_shape_y()
{
}

void NESpaceToBatchLayerKernel::configure(const ITensor *input, const ITensor *block_shape, const ITensor *paddings, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), block_shape->info(), paddings->info(), output->info()));

    _input       = input;
    _block_shape = block_shape;
    _paddings    = paddings;
    _output      = output;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    ICPPKernel::configure(win);
}

void NESpaceToBatchLayerKernel::configure(const ITensor *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right,
                                          ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    TensorShape output_shape = misc::shape_calculator::compute_space_to_batch_shape(input->info(), block_shape_x, block_shape_y, padding_left, padding_right);
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type());

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_static(input->info(), block_shape_x, block_shape_y, padding_left, padding_right, output->info()));

    _input         = input;
    _output        = output;
    _block_shape_x = block_shape_x;
    _block_shape_y = block_shape_y;
    _padding_left  = padding_left;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    INEKernel::configure(win);
}

Status NESpaceToBatchLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *paddings, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, block_shape, paddings, output));
    return Status{};
}
Status NESpaceToBatchLayerKernel::validate(const ITensorInfo *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right,
                                           const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_static(input, block_shape_x, block_shape_y, padding_left, padding_right, output));
    return Status{};
}

void NESpaceToBatchLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    if(_block_shape != nullptr)
    {
        // Retrieve the block shapes dynamically
        _block_shape_x = *(reinterpret_cast<const int *>(_block_shape->ptr_to_element(0)));
        _block_shape_y = *(reinterpret_cast<const int *>(_block_shape->ptr_to_element(1)));
    }

    if(_paddings != nullptr)
    {
        const size_t pad_left_x = *reinterpret_cast<const size_t *>(_paddings->ptr_to_element({ 0, 0 }));
        const size_t pad_left_y = *reinterpret_cast<const size_t *>(_paddings->ptr_to_element({ 1, 0 }));
        _padding_left           = Size2D(pad_left_x, pad_left_y);
    }
    const DataLayout data_layout  = _input->info()->data_layout();
    const int        height_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        width_idx    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        element_size = _input->info()->element_size();

    const size_t height     = _input->info()->dimension(height_idx);
    const size_t width      = _input->info()->dimension(width_idx);
    const size_t batch_size = _input->info()->dimension(3);

    Window slice_out = window.first_slice_window_3D();
    Window slice_in  = window.first_slice_window_4D();

    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
    slice_in.set(3, Window::Dimension(0, 0, 0));

    int batch_id = 0;

    // Main loop for NCHW and NHWC
    if(_output->info()->data_layout() == DataLayout::NCHW)
    {
        do
        {
            Iterator out(_output, slice_out);
            execute_window_loop(slice_out, [&](const Coordinates & id)
            {
                const size_t out_x = id.x();
                const size_t out_y = id.y();
                const size_t z     = id.z();
                const size_t pos_x = out_x * _block_shape_x + (batch_id / batch_size) % _block_shape_x;
                const size_t pos_y = out_y * _block_shape_y + (batch_id / batch_size) / _block_shape_x;
                if(pos_y >= _padding_left.y() && pos_y < _padding_left.y() + height && pos_x >= _padding_left.x() && pos_x < _padding_left.x() + width)
                {
                    const int   w    = batch_id % batch_size;
                    const int   in_x = pos_x - _padding_left.x();
                    const int   in_y = pos_y - _padding_left.y();
                    Coordinates input_coords{ in_x, in_y, z, w };
                    memcpy(out.ptr(), _input->ptr_to_element(input_coords), element_size);
                }
            },
            out);
            ++batch_id;
        }
        while(window.slide_window_slice_3D(slice_out));
    }
    else
    {
        do
        {
            Iterator out(_output, slice_out);
            execute_window_loop(slice_out, [&](const Coordinates & id)
            {
                const size_t out_x = id.y();
                const size_t out_y = id.z();
                const size_t z     = id.x();
                const size_t pos_x = out_x * _block_shape_x + (batch_id / batch_size) % _block_shape_x;
                const size_t pos_y = out_y * _block_shape_y + (batch_id / batch_size) / _block_shape_x;
                if(pos_y >= _padding_left.y() && pos_y < _padding_left.y() + height && pos_x >= _padding_left.x() && pos_x < _padding_left.x() + width)
                {
                    const int   w    = batch_id % batch_size;
                    const int   in_x = pos_x - _padding_left.x();
                    const int   in_y = pos_y - _padding_left.y();
                    Coordinates input_coords{ z, in_x, in_y, w };
                    memcpy(out.ptr(), _input->ptr_to_element(input_coords), element_size);
                }
            },
            out);
            ++batch_id;
        }
        while(window.slide_window_slice_3D(slice_out));
    }
}
} // namespace arm_compute
