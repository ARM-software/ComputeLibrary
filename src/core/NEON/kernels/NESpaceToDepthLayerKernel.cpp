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
#include "arm_compute/core/NEON/kernels/NESpaceToDepthLayerKernel.h"

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
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, int32_t block_shape)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

    ARM_COMPUTE_RETURN_ERROR_ON(block_shape < 1);

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        const DataLayout data_layout = input->data_layout();
        const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
        const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
        const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
        const int        idx_batch   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);
        ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_width] % block_shape != 0);
        ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_height] % block_shape != 0);
        ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_batch] != output->tensor_shape()[idx_batch]);
        ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape()[idx_channel] % (block_shape * block_shape) != 0);
        ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().total_size() != output->tensor_shape().total_size());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

NESpaceToDepthLayerKernel::NESpaceToDepthLayerKernel()
    : _input(nullptr), _output(nullptr), _block_shape(), _data_layout(DataLayout::UNKNOWN)
{
}

void NESpaceToDepthLayerKernel::configure(const ITensor *input, ITensor *output, int32_t block_shape)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    TensorShape output_shape = misc::shape_calculator::compute_space_to_depth_shape(input->info(), block_shape);
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type());

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), block_shape));

    _input       = input;
    _block_shape = block_shape;
    _output      = output;
    _data_layout = input->info()->data_layout();

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    INEKernel::configure(win);
}

Status NESpaceToDepthLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, int32_t block_shape)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, block_shape));
    return Status{};
}

void NESpaceToDepthLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    const int channel_idx  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);
    const int element_size = _input->info()->element_size();

    const size_t channel_size = _input->info()->dimension(channel_idx);

    Window slice_out = window.first_slice_window_3D();

    int batch_id = 0;

    // Main loop for NCHW and NHWC
    if(_data_layout == DataLayout::NCHW)
    {
        do
        {
            Iterator out(_output, slice_out);
            execute_window_loop(slice_out, [&](const Coordinates & id)
            {
                const size_t channel_id = id.z();
                const size_t in_x       = id.x() * _block_shape + (channel_id / channel_size) % _block_shape;
                const size_t in_y       = id.y() * _block_shape + (channel_id / channel_size) / _block_shape;
                const int    z          = channel_id % channel_size;
                Coordinates  input_coords{ in_x, in_y, z, batch_id };
                memcpy(out.ptr(), _input->ptr_to_element(input_coords), element_size);
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
                const size_t channel_id = id.x();
                const size_t in_x       = id.y() * _block_shape + (channel_id / channel_size) % _block_shape;
                const size_t in_y       = id.z() * _block_shape + (channel_id / channel_size) / _block_shape;
                const int    z          = channel_id % channel_size;
                Coordinates  input_coords{ z, in_x, in_y, batch_id };
                memcpy(out.ptr(), _input->ptr_to_element(input_coords), element_size);
            },
            out);
            ++batch_id;
        }
        while(window.slide_window_slice_3D(slice_out));
    }
}
} // namespace arm_compute
