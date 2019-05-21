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
#include "arm_compute/core/NEON/kernels/NEDepthToSpaceLayerKernel.h"

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
    ARM_COMPUTE_RETURN_ERROR_ON(block_shape < 2);

    const DataLayout data_layout = input->data_layout();
    const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_channel] % (block_shape * block_shape) != 0);
    // Validate output if initialized
    if(output->total_size() != 0)
    {
        const int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
        ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape()[idx_width] != (block_shape * input->tensor_shape()[idx_width]));
        ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape()[idx_height] != (block_shape * input->tensor_shape()[idx_height]));
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() > 4);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

NEDepthToSpaceLayerKernel::NEDepthToSpaceLayerKernel()
    : _input(nullptr), _output(nullptr), _block_shape()
{
}

void NEDepthToSpaceLayerKernel::configure(const ITensor *input, ITensor *output, int32_t block_shape)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    TensorShape output_shape = compute_depth_to_space_shape(input->info(), block_shape);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), block_shape));

    _input       = input;
    _output      = output;
    _block_shape = block_shape;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    ICPPKernel::configure(win);
}

Status NEDepthToSpaceLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, int32_t block_shape)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, block_shape));
    return Status{};
}

void NEDepthToSpaceLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    const int idx_channel  = get_data_layout_dimension_index(_input->info()->data_layout(), DataLayoutDimension::CHANNEL);
    const int depth_size   = _input->info()->dimension(idx_channel);
    const int r            = (depth_size / (_block_shape * _block_shape));
    const int element_size = _input->info()->element_size();

    Window slice_out = window.first_slice_window_3D();

    // The slice_out slice does not move
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Main loop for NCHW and NHWC
    if(_input->info()->data_layout() == DataLayout::NCHW)
    {
        Window slice_in = window.first_slice_window_2D();
        do
        {
            Iterator in(_input, slice_in);
            execute_window_loop(slice_in, [&](const Coordinates & id)
            {
                const int x = id.x();
                const int y = id.y();

                const int   z     = id.z() % r;
                const int   out_x = x * _block_shape + (id.z() / r) % _block_shape;
                const int   out_y = y * _block_shape + (id.z() / r) / _block_shape;
                Coordinates output_coords{ out_x, out_y, z, id[3] };
                memcpy(_output->ptr_to_element(output_coords), in.ptr(), element_size);
            },
            in);
        }
        while(window.slide_window_slice_2D(slice_in));
    }
    else
    {
        Window slice_in = window.first_slice_window_3D();
        do
        {
            Iterator in(_input, slice_in);
            execute_window_loop(slice_in, [&](const Coordinates & id)
            {
                const int x = id.y();
                const int y = id.z();

                const int   z     = id.x() % r;
                const int   out_x = x * _block_shape + (id.x() / r) % _block_shape;
                const int   out_y = y * _block_shape + (id.x() / r) / _block_shape;
                Coordinates output_coords{ z, out_x, out_y, id[3] };
                memcpy(_output->ptr_to_element(output_coords), in.ptr(), element_size);
            },
            in);
        }
        while(window.slide_window_slice_3D(slice_in));
    }
}
} // namespace arm_compute
