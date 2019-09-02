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
#include "arm_compute/core/CL/kernels/CLDepthToSpaceLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

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

CLDepthToSpaceLayerKernel::CLDepthToSpaceLayerKernel()
    : _input(nullptr), _output(nullptr), _block_shape()
{
}

void CLDepthToSpaceLayerKernel::configure(const ICLTensor *input, ICLTensor *output, int32_t block_shape)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    TensorShape output_shape = compute_depth_to_space_shape(input->info(), block_shape);
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type());

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), block_shape));

    _input       = input;
    _output      = output;
    _block_shape = block_shape;

    const int idx_width   = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);
    const int idx_channel = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::CHANNEL);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DCHANNEL_SIZE=" + support::cpp11::to_string(input->info()->dimension(idx_channel)));
    build_opts.add_option("-DBLOCK_SHAPE=" + support::cpp11::to_string(block_shape));
    build_opts.add_option("-DWIDTH_IN=" + support::cpp11::to_string(input->info()->dimension(idx_width)));
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("depth_to_space_" + lower_string(string_from_data_layout(input->info()->data_layout())), build_opts.options()));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    ICLKernel::configure_internal(win);
}

Status CLDepthToSpaceLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, int32_t block_shape)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, block_shape));
    return Status{};
}

void CLDepthToSpaceLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice_in  = window.first_slice_window_3D();
    Window slice_out = window.first_slice_window_4D();

    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
    slice_out.set(3, Window::Dimension(0, 0, 0));

    int batch_id = 0;
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_argument(idx, batch_id);
        add_4D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_in);

        ++batch_id;
    }
    while(window.slide_window_slice_3D(slice_in));
}
} // namespace arm_compute
