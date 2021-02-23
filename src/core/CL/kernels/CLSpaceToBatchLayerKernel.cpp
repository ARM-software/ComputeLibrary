/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLSpaceToBatchLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *block_info, const ITensorInfo *paddings, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, block_info, paddings, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(block_info, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(block_info->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(block_info->tensor_shape(), TensorShape{ 2 });
    ARM_COMPUTE_RETURN_ERROR_ON(paddings->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(paddings->tensor_shape(), TensorShape{ 2, 2 });

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        const DataLayout data_layout = input->data_layout();
        const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_channel] != output->tensor_shape()[idx_channel]);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
Status validate_arguments_static(const ITensorInfo *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right,
                                 const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(block_shape_x < 1 || block_shape_y < 1);

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        TensorShape expected_output_shape = misc::shape_calculator::compute_space_to_batch_shape(input, block_shape_x, block_shape_y, padding_left, padding_right);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), expected_output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

CLSpaceToBatchLayerKernel::CLSpaceToBatchLayerKernel()
    : _input(nullptr), _block_shape(nullptr), _paddings(nullptr), _output(nullptr)
{
}

void CLSpaceToBatchLayerKernel::configure(const ICLTensor *input, const ICLTensor *block_shape, const ICLTensor *paddings, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, block_shape, paddings, output);
}

void CLSpaceToBatchLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *block_shape, const ICLTensor *paddings, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, block_shape, paddings, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), block_shape->info(), paddings->info(), output->info()));
    auto padding_info = get_padding_info({ input, block_shape, paddings, output });

    _input       = input;
    _block_shape = block_shape;
    _paddings    = paddings;
    _output      = output;

    const DataLayout data_layout = input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_batch   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(data_size_from_type(input->info()->data_type())));
    build_opts.add_option("-DWIDTH_OUT=" + support::cpp11::to_string(output->info()->dimension(idx_width)));
    build_opts.add_option("-DHEIGHT_OUT=" + support::cpp11::to_string(output->info()->dimension(idx_height)));
    build_opts.add_option("-DBATCH_SIZE=" + support::cpp11::to_string(output->info()->dimension(idx_batch)));
    build_opts.add_option("-DWIDTH_IN=" + support::cpp11::to_string(input->info()->dimension(idx_width)));
    build_opts.add_option("-DHEIGHT_IN=" + support::cpp11::to_string(input->info()->dimension(idx_height)));
    build_opts.add_option("-DBATCH_IN=" + support::cpp11::to_string(input->info()->dimension(idx_batch)));
    _kernel = create_kernel(compile_context, "space_to_batch_" + lower_string(string_from_data_layout(input->info()->data_layout())), build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    ICLKernel::configure_internal(win);
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

void CLSpaceToBatchLayerKernel::configure(const ICLTensor *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right,
                                          ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, block_shape_x, block_shape_y, padding_left, padding_right, output);
}

void CLSpaceToBatchLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left,
                                          const Size2D &padding_right,
                                          ICLTensor    *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    TensorShape output_shape = misc::shape_calculator::compute_space_to_batch_shape(input->info(), block_shape_x, block_shape_y, padding_left, padding_right);
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->quantization_info());

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_static(input->info(), block_shape_x, block_shape_y, padding_left, padding_right, output->info()));

    _input  = input;
    _output = output;

    const DataLayout data_layout = input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_batch   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(data_size_from_type(input->info()->data_type())));
    build_opts.add_option("-DWIDTH_OUT=" + support::cpp11::to_string(output->info()->dimension(idx_width)));
    build_opts.add_option("-DHEIGHT_OUT=" + support::cpp11::to_string(output->info()->dimension(idx_height)));
    build_opts.add_option("-DBATCH_SIZE=" + support::cpp11::to_string(output->info()->dimension(idx_batch)));
    build_opts.add_option("-DWIDTH_IN=" + support::cpp11::to_string(input->info()->dimension(idx_width)));
    build_opts.add_option("-DHEIGHT_IN=" + support::cpp11::to_string(input->info()->dimension(idx_height)));
    build_opts.add_option("-DBATCH_IN=" + support::cpp11::to_string(input->info()->dimension(idx_batch)));
    build_opts.add_option("-DBLOCK_SHAPE_X=" + support::cpp11::to_string(block_shape_x));
    build_opts.add_option("-DBLOCK_SHAPE_Y=" + support::cpp11::to_string(block_shape_y));
    build_opts.add_option("-DPAD_LEFT_X=" + support::cpp11::to_string(padding_left.x()));
    build_opts.add_option("-DPAD_RIGHT_X=" + support::cpp11::to_string(padding_right.x()));
    build_opts.add_option("-DPAD_LEFT_Y=" + support::cpp11::to_string(padding_left.y()));
    build_opts.add_option("-DPAD_RIGHT_Y=" + support::cpp11::to_string(padding_right.y()));
    _kernel = create_kernel(compile_context, "space_to_batch_static_" + lower_string(string_from_data_layout(input->info()->data_layout())), build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    ICLKernel::configure_internal(win);
}

Status CLSpaceToBatchLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *paddings, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, block_shape, paddings, output));
    return Status{};
}
Status CLSpaceToBatchLayerKernel::validate(const ITensorInfo *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right,
                                           const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_static(input, block_shape_x, block_shape_y, padding_left, padding_right, output));
    return Status{};
}

void CLSpaceToBatchLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice_out = window.first_slice_window_3D();

    Window slice_in = window.first_slice_window_4D();
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
    slice_in.set(3, Window::Dimension(0, 0, 0));

    Window vector_slice = window.first_slice_window_1D();
    vector_slice.set(Window::DimX, Window::Dimension(0, 0, 0));

    Window padding_slice = window.first_slice_window_2D();
    padding_slice.set(Window::DimX, Window::Dimension(0, 0, 0));
    padding_slice.set(Window::DimY, Window::Dimension(0, 0, 0));

    int batch_id = 0;
    do
    {
        unsigned int idx  = 0;
        const bool   cond = (_paddings != nullptr && _block_shape != nullptr);
        add_4D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument_if(cond, idx, _paddings, padding_slice);
        add_1D_tensor_argument_if(cond, idx, _block_shape, vector_slice);

        add_argument(idx, batch_id);
        add_3D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out, lws_hint());
        ++batch_id;
    }
    while(window.slide_window_slice_3D(slice_out));
}
} // namespace arm_compute
