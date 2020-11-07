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
#include "src/core/CL/kernels/CLDeconvolutionLayerUpsampleKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
CLDeconvolutionLayerUpsampleKernel::CLDeconvolutionLayerUpsampleKernel()
    : _input(nullptr), _output(nullptr), _info(), _data_layout(DataLayout::UNKNOWN)
{
}

Status CLDeconvolutionLayerUpsampleKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                                    const PadStrideInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);

    const DataLayout data_layout = input->data_layout();

    const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const size_t idx_c = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(idx_w) == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(idx_h) == 0);

    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_c) != output->dimension(idx_c));
    for(size_t i = 3; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(i) != output->dimension(i));
    }

    return Status{};
}

void CLDeconvolutionLayerUpsampleKernel::configure(const ICLTensor *input, ICLTensor *output,
                                                   const PadStrideInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, info);
}

void CLDeconvolutionLayerUpsampleKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output,
                                                   const PadStrideInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(CLDeconvolutionLayerUpsampleKernel::validate(input->info(), output->info(), info));

    _input       = input;
    _output      = output;
    _info        = info;
    _data_layout = input->info()->data_layout();

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option(("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(input->info()->element_size())));
    _kernel = create_kernel(compile_context, "deconvolution_upsample", build_opts.options());

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    // Configure kernel window
    Window                 win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure_internal(win);
}

void CLDeconvolutionLayerUpsampleKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const size_t idx_w = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

    const int out_start_x = _info.pad_left();
    const int out_end_x   = _output->info()->dimension(idx_w) - _info.pad_right() + _info.stride().first - 1;
    const int out_step_x  = _info.stride().first;

    const int out_start_y = _info.pad_top();
    const int out_end_y   = _output->info()->dimension(idx_h) - _info.pad_bottom() + _info.stride().second - 1;
    const int out_step_y  = _info.stride().second;

    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

            Window slice_out = collapsed.first_slice_window_3D();
            slice_out.set(Window::DimX, Window::Dimension(out_start_x, out_end_x, out_step_x));
            slice_out.set(Window::DimY, Window::Dimension(out_start_y, out_end_y, out_step_y));

            Window slice_in = collapsed.first_slice_window_3D();

            do
            {
                unsigned int idx = 0;
                add_3D_tensor_argument(idx, _input, slice_in);
                add_3D_tensor_argument(idx, _output, slice_out);
                enqueue(queue, *this, slice_out, lws_hint());
            }
            while(collapsed.slide_window_slice_3D(slice_in) && collapsed.slide_window_slice_3D(slice_out));
            break;
        }
        case DataLayout::NHWC:
        {
            // NOTE: not collapsing in NHWC
            Window slice_out = window.first_slice_window_3D();
            slice_out.set(Window::DimY, Window::Dimension(out_start_x, out_end_x, out_step_x));
            slice_out.set(Window::DimZ, Window::Dimension(out_start_y, out_end_y, out_step_y));

            Window slice_in = window.first_slice_window_3D();

            do
            {
                unsigned int idx = 0;
                add_3D_tensor_argument(idx, _input, slice_in);
                add_3D_tensor_argument(idx, _output, slice_out);
                enqueue(queue, *this, slice_out, lws_hint());
            }
            while(window.slide_window_slice_3D(slice_in) && window.slide_window_slice_3D(slice_out));
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported data layout");
    }
}
} // namespace arm_compute
