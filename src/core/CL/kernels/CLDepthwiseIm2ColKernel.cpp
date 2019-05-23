/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDepthwiseIm2ColKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

#include <tuple>

using namespace arm_compute;

CLDepthwiseIm2ColKernel::CLDepthwiseIm2ColKernel()
    : _input(nullptr), _output(nullptr)
{
}

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, unsigned int depth_multiplier,
                          const Size2D &dilation)
{
    const size_t idx_c = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_UNUSED(conv_info);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(input->data_type()) && has_bias);
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(idx_c) * depth_multiplier) != output->dimension(2));
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(0) != (kernel_dims.width * kernel_dims.height + ((has_bias) ? 1 : 0)));
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || dilation.y() < 1);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);

    return Status{};
}
} // namespace

void CLDepthwiseIm2ColKernel::configure(const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, unsigned int depth_multiplier,
                                        const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), kernel_dims, conv_info, has_bias, depth_multiplier, dilation));

    _input  = input;
    _output = output;

    const DataLayout data_layout = input->info()->data_layout();
    const size_t     idx_w       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t     idx_h       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Create kernel
    CLBuildOptions build_opts;

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_info.stride().first));
    build_opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_info.stride().second));
    build_opts.add_option("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
    build_opts.add_option("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
    build_opts.add_option("-DPAD_RIGHT=" + support::cpp11::to_string(conv_info.pad_right()));
    build_opts.add_option("-DPAD_BOTTOM=" + support::cpp11::to_string(conv_info.pad_bottom()));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input->info()->dimension(idx_w)));
    build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(idx_h)));
    build_opts.add_option("-DKERNEL_WIDTH=" + support::cpp11::to_string(kernel_dims.width));
    build_opts.add_option("-DKERNEL_HEIGHT=" + support::cpp11::to_string(kernel_dims.height));
    build_opts.add_option("-DDEPTH_MULTIPLIER=" + support::cpp11::to_string(depth_multiplier));
    build_opts.add_option("-DDILATION_X=" + support::cpp11::to_string(dilation.x()));
    build_opts.add_option("-DDILATION_Y=" + support::cpp11::to_string(dilation.y()));
    build_opts.add_option("-D" + string_from_data_layout(input->info()->data_layout()));
    build_opts.add_option_if(has_bias, "-DHAS_BIAS");
    build_opts.add_option_if_else(is_data_type_quantized_asymmetric(input->info()->data_type()),
                                  "-DPAD_VALUE=" + support::cpp11::to_string(input->info()->quantization_info().offset),
                                  "-DPAD_VALUE=0");

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("depthwise_im2col", build_opts.options()));

    // Configure  kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    // CLDepthwiseIm2ColKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure_internal(win);
}

Status CLDepthwiseIm2ColKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, unsigned int depth_multiplier,
                                         const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, kernel_dims, conv_info, has_bias, depth_multiplier, dilation));

    return Status{};
}

void CLDepthwiseIm2ColKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window slice    = window.first_slice_window_3D();
    Window slice_in = window.first_slice_window_3D();

    // Setup slice
    slice.set(Window::DimX, Window::Dimension(0, _output->info()->dimension(0), _output->info()->dimension(0)));
    slice.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    slice.set(Window::DimZ, Window::Dimension(0, _output->info()->dimension(2), 1));

    // Setup input slice
    // The first three dimensions of the input are increased by the inner loops
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice) && window.slide_window_slice_3D(slice_in));
}
