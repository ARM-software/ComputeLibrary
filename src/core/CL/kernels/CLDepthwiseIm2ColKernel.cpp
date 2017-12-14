/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
#include <tuple>

using namespace arm_compute;

CLDepthwiseIm2ColKernel::CLDepthwiseIm2ColKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLDepthwiseIm2ColKernel::configure(const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) != output->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(0) != (kernel_dims.width * kernel_dims.height + ((has_bias) ? 1 : 0)));

    _input  = input;
    _output = output;

    // Create kernel
    std::set<std::string> build_opts;

    build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.emplace("-DSTRIDE_X=" + support::cpp11::to_string(conv_info.stride().first));
    build_opts.emplace("-DSTRIDE_Y=" + support::cpp11::to_string(conv_info.stride().second));
    build_opts.emplace("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
    build_opts.emplace("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
    build_opts.emplace("-DPAD_RIGHT=" + support::cpp11::to_string(conv_info.pad_right()));
    build_opts.emplace("-DPAD_BOTTOM=" + support::cpp11::to_string(conv_info.pad_bottom()));
    build_opts.emplace("-DSRC_WIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.emplace("-DSRC_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(1)));
    build_opts.emplace("-DKERNEL_WIDTH=" + support::cpp11::to_string(kernel_dims.width));
    build_opts.emplace("-DKERNEL_HEIGHT=" + support::cpp11::to_string(kernel_dims.height));
    if(has_bias)
    {
        build_opts.emplace("-DHAS_BIAS");
    }
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("depthwise_im2col", build_opts));

    // Configure the local work size for Bifrost with a value obtained
    // via exhaustive autotuning for the MobileNets tensor shapes.
    const GPUTarget gpu_target = get_arch_from_target(get_target());
    if(gpu_target == GPUTarget::BIFROST)
    {
        _lws_hint = cl::NDRange(1, 2, 1);
    }

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    // The CLDepthwiseIm2ColKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
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
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice) && window.slide_window_slice_3D(slice_in));
}
