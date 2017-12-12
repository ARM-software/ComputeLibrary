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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCNormalizePlanarYUVLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

using namespace arm_compute;

GCNormalizePlanarYUVLayerKernel::GCNormalizePlanarYUVLayerKernel()
    : _input(nullptr), _output(nullptr), _mean(nullptr), _sd(nullptr)
{
}

void GCNormalizePlanarYUVLayerKernel::configure(const IGCTensor *input, IGCTensor *output, const IGCTensor *mean, const IGCTensor *sd)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, mean, sd);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(mean, sd);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) != mean->info()->dimension(0));

    _input  = input;
    _output = output;
    _mean   = mean;
    _sd     = sd;

    const unsigned int num_elems_processed_per_iteration = 4;

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("#define LOCAL_SIZE_X " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1)));

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("normalize_planar_yuv_layer", build_opts));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    const int              mean_padding = ceil_to_multiple(mean->info()->dimension(0), num_elems_processed_per_iteration) - mean->info()->dimension(0);
    const int              sd_padding   = ceil_to_multiple(sd->info()->dimension(0), num_elems_processed_per_iteration) - sd->info()->dimension(0);
    AccessWindowStatic     mean_access(mean->info(), 0, 0, mean->info()->dimension(0) + mean_padding, mean->info()->dimension(1));
    AccessWindowStatic     sd_access(sd->info(), 0, 0, sd->info()->dimension(0) + sd_padding, sd->info()->dimension(1));

    update_window_and_padding(win, input_access, output_access, mean_access, sd_access);
    output_access.set_valid_region(win, input->info()->valid_region());

    IGCKernel::configure(win);
}

void GCNormalizePlanarYUVLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    _kernel.use();

    Window slice = window.first_slice_window_3D();

    Window slice_in;
    //slice_in.use_tensor_dimensions(_mean->info()->tensor_shape());
    slice_in = window.first_slice_window_1D();
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));

    unsigned int idx = 2 * num_arguments_per_3D_tensor();
    add_1D_tensor_argument(idx, _mean, 3, slice_in);
    add_1D_tensor_argument(idx, _sd, 4, slice_in);

    do
    {
        idx = 0;
        add_3D_tensor_argument(idx, _input, 1, slice);
        add_3D_tensor_argument(idx, _output, 2, slice);

        _kernel.update_shader_params();

        enqueue(*this, slice);
    }
    while(window.slide_window_slice_3D(slice));
}
