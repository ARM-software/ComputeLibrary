/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#include "arm_compute/core/GLES_COMPUTE/kernels/GCDropoutLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <random>
#include <tuple>

using namespace arm_compute;

GCDropoutLayerKernel::GCDropoutLayerKernel()
    : _input(nullptr), _mask(nullptr), _output(nullptr), _num_elems_processed_per_iteration(0)
{
}

void GCDropoutLayerKernel::configure(const IGCTensor *input, IGCTensor *mask, IGCTensor *output, float ratio, bool forward)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, mask, output);

    _input  = input;
    _mask   = mask;
    _output = output;

    std::set<std::string>                 build_opts;
    std::string                           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    std::string                           fporbp  = forward ? "FORWARD" : "BACKWARD";
    std::random_device                    rd;
    std::mt19937                          mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.emplace("#define RATIO " + support::cpp11::to_string(ratio));
    build_opts.emplace("#define SCALE " + support::cpp11::to_string(1. / (1. - ratio)));
    build_opts.emplace("#define SEED " + support::cpp11::to_string(dist(mt)));
    build_opts.emplace("#define " + dt_name);
    build_opts.emplace("#define " + fporbp);

    _num_elems_processed_per_iteration = 4 / input->info()->element_size();

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("dropout", build_opts));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(_num_elems_processed_per_iteration));

    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    IGCKernel::configure(win);
}

void GCDropoutLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IGCKernel::window(), window);

    _kernel.use();

    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;

        add_3D_tensor_argument(idx, _input, 1, slice);
        add_3D_tensor_argument(idx, _mask, 2, slice);
        add_3D_tensor_argument(idx, _output, 3, slice);

        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(window.slide_window_slice_3D(slice));
}
