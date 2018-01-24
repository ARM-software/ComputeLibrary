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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCWeightsReshapeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"

using namespace arm_compute;
using namespace arm_compute::gles_compute;

GCWeightsReshapeKernel::GCWeightsReshapeKernel()
    : _input(nullptr), _biases(nullptr), _output(nullptr)
{
}

void GCWeightsReshapeKernel::configure(const IGCTensor *input, const IGCTensor *biases, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    // Calculate output shape
    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.collapse(3);
    const size_t tmp_dim = output_shape[0];
    output_shape.set(0, output_shape[1]);
    output_shape.set(1, tmp_dim + (biases != nullptr ? 1 : 0));

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 4) && (biases->info()->num_dimensions() != 1));
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 5) && (biases->info()->num_dimensions() != 2));
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 4) && (biases->info()->dimension(0) != input->info()->tensor_shape()[3]));
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 5) && (biases->info()->dimension(0) != input->info()->tensor_shape()[3] || biases->info()->dimension(1) != input->info()->tensor_shape()[4]));
    }

    _biases = biases;
    _output = output;
    _input  = input;

    // Create build options
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace("#define " + dt_name);
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.emplace("#define RESHAPE_TO_COLUMNS");
    if(biases != nullptr)
    {
        build_opts.emplace("#define HAS_BIAS");
    }

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("reshape_to_columns", build_opts));

    // Set static arguments
    unsigned int idx = num_arguments_per_3D_tensor() + num_arguments_per_2D_tensor();
    idx += (biases != nullptr) ? num_arguments_per_1D_tensor() : 0;
    _kernel.set_argument(idx++, _input->info()->dimension(0));
    _kernel.set_argument(idx++, _input->info()->dimension(1));
    _kernel.set_argument(idx++, _input->info()->dimension(2));
    _kernel.set_argument(idx++, _input->info()->dimension(3));

    // Configure window
    Window win = calculate_max_window(*input->info(), Steps());

    // The GCWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));
    IGCKernel::configure(win);
}

void GCWeightsReshapeKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IGCKernel::window(), window);

    Window out_window;
    out_window.use_tensor_dimensions(_output->info()->tensor_shape());

    Window in_slice  = window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_2D();

    Window biases_window;
    Window biases_slice;

    if(_biases != nullptr)
    {
        biases_window.use_tensor_dimensions(_biases->info()->tensor_shape());
        biases_slice = biases_window.first_slice_window_1D();
    }

    _kernel.use();

    do
    {
        // Set arguments
        unsigned idx = 0;
        add_3D_tensor_argument(idx, _input, 1, in_slice);
        add_2D_tensor_argument(idx, _output, 2, out_slice);
        if(_biases != nullptr)
        {
            add_1D_tensor_argument(idx, _biases, 3, biases_slice);
            biases_window.slide_window_slice_1D(biases_slice);
        }

        _kernel.update_shader_params();
        // Run kernel
        enqueue(*this, in_slice);
    }
    while(window.slide_window_slice_4D(in_slice) && out_window.slide_window_slice_2D(out_slice));
}
