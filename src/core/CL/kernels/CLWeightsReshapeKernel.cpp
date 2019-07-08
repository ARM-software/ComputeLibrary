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
#include "arm_compute/core/CL/kernels/CLWeightsReshapeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *biases, const ITensorInfo *output, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(num_groups == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::NHWC && num_groups > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4 && num_groups > 1);
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(3) % num_groups) != 0);

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(input->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 4) && (biases->num_dimensions() != 1));
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 5) && (biases->num_dimensions() != 2));
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 4) && (biases->dimension(0) != input->tensor_shape()[3]));
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 5) && (biases->dimension(0) != input->tensor_shape()[3] || biases->dimension(1) != input->tensor_shape()[4]));
    }

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_weights_reshaped_shape(*input, biases != nullptr, num_groups));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

CLWeightsReshapeKernel::CLWeightsReshapeKernel()
    : _input(nullptr), _biases(nullptr), _output(nullptr)
{
}

void CLWeightsReshapeKernel::configure(const ICLTensor *input, const ICLTensor *biases, ICLTensor *output, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(compute_weights_reshaped_shape(*input->info(), (biases != nullptr), num_groups)));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(),
                                                  (biases != nullptr) ? biases->info() : nullptr,
                                                  output->info(), num_groups));

    const DataType data_type = input->info()->data_type();

    _biases = biases;
    _output = output;
    _input  = input;

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DNUM_GROUPS=" + support::cpp11::to_string(num_groups));
    build_opts.add_option_if(biases != nullptr, "-DHAS_BIAS");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("reshape_to_columns", build_opts.options()));

    // Configure window
    Window win = calculate_max_window(*input->info(), Steps());
    // The CLWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));
    ICLKernel::configure_internal(win);
}

Status CLWeightsReshapeKernel::validate(const ITensorInfo *input, const ITensorInfo *biases, const ITensorInfo *output, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, biases, output, num_groups));
    return Status{};
}

void CLWeightsReshapeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window out_window;
    out_window.use_tensor_dimensions(_output->info()->tensor_shape());

    Window in_slice  = window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_2D();

    Window biases_window;
    Window biases_slice;

    unsigned int idx = num_arguments_per_3D_tensor() + num_arguments_per_2D_tensor();
    idx += (_biases != nullptr) ? num_arguments_per_1D_tensor() : 0;
    _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(0));
    _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(1));
    _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(2));
    _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(3));
    _kernel.setArg<cl_uint>(idx++, _output->info()->strides_in_bytes().z());

    if(_biases != nullptr)
    {
        biases_window.use_tensor_dimensions(_biases->info()->tensor_shape());
        biases_slice = biases_window.first_slice_window_1D();
    }

    do
    {
        // Set arguments
        unsigned idx = 0;
        add_3D_tensor_argument(idx, _input, in_slice);
        add_2D_tensor_argument(idx, _output, out_slice);
        if(_biases != nullptr)
        {
            add_1D_tensor_argument(idx, _biases, biases_slice);
            ARM_COMPUTE_UNUSED(biases_window.slide_window_slice_1D(biases_slice));
        }

        // Run kernel
        enqueue(queue, *this, in_slice);
    }
    while(window.slide_window_slice_4D(in_slice) && out_window.slide_window_slice_2D(out_slice));
}
