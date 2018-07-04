/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLConvertFullyConnectedWeightsKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
CLConvertFullyConnectedWeightsKernel::CLConvertFullyConnectedWeightsKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLConvertFullyConnectedWeightsKernel::configure(const ICLTensor *input, ICLTensor *output, const TensorShape &original_input_shape,
                                                     DataLayout data_layout)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLConvertFullyConnectedWeightsKernel::validate(input->info(), output->info(), original_input_shape, data_layout));

    _input  = input;
    _output = output;

    const unsigned int num_elems_per_input_plane = original_input_shape.x() * original_input_shape.y();
    const unsigned int num_channels              = original_input_shape.z();

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    if(data_layout == DataLayout::NCHW)
    {
        build_opts.add_option("-DFACTOR_1=" + support::cpp11::to_string(num_elems_per_input_plane));
        build_opts.add_option("-DFACTOR_2=" + support::cpp11::to_string(num_channels));
    }
    else
    {
        build_opts.add_option("-DFACTOR_1=" + support::cpp11::to_string(num_channels));
        build_opts.add_option("-DFACTOR_2=" + support::cpp11::to_string(num_elems_per_input_plane));
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("convert_fc_weights", build_opts.options()));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    ICLKernel::configure(win);
}

Status CLConvertFullyConnectedWeightsKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const TensorShape &original_input_shape,
                                                      DataLayout data_layout)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16, DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(1) != original_input_shape.total_size_lower(3));
    ARM_COMPUTE_RETURN_ERROR_ON(data_layout == DataLayout::UNKNOWN);

    return Status{};
}

void CLConvertFullyConnectedWeightsKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    unsigned int idx = 0;
    add_2D_tensor_argument(idx, _input, window);
    add_2D_tensor_argument(idx, _output, window);
    enqueue(queue, *this, window);
}
} // namespace arm_compute