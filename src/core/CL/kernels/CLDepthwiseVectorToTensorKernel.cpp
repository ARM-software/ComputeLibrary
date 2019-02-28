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
#include "arm_compute/core/CL/kernels/CLDepthwiseVectorToTensorKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, size_t conv_w, size_t conv_h)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::S32, DataType::F16, DataType::F32);

    if(output->total_size() != 0)
    {
        TensorShape output_shape = compute_vector_to_tensor_output_shape(input->tensor_shape(), conv_w, conv_h, output->data_layout());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

CLDepthwiseVectorToTensorKernel::CLDepthwiseVectorToTensorKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLDepthwiseVectorToTensorKernel::configure(const ICLTensor *input, ICLTensor *output, size_t conv_w, size_t conv_h)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto inizialitation if not yet initialized
    TensorShape output_shape = compute_vector_to_tensor_output_shape(input->info()->tensor_shape(), conv_w, conv_h, output->info()->data_layout());
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), conv_w, conv_h));

    _input  = input;
    _output = output;

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DCONV_WIDTH=" + support::cpp11::to_string(conv_w));
    build_opts.add_option("-DCONV_HEIGHT=" + support::cpp11::to_string(conv_h));
    build_opts.add_option("-D" + string_from_data_layout(output->info()->data_layout()));

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("depthwise_vector_to_tensor", build_opts.options()));

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    // The CLDepthwisevectorToTensorKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure_internal(win);
}

Status CLDepthwiseVectorToTensorKernel::validate(const ITensorInfo *input, const ITensorInfo *output, size_t conv_w, size_t conv_h)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, conv_w, conv_h));
    return Status{};
}

void CLDepthwiseVectorToTensorKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window slice     = window.first_slice_window_1D();
    Window slice_out = window.first_slice_window_3D();

    // Setup slice
    slice.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), 1));

    // Setup output slice
    // The first three dimensions of the output are increased by the inner loops
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    do
    {
        unsigned int idx = 0;
        add_1D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_1D(slice) && window.slide_window_slice_3D(slice_out));
}
