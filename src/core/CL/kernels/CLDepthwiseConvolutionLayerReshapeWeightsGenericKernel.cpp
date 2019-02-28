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
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *biases)
{
    const size_t idx_w = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    const size_t idx_c = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(input->data_type()) && (biases != nullptr));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_c) != output->dimension(1));
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(0) != (input->dimension(idx_w) * input->dimension(idx_h) + ((biases != nullptr) ? 1 : 0)));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != input->dimension(idx_c));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    return Status{};
}
} // namespace

CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel::CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel()
    : _input(nullptr), _biases(nullptr), _output(nullptr)
{
}

void CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel::configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *biases)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), (biases != nullptr) ? biases->info() : nullptr));

    _input  = input;
    _biases = biases;
    _output = output;

    const size_t idx_w = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);

    // Create kernel
    std::set<std::string> build_opts;

    build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.emplace("-DSRC_WIDTH=" + support::cpp11::to_string(input->info()->dimension(idx_w)));
    build_opts.emplace("-D" + string_from_data_layout(input->info()->data_layout()));
    if(_biases != nullptr)
    {
        build_opts.emplace("-DHAS_BIAS");
    }

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("depthwise_convolution_reshape_weights_generic", build_opts));

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    // The CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure_internal(win);
}

Status CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *biases)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, biases));
    return Status{};
}

void CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window slice     = window.first_slice_window_3D();
    Window slice_out = window.first_slice_window_2D();

    const size_t idx_w = get_data_layout_dimension_index(_input->info()->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(_input->info()->data_layout(), DataLayoutDimension::HEIGHT);
    const size_t idx_c = get_data_layout_dimension_index(_input->info()->data_layout(), DataLayoutDimension::CHANNEL);

    // Setup slice
    slice.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(idx_w), _input->info()->dimension(idx_w)));
    slice.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(idx_h), 1));
    slice.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(idx_c), 1));

    // Setup output slice
    // The first two dimensions of the output are increased by the inner loops
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    // Set biases
    if(_biases != nullptr)
    {
        unsigned int idx = num_arguments_per_3D_tensor() + num_arguments_per_2D_tensor();
        Window       slice_biases;
        slice_biases.use_tensor_dimensions(_biases->info()->tensor_shape());
        add_1D_tensor_argument(idx, _biases, slice_biases);
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_3D(slice) && window.slide_window_slice_2D(slice_out));
}
