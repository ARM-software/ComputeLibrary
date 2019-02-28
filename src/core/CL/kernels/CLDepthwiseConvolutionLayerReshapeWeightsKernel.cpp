/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayerReshapeWeightsKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const DepthwiseConvolutionReshapeInfo &info)
{
    const size_t idx_w = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON(info.c0 != 4);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_h) != 3);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_w) != 3);

    if(output->total_size() != 0)
    {
        auto reshaped_weights_shape = arm_compute::misc::shape_calculator::compute_reshaped_depthwise_weights_shape(*input, info);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), reshaped_weights_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const DepthwiseConvolutionReshapeInfo &info)
{
    auto reshaped_input_shape = arm_compute::misc::shape_calculator::compute_reshaped_depthwise_weights_shape(*input, info);
    auto_init_if_empty(*output, reshaped_input_shape, 1, input->data_type(), input->quantization_info());

    Window                 win = calculate_max_window(*input, Steps(info.c0));
    AccessWindowHorizontal weights_access(input, 0, info.c0);
    const bool             window_changed = update_window_and_padding(win, weights_access);

    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));
    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLDepthwiseConvolutionLayerReshapeWeightsKernel::CLDepthwiseConvolutionLayerReshapeWeightsKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLDepthwiseConvolutionLayerReshapeWeightsKernel::configure(const ICLTensor *input, ICLTensor *output, const DepthwiseConvolutionReshapeInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), info));
    auto win_config = validate_and_configure_window(input->info(), output->info(), info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    ICLKernel::configure_internal(win_config.second);

    _input  = input;
    _output = output;

    // Build the kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(_input->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(info.c0));
    build_opts.add_option("-DDST_WIDTH=" + support::cpp11::to_string(_output->info()->dimension(0)));
    build_opts.add_option_if(info.transpose, "-DTRANSPOSE");

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("depthwise_convolution_reshape_weights", build_opts.options()));
}

Status CLDepthwiseConvolutionLayerReshapeWeightsKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const DepthwiseConvolutionReshapeInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), info).first);
    return Status{};
}

void CLDepthwiseConvolutionLayerReshapeWeightsKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    unsigned int idx = 0;
    add_3D_tensor_argument(idx, _input, window);
    add_2D_tensor_argument(idx, _output, window);
    enqueue(queue, *this, window);
}
} // namespace arm_compute
