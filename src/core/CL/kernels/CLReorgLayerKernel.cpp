/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "src/core/CL/kernels/CLReorgLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

#include <string>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, int32_t stride)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);

    const size_t idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON(stride <= 0);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((input->tensor_shape()[idx_width] % stride) != 0,
                                    "The width of the input tensor must be a multiple of stride");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((input->tensor_shape()[idx_height] % stride) != 0,
                                    "The height of the input tensor must be a multiple of stride");

    // Validate output if initialized
    if (output->total_size() != 0)
    {
        const TensorInfo tensor_info_output =
            output->clone()->set_tensor_shape(misc::shape_calculator::compute_reorg_output_shape(*input, stride));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

CLReorgLayerKernel::CLReorgLayerKernel() : _input(nullptr), _output(nullptr)
{
    _type = CLKernelType::ELEMENTWISE;
}

void CLReorgLayerKernel::configure(const ICLTensor *input, ICLTensor *output, int32_t stride)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, stride);
}

void CLReorgLayerKernel::configure(const CLCompileContext &compile_context,
                                   const ICLTensor        *input,
                                   ICLTensor              *output,
                                   int32_t                 stride)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), stride));
    auto padding_info = get_padding_info({input, output});

    _input  = input;
    _output = output;

    std::string kernel_name =
        std::string("reorg_layer_") + lower_string(string_from_data_layout(input->info()->data_layout()));
    const size_t idx_channel =
        get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::CHANNEL);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DSRC_DEPTH=" + support::cpp11::to_string(input->info()->dimension(idx_channel)));
    build_opts.add_option("-DSTRIDE=" + support::cpp11::to_string(stride));
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure window
    // auto inizialize the output tensor if not yet initialized
    auto_init_if_empty(*output->info(),
                       input->info()->clone()->set_tensor_shape(
                           misc::shape_calculator::compute_reorg_output_shape(*input->info(), stride)));

    Window win = calculate_max_window(*output->info(), Steps());

    // The CLWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    ICLKernel::configure_internal(win);

    _config_id = kernel_name;
    _config_id += "_";
    _config_id += string_from_data_type(input->info()->data_type());
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(stride);
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLReorgLayerKernel::validate(const arm_compute::ITensorInfo *input,
                                    const arm_compute::ITensorInfo *output,
                                    int32_t                         stride)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, stride));

    return Status{};
}

void CLReorgLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    } while (window.slide_window_slice_3D(slice));
}
} // namespace arm_compute
