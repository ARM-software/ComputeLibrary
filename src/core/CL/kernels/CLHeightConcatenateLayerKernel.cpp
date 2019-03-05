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
#include "arm_compute/core/CL/kernels/CLHeightConcatenateLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

#include <map>

using namespace arm_compute;
namespace
{
std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, unsigned int height_offset, ITensorInfo *output, unsigned int &num_elems_processed_per_iteration)
{
    num_elems_processed_per_iteration = 4;
    // The window needs to be based on input as we copy all the heights of input
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, height_offset, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);

    Window win_collapsed = win.collapse(win, Window::DimZ);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win_collapsed);
}
Status validate_arguments(const ITensorInfo *input, unsigned int height_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16, DataType::F16, DataType::U32,
                                                         DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimY) + height_offset > output->dimension(Window::DimY));

    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != output->dimension(0));
    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(i) != output->dimension(i));
    }
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

    return Status{};
}
} // namespace

CLHeightConcatenateLayerKernel::CLHeightConcatenateLayerKernel()
    : _input(nullptr), _output(nullptr), _height_offset(0), _num_elems_processed_per_iteration()
{
}

Status CLHeightConcatenateLayerKernel::validate(const ITensorInfo *input, unsigned int height_offset, const ITensorInfo *output)
{
    unsigned int num_elems_processed_per_iteration;
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, height_offset, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), height_offset, output->clone().get(), num_elems_processed_per_iteration).first);
    return Status{};
}

void CLHeightConcatenateLayerKernel::configure(const ICLTensor *input, unsigned int height_offset, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), height_offset, output->info()));

    _input         = input;
    _output        = output;
    _height_offset = height_offset;

    auto win_config = validate_and_configure_window(input->info(), height_offset, output->info(), _num_elems_processed_per_iteration);

    // Add build options
    CLBuildOptions build_opts;

    switch(input->info()->element_size())
    {
        case 1:
        {
            build_opts.add_option("-DDATA_TYPE=uchar");
            break;
        }
        case 2:
        {
            build_opts.add_option("-DDATA_TYPE=short");
            break;
        }
        case 4:
        {
            build_opts.add_option("-DDATA_TYPE=int");
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported input data type.");
            break;
        }
    }

    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(_num_elems_processed_per_iteration));
    build_opts.add_option("-DHEIGHT_OFFSET=" + support::cpp11::to_string(_height_offset));
    build_opts.add_option("-DDEPTH=" + support::cpp11::to_string(input->info()->dimension(2)));

    if(is_data_type_quantized_asymmetric(input->info()->data_type()) && input->info()->quantization_info() != output->info()->quantization_info())
    {
        build_opts.add_option("-DOFFSET_IN1=" + float_to_string_with_full_precision(input->info()->quantization_info().offset));
        build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(output->info()->quantization_info().offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(input->info()->quantization_info().scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(output->info()->quantization_info().scale));
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("concatenate_height", build_opts.options()));
    // Configure kernel window

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    ICLKernel::configure_internal(std::get<1>(win_config));
}

void CLHeightConcatenateLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    unsigned int idx = 0;
    add_4D_tensor_argument(idx, _input, window);
    add_4D_tensor_argument(idx, _output, window);
    enqueue(queue, *this, window);
}
