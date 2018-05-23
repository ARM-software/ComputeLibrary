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
#include "arm_compute/core/CL/kernels/CLWidthConcatenateLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

#include <map>

using namespace arm_compute;
namespace
{
std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, unsigned int width_offset, ITensorInfo *output)
{
    const unsigned int num_elems_processed_per_iteration = 16;

    // The window needs to be based on input as we copy all the widths of input
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, width_offset, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
Status validate_arguments(const ITensorInfo *input, unsigned int width_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QS8, DataType::QASYMM8, DataType::U16, DataType::S16, DataType::QS16, DataType::F16, DataType::U32,
                                                         DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) + width_offset > output->dimension(0));

    for(size_t i = 1; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(i) != output->dimension(i));
    }
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 3);

    return Status{};
}
} // namespace

CLWidthConcatenateLayerKernel::CLWidthConcatenateLayerKernel()
    : _input(nullptr), _output(nullptr), _width_offset(0)
{
}

Status CLWidthConcatenateLayerKernel::validate(const ITensorInfo *input, unsigned int width_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, width_offset, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), width_offset, output->clone().get()).first);
    return Status{};
}

void CLWidthConcatenateLayerKernel::configure(const ICLTensor *input, unsigned int width_offset, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), width_offset, output->info()));

    _input        = input;
    _output       = output;
    _width_offset = width_offset;

    const unsigned int num_elems_processed_per_iteration = 16;

    // Add build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_underlying_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("concatenate_width", build_opts.options()));

    const int offset_to_first_elements_in_bytes = _width_offset * _output->info()->strides_in_bytes()[0];

    unsigned int idx = 2 * num_arguments_per_3D_tensor(); // Skip the input and output parameters
    _kernel.setArg<cl_int>(idx, offset_to_first_elements_in_bytes);

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), width_offset, output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    ICLKernel::configure(std::get<1>(win_config));
}

void CLWidthConcatenateLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_3D(slice));
}
