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
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <cmath>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, int mult_transpose1xW_width)
{
    ARM_COMPUTE_RETURN_ERROR_ON(mult_transpose1xW_width < 1);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::U8, DataType::S8,
                                                         DataType::U16, DataType::S16, DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(),
                                                           compute_transpose1xW_with_element_size_shape(*input, mult_transpose1xW_width));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, unsigned int &num_elems_processed_per_iteration, int mult_transpose1xW_width)
{
    num_elems_processed_per_iteration = 16 / input->element_size();

    const int scale_x        = num_elems_processed_per_iteration * mult_transpose1xW_width;
    bool      window_changed = false;

    // Configure kernel window
    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_transpose1xW_with_element_size_shape(*input, mult_transpose1xW_width)));

    // Configure window in case of configured output
    AccessWindowStatic output_access(output, 0, 0, ceil_to_multiple(output->dimension(0), scale_x), output->dimension(1));
    window_changed = window_changed || update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), input->tensor_shape()));

    // Collapse along the Z direction
    Window collapsed = win.collapse(win, Window::DimZ);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, collapsed);
}
} // namespace

void CLGEMMTranspose1xWKernel::configure(const ICLTensor *input, ICLTensor *output, int mult_transpose1xW_width)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(compute_transpose1xW_with_element_size_shape(*input->info(), mult_transpose1xW_width)));

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), mult_transpose1xW_width));

    _input  = input;
    _output = output;

    // Configure kernel window
    // Note: num_elems_processed_per_iteration will be set in validate_and_configure_window()
    unsigned int num_elems_processed_per_iteration = 1;
    auto         win_config                        = validate_and_configure_window(input->info(), output->info(), num_elems_processed_per_iteration, mult_transpose1xW_width);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(input->info()->element_size()));
    build_opts.add_option("-DTRANSPOSE_W=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DMULT_TRANSPOSE1XW_WIDTH=" + support::cpp11::to_string(mult_transpose1xW_width));

    /*
     * Following an example of how the transposition1xW works when the input data type is F32
     *
     *         |a00 a01 a02 a03|
     *         |a10 a11 a12 a13|
     *         |a20 a21 a22 a23| = | a00 a01 a02 a03 || a10 a11 a12 a13 || a20 a21 a22 a23 || a30 a31 a32 a33 |
     *         |a30 a31 a32 a33|
     *
     * The output matrix will have the following shape: [ height * W, ceil(width / W) ], where W = (16 / element size of the tensor) * mult_transpose1xW_width
     */
    // Create kernel
    std::string kernel_name = "gemm_transpose1xW";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));
}

Status CLGEMMTranspose1xWKernel::validate(const ITensorInfo *input, const ITensorInfo *output, int mult_transpose1xW_width)
{
    unsigned int num_elems_processed_per_iteration = 1;
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, mult_transpose1xW_width));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), num_elems_processed_per_iteration, mult_transpose1xW_width).first);

    return Status{};
}

void CLGEMMTranspose1xWKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    // Output is transposed
    Window out_window(window);
    out_window.set(Window::DimX, window.y());
    out_window.set(Window::DimY, window.x());

    Window in_slice  = window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, in_slice);
        add_3D_tensor_argument(idx, _output, out_slice);
        enqueue(queue, *this, in_slice, lws_hint());
    }
    while(window.slide_window_slice_3D(in_slice) && out_window.slide_window_slice_3D(out_slice));
}
