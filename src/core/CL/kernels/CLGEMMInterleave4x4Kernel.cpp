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
#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, int mult_interleave4x4_height)
{
    ARM_COMPUTE_RETURN_ERROR_ON(mult_interleave4x4_height < 1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::U8, DataType::S8,
                                                         DataType::QS16, DataType::U16, DataType::S16, DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_interleaved_shape(*input, mult_interleave4x4_height));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, int mult_interleave4x4_height)
{
    constexpr unsigned int num_elems_processed_per_iteration_x = 4;
    constexpr unsigned int num_elems_processed_per_iteration_y = 4;
    const unsigned int     num_elems_written_per_iteration     = num_elems_processed_per_iteration_x * num_elems_processed_per_iteration_y * mult_interleave4x4_height;
    bool                   window_changed                      = false;

    // Configure kernel window
    Window                win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);
    window_changed = window_changed || update_window_and_padding(win, input_access);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_interleaved_shape(*input, mult_interleave4x4_height)));

    // Configure window
    const float scale_x = 4.0f * static_cast<float>(mult_interleave4x4_height);
    const float scale_y = 1.0f / (scale_x);

    AccessWindowRectangle output_access(output, 0, 0, num_elems_written_per_iteration, 1, scale_x, scale_y);
    window_changed = window_changed || update_window_and_padding(win, output_access);
    output_access.set_valid_region(win, input->valid_region());

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window collapsed = win.collapse(win, Window::DimZ);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, collapsed);
}
} // namespace

CLGEMMInterleave4x4Kernel::CLGEMMInterleave4x4Kernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLGEMMInterleave4x4Kernel::configure(const ICLTensor *input, ICLTensor *output, int mult_interleave4x4_height)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(compute_interleaved_shape(*input->info(), mult_interleave4x4_height)));

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), mult_interleave4x4_height));

    _input  = input;
    _output = output;

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DMULT_INTERLEAVE4X4_HEIGHT=" + support::cpp11::to_string(mult_interleave4x4_height));
    switch(input->info()->element_size())
    {
        case 1:
            build_opts.add_option("-DDATA_TYPE=uchar");
            break;
        case 2:
            build_opts.add_option("-DDATA_TYPE=ushort");
            break;
        case 4:
            build_opts.add_option("-DDATA_TYPE=uint");
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gemm_interleave4x4", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), mult_interleave4x4_height);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "interleave4x4_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(3));
}

Status CLGEMMInterleave4x4Kernel::validate(const ITensorInfo *input, const ITensorInfo *output, int mult_interleave4x4_height)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, mult_interleave4x4_height));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), mult_interleave4x4_height).first);

    return Status{};
}

void CLGEMMInterleave4x4Kernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    /*
     * This kernel puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
     *         |a00 a01 a02 a03|
     *         |a10 a11 a12 a13|
     *         |a20 a21 a22 a23| = | a00 a10 a20 a30 || a01 a11 a21 a31 || a02 a12 a22 a32 || a03 a13 a23 a33 |
     *         |a30 a31 a32 a33|
     *
     * After this operation, the output matrix will have the following shape: [ height * 4, width / 4 ]
     */
    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice));
}
