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
#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, int mult_interleave4x4_height, bool reinterpret_input_as_3d)
{
    ARM_COMPUTE_RETURN_ERROR_ON(mult_interleave4x4_height < 1);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::U8, DataType::S8,
                                                         DataType::U16, DataType::S16, DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_interleaved_shape(*input, mult_interleave4x4_height, reinterpret_input_as_3d));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, int mult_interleave4x4_height, bool reinterpret_input_as_3d)
{
    constexpr unsigned int num_elems_processed_per_iteration_x = 4;
    constexpr unsigned int num_elems_processed_per_iteration_y = 4;
    const unsigned int     num_elems_written_per_iteration     = num_elems_processed_per_iteration_x * num_elems_processed_per_iteration_y * mult_interleave4x4_height;
    bool                   window_changed                      = false;

    TensorInfo tmp_info(*input);

    if(reinterpret_input_as_3d)
    {
        // Since the input tensor has to be reinterpreted as 3D and the execute window is based on a 2D interleave,
        // the window needs to be constructed on the 2D collapsed version of the tensor
        TensorShape tmp_shape(input->tensor_shape());
        tmp_shape.collapse(2U, 1U);
        tmp_info.set_tensor_shape(tmp_shape);
    }

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_interleaved_shape(*input, mult_interleave4x4_height)));

    // Configure window
    const float scale_x = 4.0f * static_cast<float>(mult_interleave4x4_height);
    const float scale_y = 1.0f / (scale_x);

    // Note: bottom paddings are calculated manually as the input can be reinterpreted as 3D tensor
    // The only way to set properly the paddings, it is to set those explicitly through the AccessWindowStatic
    const int m          = reinterpret_input_as_3d ? input->tensor_shape()[1] * input->tensor_shape()[2] : input->tensor_shape()[1];
    const int bottom_pad = (num_elems_processed_per_iteration_y - (m % num_elems_processed_per_iteration_y)) % num_elems_processed_per_iteration_y;

    Window win    = calculate_max_window(tmp_info, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    Window win_in = calculate_max_window(*input, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    AccessWindowStatic input_access(input, 0, 0,
                                    ceil_to_multiple(input->dimension(0), num_elems_processed_per_iteration_x),
                                    input->dimension(1) + bottom_pad);
    AccessWindowRectangle output_access(output, 0, 0, num_elems_written_per_iteration, 1, scale_x, scale_y);

    window_changed = update_window_and_padding(win_in, input_access) || // window used by the execute_window_loop
                     update_window_and_padding(win, output_access);     // window used to update the padding requirements of output tensor
    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->tensor_shape()));

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window collapsed = win.collapse(win, Window::DimZ);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, collapsed);
}
} // namespace

CLGEMMInterleave4x4Kernel::CLGEMMInterleave4x4Kernel()
    : _input(nullptr), _output(nullptr), _reinterpret_input_as_3d(false)
{
}

void CLGEMMInterleave4x4Kernel::configure(const ICLTensor *input, ICLTensor *output, int mult_interleave4x4_height, bool reinterpret_input_as_3d, bool unroll_block)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(compute_interleaved_shape(*input->info(), mult_interleave4x4_height, reinterpret_input_as_3d)));

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), mult_interleave4x4_height, reinterpret_input_as_3d));

    _input                   = input;
    _output                  = output;
    _reinterpret_input_as_3d = reinterpret_input_as_3d;

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DMULT_INTERLEAVE4X4_HEIGHT=" + support::cpp11::to_string(mult_interleave4x4_height));
    build_opts.add_option_if(unroll_block, "-DUNROLL_BLOCK");
    build_opts.add_option_if(_reinterpret_input_as_3d, "-DREINTERPRET_INPUT_AS_3D");
    build_opts.add_option_if(_reinterpret_input_as_3d, "-DHEIGHT_GEMM3D=" + support::cpp11::to_string(input->info()->dimension(1)));
    build_opts.add_option_if(_reinterpret_input_as_3d, "-DDEPTH_GEMM3D=" + support::cpp11::to_string(input->info()->dimension(2)));

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
    auto win_config = validate_and_configure_window(input->info(), output->info(), mult_interleave4x4_height, reinterpret_input_as_3d);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "interleave4x4_";
    _config_id += (_reinterpret_input_as_3d ? "3d_" : "");
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

Status CLGEMMInterleave4x4Kernel::validate(const ITensorInfo *input, const ITensorInfo *output, int mult_interleave4x4_height, bool reinterpret_input_as_3d)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, mult_interleave4x4_height, reinterpret_input_as_3d));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), mult_interleave4x4_height, reinterpret_input_as_3d).first);

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

    if(_reinterpret_input_as_3d)
    {
        // Pass bottom paddings to the kernel if the input has to be reinterpreted as 3D tensor
        const unsigned int idx0                  = 2 * num_arguments_per_3D_tensor();
        const unsigned int total_cross_plane_pad = _input->info()->padding().top + _input->info()->padding().bottom;
        _kernel.setArg<cl_uint>(idx0, static_cast<unsigned int>(total_cross_plane_pad));
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice));
}
