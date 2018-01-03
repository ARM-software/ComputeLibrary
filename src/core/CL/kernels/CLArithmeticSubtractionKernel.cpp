/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLArithmeticSubtractionKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8, DataType::QS8, DataType::QS16, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::U8, DataType::QS8, DataType::QS16, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input1, input2);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input1, input2);

    // Validate in case of configured output
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::QS8, DataType::QS16, DataType::S16, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->data_type() == DataType::U8 && (input1->data_type() != DataType::U8 || input2->data_type() != DataType::U8),
                                        "Output can only be U8 if both inputs are U8");
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input1, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input1, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output)
{
    constexpr unsigned int num_elems_processed_per_iteration = 16;

    Window                 win = calculate_max_window(*input1, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input1_access(input1, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal input2_access(input2, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input1_access, input2_access, output_access);

    ValidRegion valid_region = intersect_valid_regions(input1->valid_region(),
                                                       input2->valid_region());

    output_access.set_valid_region(win, valid_region);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLArithmeticSubtractionKernel::CLArithmeticSubtractionKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void CLArithmeticSubtractionKernel::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    // Auto initialize output if not initialized
    {
        set_shape_if_empty(*output->info(), input1->info()->tensor_shape());

        if(input1->info()->data_type() == DataType::S16 || input2->info()->data_type() == DataType::S16)
        {
            set_format_if_unknown(*output->info(), Format::S16);
        }
        else if(input1->info()->data_type() == DataType::F32 || input2->info()->data_type() == DataType::F32)
        {
            set_format_if_unknown(*output->info(), Format::F32);
        }
    }

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input1->info(), input2->info(), output->info(), policy));

    _input1 = input1;
    _input2 = input2;
    _output = output;

    bool has_float_out = is_data_type_float(output->info()->data_type());

    // Set kernel build options
    std::set<std::string> build_opts;
    build_opts.emplace((policy == ConvertPolicy::WRAP || has_float_out) ? "-DWRAP" : "-DSATURATE");
    build_opts.emplace("-DDATA_TYPE_IN1=" + get_cl_type_from_data_type(input1->info()->data_type()));
    build_opts.emplace("-DDATA_TYPE_IN2=" + get_cl_type_from_data_type(input2->info()->data_type()));
    build_opts.emplace("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output->info()->data_type()));
    if(is_data_type_fixed_point(input1->info()->data_type()))
    {
        build_opts.emplace("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input1->info()->fixed_point_position()));
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("arithmetic_sub", build_opts));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input1->info(), input2->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);
}

Status CLArithmeticSubtractionKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input1, input2, output, policy));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input1->clone().get(), input2->clone().get(), output->clone().get()).first);

    return Status{};
}

void CLArithmeticSubtractionKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input1, slice);
        add_3D_tensor_argument(idx, _input2, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(collapsed.slide_window_slice_3D(slice));
}
