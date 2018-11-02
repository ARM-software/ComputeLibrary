/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLArithmeticAdditionKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"

using namespace arm_compute;

namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;

Status validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&input2);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input2, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::F16, DataType::F32);

    const bool is_qasymm = is_data_type_quantized_asymmetric(input1.data_type()) || is_data_type_quantized_asymmetric(input2.data_type());
    if(is_qasymm)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &input2);
    }

    const TensorShape out_shape = TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&output);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((output.data_type() == DataType::U8) && ((input1.data_type() != DataType::U8) || (input2.data_type() != DataType::U8)),
                                        "Output can only be U8 if both inputs are U8");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
                                        "Wrong shape for output");
        if(is_qasymm)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &output);
        }
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(input1, input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    {
        set_shape_if_empty(output, out_shape);

        if(input1.data_type() == DataType::S16 || input2.data_type() == DataType::S16)
        {
            set_format_if_unknown(output, Format::S16);
        }
        else if(input1.data_type() == DataType::F16 && input2.data_type() == DataType::F16)
        {
            set_format_if_unknown(output, Format::F16);
        }
        else if(input1.data_type() == DataType::F32 || input2.data_type() == DataType::F32)
        {
            set_format_if_unknown(output, Format::F32);
        }
    }

    Window win        = calculate_max_window(valid_region, Steps(num_elems_processed_per_iteration));
    Window win_input1 = win.broadcast_if_dimension_le_one(input1);
    Window win_input2 = win.broadcast_if_dimension_le_one(input2);

    AccessWindowHorizontal input1_access(&input1, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal input2_access(&input2, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(&output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win_input1, input1_access)
                          || update_window_and_padding(win_input2, input2_access)
                          || update_window_and_padding(win, output_access);

    output_access.set_valid_region(win, valid_region);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLArithmeticAdditionKernel::CLArithmeticAdditionKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void CLArithmeticAdditionKernel::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1->info(), *input2->info(), *output->info(), policy));

    // Configure kernel window
    auto win_config = validate_and_configure_window(*input1->info(), *input2->info(), *output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    _input1 = input1;
    _input2 = input2;
    _output = output;

    const bool has_float_out = is_data_type_float(output->info()->data_type());

    std::string kernel_name = "arithmetic_add";

    // Set kernel build options
    std::set<std::string> build_opts;
    build_opts.emplace((policy == ConvertPolicy::WRAP || has_float_out) ? "-DWRAP" : "-DSATURATE");
    build_opts.emplace("-DDATA_TYPE_IN1=" + get_cl_type_from_data_type(input1->info()->data_type()));
    build_opts.emplace("-DDATA_TYPE_IN2=" + get_cl_type_from_data_type(input2->info()->data_type()));
    build_opts.emplace("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output->info()->data_type()));
    if(is_data_type_quantized_asymmetric(input1->info()->data_type()))
    {
        build_opts.emplace("-DOFFSET_IN1=" + support::cpp11::to_string(input1->info()->quantization_info().offset));
        build_opts.emplace("-DOFFSET_IN2=" + support::cpp11::to_string(input2->info()->quantization_info().offset));
        build_opts.emplace("-DOFFSET_OUT=" + support::cpp11::to_string(output->info()->quantization_info().offset));
        build_opts.emplace("-DSCALE_IN1=" + support::cpp11::to_string(input1->info()->quantization_info().scale));
        build_opts.emplace("-DSCALE_IN2=" + support::cpp11::to_string(input2->info()->quantization_info().scale));
        build_opts.emplace("-DSCALE_OUT=" + support::cpp11::to_string(output->info()->quantization_info().scale));
        kernel_name += "_quantized";
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    ICLKernel::configure_internal(win_config.second);
}

Status CLArithmeticAdditionKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output, policy));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*input1->clone(), *input2->clone(), *output->clone()).first);

    return Status{};
}

void CLArithmeticAdditionKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const TensorShape &in_shape1 = _input1->info()->tensor_shape();
    const TensorShape &in_shape2 = _input2->info()->tensor_shape();
    const TensorShape &out_shape = _output->info()->tensor_shape();

    bool can_collapse = true;
    if(std::min(in_shape1.total_size(), in_shape2.total_size()) > 1)
    {
        can_collapse = (std::min(in_shape1.num_dimensions(), in_shape2.num_dimensions()) > Window::DimZ);
        for(size_t d = Window::DimZ; can_collapse && (d < out_shape.num_dimensions()); d++)
        {
            can_collapse = (in_shape1[d] == in_shape2[d]);
        }
    }

    bool   has_collapsed = false;
    Window collapsed     = can_collapse ? window.collapse_if_possible(ICLKernel::window(), Window::DimZ, &has_collapsed) : window;

    const TensorShape &in_shape1_collapsed = has_collapsed ? in_shape1.collapsed_from(Window::DimZ) : in_shape1;
    const TensorShape &in_shape2_collapsed = has_collapsed ? in_shape2.collapsed_from(Window::DimZ) : in_shape2;

    Window slice        = collapsed.first_slice_window_3D();
    Window slice_input1 = slice.broadcast_if_dimension_le_one(in_shape1_collapsed);
    Window slice_input2 = slice.broadcast_if_dimension_le_one(in_shape2_collapsed);

    do
    {
        unsigned int idx = 0;

        add_3D_tensor_argument(idx, _input1, slice_input1);
        add_3D_tensor_argument(idx, _input2, slice_input2);
        add_3D_tensor_argument(idx, _output, slice);

        enqueue(queue, *this, slice);

        collapsed.slide_window_slice_3D(slice_input1);
        collapsed.slide_window_slice_3D(slice_input2);
    }
    while(collapsed.slide_window_slice_3D(slice));
}

BorderSize CLArithmeticAdditionKernel::border_size() const
{
    const unsigned int replicateSize = _output->info()->dimension(0) - std::min(_input1->info()->dimension(0), _input2->info()->dimension(0));
    const unsigned int border        = std::min<unsigned int>(num_elems_processed_per_iteration - 1U, replicateSize);
    return BorderSize(0, border, 0, 0);
}
