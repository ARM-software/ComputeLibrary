/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLComparisonKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"

#include <map>

namespace arm_compute
{
namespace
{
// Create supported comparisons map
const std::map<ComparisonOperation, std::string> supported_comparison_ops =
{
    { ComparisonOperation::Equal, "EQUAL" },
    { ComparisonOperation::NotEqual, "NOTEQUAL" },
    { ComparisonOperation::Greater, "GREATER" },
    { ComparisonOperation::GreaterEqual, "GREATEREQUAL" },
    { ComparisonOperation::Less, "LESS" },
    { ComparisonOperation::LessEqual, "LESSEQUAL" },
};

int calculate_num_elems_processed_per_iteration(const ITensorInfo &input)
{
    return 16 / input.element_size();
}

Status validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output, ComparisonOperation operation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1,
                                                         1,
                                                         DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &input2);
    ARM_COMPUTE_RETURN_ERROR_ON(supported_comparison_ops.count(operation) == 0);

    const TensorShape out_shape = TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::U8);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
                                        "Wrong shape for output");
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(input1, input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    const unsigned int num_elems_processed_per_iteration = calculate_num_elems_processed_per_iteration(input1);

    // Auto initialize output if not initialized
    auto_init_if_empty(output, out_shape, 1, DataType::U8, QuantizationInfo());

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

CLComparisonKernel::CLComparisonKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void CLComparisonKernel::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ComparisonOperation operation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1->info(), *input2->info(), *output->info(), operation));

    // Configure kernel window
    auto win_config = validate_and_configure_window(*input1->info(), *input2->info(), *output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    _input1 = input1;
    _input2 = input2;
    _output = output;

    const std::string &operation_name = supported_comparison_ops.at(operation);
    std::string        kernel_name    = "compare_" + lower_string(operation_name);

    // Set kernel build options
    std::set<std::string> build_opts;
    build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input1->info()->data_type()));
    build_opts.emplace("-DVEC_SIZE=" + support::cpp11::to_string(calculate_num_elems_processed_per_iteration(*input1->info())));
    build_opts.emplace("-DOP=" + operation_name);
    build_opts.emplace("-DOP_NAME=" + lower_string(operation_name));
    if(is_data_type_quantized_asymmetric(input1->info()->data_type()))
    {
        const UniformQuantizationInfo iq1_info = input1->info()->quantization_info().uniform();
        const UniformQuantizationInfo iq2_info = input2->info()->quantization_info().uniform();

        build_opts.emplace("-DOFFSET_IN1=" + support::cpp11::to_string(iq1_info.offset));
        build_opts.emplace("-DOFFSET_IN2=" + support::cpp11::to_string(iq2_info.offset));
        build_opts.emplace("-DSCALE_IN1=" + float_to_string_with_full_precision(iq1_info.scale));
        build_opts.emplace("-DSCALE_IN2=" + float_to_string_with_full_precision(iq2_info.scale));
        kernel_name += "_quantized";
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input1->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += lower_string(string_from_data_layout(input1->info()->data_layout()));
}

Status CLComparisonKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation operation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output, operation));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*input1->clone(), *input2->clone(), *output->clone()).first);

    return Status{};
}

void CLComparisonKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const TensorShape &in_shape1 = _input1->info()->tensor_shape();
    const TensorShape &in_shape2 = _input2->info()->tensor_shape();
    const TensorShape &out_shape = _output->info()->tensor_shape();

    bool       can_collapse = true;
    const bool is_vector    = in_shape1.num_dimensions() == 1 || in_shape2.num_dimensions() == 1;
    if(std::min(in_shape1.total_size(), in_shape2.total_size()) > 1 && !is_vector)
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

        enqueue(queue, *this, slice, lws_hint());

        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input1));
        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input2));
    }
    while(collapsed.slide_window_slice_3D(slice));
}

BorderSize CLComparisonKernel::border_size() const
{
    const int num_elems_processed_per_iteration = calculate_num_elems_processed_per_iteration(*_input1->info());

    const unsigned int replicateSize = _output->info()->dimension(0) - std::min(_input1->info()->dimension(0), _input2->info()->dimension(0));
    const unsigned int border        = std::min<unsigned int>(num_elems_processed_per_iteration - 1U, replicateSize);
    return BorderSize{ 0, border, 0, 0 };
}
} // namespace arm_compute
