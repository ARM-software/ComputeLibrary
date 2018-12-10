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
#include "arm_compute/core/CL/kernels/CLStridedSliceKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/helpers/bit_ops.h"
#include "arm_compute/core/utils/helpers/tensor_transform.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output,
                          const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                          int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1,
                                                         DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);

    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(starts.num_dimensions() > input->num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(ends.num_dimensions() > input->num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(strides.num_dimensions() > input->num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(std::any_of(strides.cbegin(), strides.cbegin() + strides.num_dimensions(), [](int i)
    {
        return i == 0;
    }));

    // Get expected output shape
    const TensorShape exp_output_shape = arm_compute::misc::shape_calculator::compute_strided_slice_shape(*input,
                                                                                                          starts, ends, strides,
                                                                                                          begin_mask, end_mask, shrink_axis_mask);
    ARM_COMPUTE_RETURN_ERROR_ON(exp_output_shape.total_size() == 0);

    // Checks output if configured
    if(output->total_size() != 0)
    {
        const TensorInfo exp_output_info = output->clone()->set_tensor_shape(exp_output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &exp_output_info);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output,
                                                        const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                                        int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    // Output tensor auto initialization if not yet initialized
    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_strided_slice_shape(*input,
                                                                                                      starts, ends, strides,
                                                                                                      begin_mask, end_mask, shrink_axis_mask);
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));

    // Create window
    const unsigned int num_elems_processed_per_iteration = 1;

    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    return std::make_pair(Status{}, win);
}
} // namespace

CLStridedSliceKernel::CLStridedSliceKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLStridedSliceKernel::configure(const ICLTensor *input, ICLTensor *output,
                                     const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                     int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), starts, ends, strides, begin_mask, end_mask, shrink_axis_mask));

    _input  = input;
    _output = output;

    const TensorShape &input_shape = input->info()->tensor_shape();

    Coordinates starts_abs, ends_abs, final_strides;
    std::tie(starts_abs, ends_abs, final_strides) = arm_compute::helpers::tensor_transform::calculate_strided_slice_coords(
                                                        input_shape,
                                                        starts, ends, strides,
                                                        begin_mask, end_mask, shrink_axis_mask);

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    // Enable multiple elements processing along x if stride_x is 1 and output width greater than the access vector size
    const int  vec_size_x     = 16 / input->info()->element_size();
    const int  output_width_x = output->info()->tensor_shape().x();
    const bool is_shrink_on_x = arm_compute::helpers::bit_ops::is_bit_set(shrink_axis_mask, 0);
    const bool multi_access_x = !is_shrink_on_x && (final_strides.x() == 1) && (output_width_x / vec_size_x > 0);

    // Update window if needed
    if(multi_access_x)
    {
        Window &updated_window = std::get<1>(win_config);
        updated_window.set(Window::DimX,
                           Window::Dimension(updated_window.x().start(), ceil_to_multiple(updated_window.x().end(), vec_size_x), vec_size_x));
    }
    ICLKernel::configure_internal(win_config.second);

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    for(unsigned int i = 0; i < input_shape.num_dimensions(); ++i)
    {
        const bool is_shrink = arm_compute::helpers::bit_ops::is_bit_set(shrink_axis_mask, i);
        build_opts.add_option("-DSTART_" + support::cpp11::to_string(i) + "=" + support::cpp11::to_string(starts_abs[i]));
        build_opts.add_option("-DSTRIDE_" + support::cpp11::to_string(i) + "=" + support::cpp11::to_string(final_strides[i]));
        build_opts.add_option_if(is_shrink, "-DSHRINK_" + support::cpp11::to_string(i));
    }
    build_opts.add_option_if(multi_access_x, "-DLAST_ACCESSED_X=" + support::cpp11::to_string(std::max<int>(output_width_x - vec_size_x, 0)));
    build_opts.add_option_if(multi_access_x, "-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option_if_else(input_shape.num_dimensions() > 2,
                                  "-DSRC_DEPTH=" + support::cpp11::to_string(input_shape.z()),
                                  "-DSRC_DEPTH=1");
    build_opts.add_option_if_else(_output->info()->num_dimensions() > 2,
                                  "-DDST_DEPTH=" + support::cpp11::to_string(_output->info()->tensor_shape().z()),
                                  "-DDST_DEPTH=1");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("strided_slice", build_opts.options()));

    // Set config_id for enabling LWS tuning
    _config_id = "strided_slice";
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    for(unsigned int i = 0; i < input_shape.num_dimensions(); ++i)
    {
        _config_id += "_";
        _config_id += support::cpp11::to_string(input->info()->dimension(i));
        _config_id += "_";
        _config_id += support::cpp11::to_string(starts_abs[i]);
        _config_id += "_";
        _config_id += support::cpp11::to_string(ends_abs[i]);
        _config_id += "_";
        _config_id += support::cpp11::to_string(final_strides[i]);
    }
}

Status CLStridedSliceKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                      const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                      int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(),
                                                              starts, ends, strides, begin_mask, end_mask, shrink_axis_mask)
                                .first);

    return Status{};
}

void CLStridedSliceKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_4D();

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, _input, slice);
        add_4D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window_collapsed.slide_window_slice_4D(slice));
}
} // namespace arm_compute
