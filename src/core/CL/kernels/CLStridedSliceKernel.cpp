/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/core/CL/kernels/CLStridedSliceKernel.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/helpers/tensor_transform.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/helpers/bit_ops.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output,
                          const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                          int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);

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
} // namespace

void CLStridedSliceKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output,
                                     const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                     int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    auto padding_info = get_padding_info({ input, output });
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask));

    const TensorShape &input_shape = input->tensor_shape();

    Coordinates starts_abs;
    Coordinates ends_abs;
    Coordinates final_strides;
    std::tie(starts_abs, ends_abs, final_strides) = arm_compute::helpers::tensor_transform::calculate_strided_slice_coords(
                                                        input_shape,
                                                        starts, ends, strides,
                                                        begin_mask, end_mask, shrink_axis_mask);

    // Configure kernel window
    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_strided_slice_shape(*input,
                                                                                                      starts, ends, strides,
                                                                                                      begin_mask, end_mask, shrink_axis_mask);
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));
    Window win = calculate_max_window(*output, Steps());

    // Enable multiple elements processing along x if stride_x is 1 and output width greater than the access vector size
    const int  vec_size_x     = 16 / input->element_size();
    const int  output_width_x = output->tensor_shape().x();
    const bool is_shrink_on_x = arm_compute::helpers::bit_ops::is_bit_set(shrink_axis_mask, 0);
    const bool multi_access_x = !is_shrink_on_x && (final_strides.x() == 1) && (output_width_x / vec_size_x > 0);

    // Update window if needed
    if(multi_access_x)
    {
        Window &updated_window = win;
        updated_window.set(Window::DimX,
                           Window::Dimension(updated_window.x().start(), ceil_to_multiple(updated_window.x().end(), vec_size_x), vec_size_x));
    }
    ICLKernel::configure_internal(win);

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(data_size_from_type(input->data_type())));
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
    build_opts.add_option_if_else(output->num_dimensions() > 2,
                                  "-DDST_DEPTH=" + support::cpp11::to_string(output->tensor_shape().z()),
                                  "-DDST_DEPTH=1");

    // Create kernel
    _kernel = create_kernel(compile_context, "strided_slice", build_opts.options());

    // Set config_id for enabling LWS tuning
    _config_id = "strided_slice";
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->data_type()));
    for(unsigned int i = 0; i < input_shape.num_dimensions(); ++i)
    {
        _config_id += "_";
        _config_id += support::cpp11::to_string(input->dimension(i));
        _config_id += "_";
        _config_id += support::cpp11::to_string(starts_abs[i]);
        _config_id += "_";
        _config_id += support::cpp11::to_string(ends_abs[i]);
        _config_id += "_";
        _config_id += support::cpp11::to_string(final_strides[i]);
    }
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLStridedSliceKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                      const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                      int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask));

    return Status{};
}

void CLStridedSliceKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_4D();

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, src, slice);
        add_4D_tensor_argument(idx, dst, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window_collapsed.slide_window_slice_4D(slice));
}
} // namespace arm_compute
