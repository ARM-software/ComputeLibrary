/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "src/core/NEON/kernels/NEStridedSliceKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/helpers/tensor_transform.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/helpers/bit_ops.h"

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

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo *input, ITensorInfo *output,
                                                        const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                                        int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    // Output tensor auto initialization if not yet initialized
    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_strided_slice_shape(*input,
                                                                                                      starts, ends, strides,
                                                                                                      begin_mask, end_mask, shrink_axis_mask);
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));

    // Create window
    Window win = calculate_max_window(*output, Steps());

    return std::make_pair(Status{}, win);
}
} // namespace

NEStridedSliceKernel::NEStridedSliceKernel()
    : _starts_abs(), _final_strides(), _shrink_mask()
{
}

void NEStridedSliceKernel::configure(const ITensorInfo *input, ITensorInfo *output,
                                     const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                     int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask));
    _shrink_mask                   = shrink_axis_mask;
    const TensorShape &input_shape = input->tensor_shape();
    Coordinates        ends_abs;
    std::tie(_starts_abs, ends_abs, _final_strides) = arm_compute::helpers::tensor_transform::calculate_strided_slice_coords(
                                                          input_shape,
                                                          starts, ends, strides,
                                                          begin_mask, end_mask, shrink_axis_mask);
    // Configure kernel window
    auto win_config = validate_and_configure_window(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEStridedSliceKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                      const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                      int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(),
                                                              starts, ends, strides, begin_mask, end_mask, shrink_axis_mask)
                                .first);

    return Status{};
}

void NEStridedSliceKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const ITensor *input  = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *output = tensors.get_tensor(TensorType::ACL_DST);

    size_t width_size = input->info()->element_size();

    const bool is_shrink_x = arm_compute::helpers::bit_ops::is_bit_set(_shrink_mask, 0);
    const bool is_shrink_y = arm_compute::helpers::bit_ops::is_bit_set(_shrink_mask, 1);
    const bool is_shrink_z = arm_compute::helpers::bit_ops::is_bit_set(_shrink_mask, 2);
    const bool is_shrink_w = arm_compute::helpers::bit_ops::is_bit_set(_shrink_mask, 3);

    unsigned int index = 0;
    const int    idx_x = is_shrink_x ? 0 : index++;
    const int    idx_y = is_shrink_y ? 0 : index++;
    const int    idx_z = is_shrink_z ? 0 : index++;
    const int    idx_w = is_shrink_w ? 0 : index;

    BiStrides shrinked_strides;
    shrinked_strides.set(0, is_shrink_x ? 0 : _final_strides[0]);
    shrinked_strides.set(1, is_shrink_y ? 0 : _final_strides[1]);
    shrinked_strides.set(2, is_shrink_z ? 0 : _final_strides[2]);
    shrinked_strides.set(3, is_shrink_w ? 0 : _final_strides[3]);

    Window win = window;

    size_t length_x = win.shape()[0];

    if(_final_strides[0] == 1 && !is_shrink_x)
    {
        win.set(Window::DimX, Window::Dimension(0, 1, 1));
        width_size = width_size * length_x;
    }

    Iterator output_it(output, win);

    const int start_0 = _starts_abs[0];
    const int start_1 = _starts_abs[1];
    const int start_2 = _starts_abs[2];
    const int start_3 = _starts_abs[3];

    const int shrinked_stride_0 = shrinked_strides[0];
    const int shrinked_stride_1 = shrinked_strides[1];
    const int shrinked_stride_2 = shrinked_strides[2];
    const int shrinked_stride_3 = shrinked_strides[3];

    const int byte_increment_0 = static_cast<int>(input->info()->strides_in_bytes()[0]);
    const int byte_increment_1 = static_cast<int>(input->info()->strides_in_bytes()[1]);
    const int byte_increment_2 = static_cast<int>(input->info()->strides_in_bytes()[2]);
    const int byte_increment_3 = static_cast<int>(input->info()->strides_in_bytes()[3]);

    uint8_t *input_base = input->ptr_to_element(Coordinates(0, 0, 0, 0));
    uint8_t *cur_ptr;

    execute_window_loop(
        win, [&](const Coordinates & id)
    {
        cur_ptr = input_base;
        cur_ptr += (start_0 + (id[idx_x] * shrinked_stride_0)) * byte_increment_0;
        cur_ptr += (start_1 + (id[idx_y] * shrinked_stride_1)) * byte_increment_1;
        cur_ptr += (start_2 + (id[idx_z] * shrinked_stride_2)) * byte_increment_2;
        cur_ptr += (start_3 + (id[idx_w] * shrinked_stride_3)) * byte_increment_3;

        std::copy_n(cur_ptr, width_size, output_it.ptr());
    },
    output_it);
}
} // namespace arm_compute
