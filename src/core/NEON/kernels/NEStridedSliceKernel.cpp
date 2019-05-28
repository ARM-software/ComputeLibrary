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
#include "arm_compute/core/NEON/kernels/NEStridedSliceKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
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
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1,
                                                         DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16, DataType::QSYMM16,
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
    Window win = calculate_max_window(*output, Steps());
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    return std::make_pair(Status{}, win);
}

void strided_slice_generic(const ITensor *input, ITensor *output,
                           const Coordinates &starts, const BiStrides &strides, int32_t shrink_axis_mask,
                           const Window &window)
{
    Iterator     output_it(output, window);
    const size_t width_size = input->info()->element_size();

    const bool is_shrink_w = arm_compute::helpers::bit_ops::is_bit_set(shrink_axis_mask, 0);
    const bool is_shrink_h = arm_compute::helpers::bit_ops::is_bit_set(shrink_axis_mask, 1);
    const bool is_shrink_c = arm_compute::helpers::bit_ops::is_bit_set(shrink_axis_mask, 2);
    const bool is_shrink_n = arm_compute::helpers::bit_ops::is_bit_set(shrink_axis_mask, 3);

    unsigned int index = 0;
    const int    idx_w = is_shrink_w ? 0 : index++;
    const int    idx_h = is_shrink_h ? 0 : index++;
    const int    idx_c = is_shrink_c ? 0 : index++;
    const int    idx_n = is_shrink_n ? 0 : index;

    BiStrides shrinked_strides;
    shrinked_strides.set(0, is_shrink_w ? 0 : strides[0]);
    shrinked_strides.set(1, is_shrink_h ? 0 : strides[1]);
    shrinked_strides.set(2, is_shrink_c ? 0 : strides[2]);
    shrinked_strides.set(3, is_shrink_n ? 0 : strides[3]);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int w_coord = starts[0] + (id[idx_w] * shrinked_strides[0]);
        const int h_coord = starts[1] + (id[idx_h] * shrinked_strides[1]);
        const int c_coord = starts[2] + (id[idx_c] * shrinked_strides[2]);
        const int n_coord = starts[3] + (id[idx_n] * shrinked_strides[3]);

        Coordinates in_coords(w_coord, h_coord, c_coord, n_coord);
        std::copy_n(input->ptr_to_element(in_coords), width_size, output_it.ptr());
    },
    output_it);
}
} // namespace

NEStridedSliceKernel::NEStridedSliceKernel()
    : _input(nullptr), _output(nullptr), _starts_abs(), _final_strides(), _shrink_mask()
{
}

void NEStridedSliceKernel::configure(const ITensor *input, ITensor *output,
                                     const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                     int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), starts, ends, strides, begin_mask, end_mask, shrink_axis_mask));

    _input       = input;
    _output      = output;
    _shrink_mask = shrink_axis_mask;

    const TensorShape &input_shape = input->info()->tensor_shape();

    Coordinates ends_abs;
    std::tie(_starts_abs, ends_abs, _final_strides) = arm_compute::helpers::tensor_transform::calculate_strided_slice_coords(
                                                          input_shape,
                                                          starts, ends, strides,
                                                          begin_mask, end_mask, shrink_axis_mask);

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
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

void NEStridedSliceKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    // Dispatch kernel
    strided_slice_generic(_input, _output, _starts_abs, _final_strides, _shrink_mask, window);
}
} // namespace arm_compute
