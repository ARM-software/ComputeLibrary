/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEReductionOperationKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace
{
template <class F>
class Reducer
{
public:
    static void reduceX(const Window &window, const ITensor *input, ITensor *output, F f)
    {
        // Set out window
        Window out_window(window);
        out_window.set(Window::DimX, Window::Dimension(0, 0, 0));

        // Get first input and output slices
        Window in_slice  = window.first_slice_window_1D();
        Window out_slice = out_window.first_slice_window_1D();

        do
        {
            Iterator in(input, in_slice);
            Iterator out(output, out_slice);

            f(in, out, in_slice, out_slice);
        }
        while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(out_slice));
    }
};

struct SumsqOpX
{
    inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice)
    {
        ARM_COMPUTE_UNUSED(out_slice);
        float32x4_t vec_sum_value = vdupq_n_f32(0.f);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto        in_ptr       = reinterpret_cast<const float *>(input.ptr());
            const float32x4_t vec_elements = vld1q_f32(in_ptr);
            vec_sum_value                  = vaddq_f32(vmulq_f32(vec_elements, vec_elements), vec_sum_value);
        },
        input);

        float32x2_t carry_addition = vpadd_f32(vget_high_f32(vec_sum_value), vget_low_f32(vec_sum_value));
        carry_addition             = vpadd_f32(carry_addition, carry_addition);

        *(reinterpret_cast<float *>(output.ptr())) = vget_lane_f32(carry_addition, 0);
    }
};

void reduce_sumsq(const Window &window, const ITensor *input, ITensor *output, unsigned int axis)
{
    switch(axis)
    {
        case 0:
            return Reducer<SumsqOpX>::reduceX(window, input, output, SumsqOpX());
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction axis");
    }
}
} // namespace

NEReductionOperationKernel::NEReductionOperationKernel()
    : _input(nullptr), _output(nullptr), _reduction_axis(0), _op(ReductionOperation::SUM_SQUARE), _border_size()
{
}

BorderSize NEReductionOperationKernel::border_size() const
{
    return _border_size;
}

void NEReductionOperationKernel::configure(const ITensor *input, ITensor *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_ERROR_ON_MSG(axis > 0, "Unsupported reduction axis, Supported axis is 0");

    // Calculate output shape and set if empty
    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.set(axis, 1);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);

    unsigned int num_elems_processed_per_iteration = 16 / data_size_from_type(input->info()->data_type());

    _input       = input;
    _output      = output;
    _border_size = (axis == 0) ? BorderSize(0, num_elems_processed_per_iteration - (input->info()->dimension(0) % num_elems_processed_per_iteration), 0, 0) : BorderSize();
    _op          = op;

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEReductionOperationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_op)
    {
        case ReductionOperation::SUM_SQUARE:
            reduce_sumsq(window, _input, _output, _reduction_axis);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction operation.");
    }
}
