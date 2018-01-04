/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_MISC_SHAPE_CALCULATOR_H__
#define __ARM_COMPUTE_MISC_SHAPE_CALCULATOR_H__

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Utils.h"

namespace arm_compute
{
namespace misc
{
namespace shape_calculator
{
inline TensorShape compute_permutation_output_shape(const ITensorInfo &input, const PermutationVector &perm)
{
    TensorShape output_shape = input.tensor_shape();
    permute(output_shape, perm);
    return output_shape;
}
inline TensorShape compute_interleaved_shape(const ITensorInfo &a)
{
    // The interleaved output matrix will have the following shape: [ a_height * 4, ceil(a_width / 4.0f) ]
    TensorShape shape_interleaved_a{ a.tensor_shape() };
    shape_interleaved_a.set(0, a.dimension(0) * 4);
    shape_interleaved_a.set(1, std::ceil(a.dimension(1) / 4.f));

    return shape_interleaved_a;
}
inline TensorShape compute_transpose1xW_shape(const ITensorInfo &b)
{
    // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
    TensorShape shape_transposed1xW_b{ b.tensor_shape() };
    shape_transposed1xW_b.set(0, b.dimension(1) * 16);
    shape_transposed1xW_b.set(1, std::ceil(b.dimension(0) / 16.f));

    return shape_transposed1xW_b;
}
inline TensorShape compute_transpose1xW_with_element_size_shape(const ITensorInfo &b)
{
    // The transpose1xW output matrix will have the following shape:
    // [ b_height * (16 / element_size), ceil(b_width / (16.0f / element_size) ]
    TensorShape  shape_transposed1xW_b{ b.tensor_shape() };
    const size_t transpose_width = 16 / b.element_size();
    shape_transposed1xW_b.set(0, b.dimension(1) * transpose_width);
    shape_transposed1xW_b.set(1, static_cast<size_t>(std::ceil(b.dimension(0) / static_cast<float>(transpose_width))));

    return shape_transposed1xW_b;
}
inline TensorShape compute_reductionA_shape(const ITensorInfo &b)
{
    TensorShape shape_vector_sum_col{ b.tensor_shape() };
    if(shape_vector_sum_col.num_dimensions() > 1)
    {
        shape_vector_sum_col.remove_dimension(1);
    }

    return shape_vector_sum_col;
}
inline TensorShape compute_reductionB_shape(const ITensorInfo &a)
{
    TensorShape shape_vector_sum_row{ a.tensor_shape() };
    shape_vector_sum_row.set(Window::DimX, a.dimension(1));
    if(a.num_dimensions() > 1)
    {
        shape_vector_sum_row.remove_dimension(1);
    }

    return shape_vector_sum_row;
}
inline TensorShape compute_im2col_shape(const ITensorInfo &input)
{
    TensorShape shape_im2col{ input.tensor_shape() };
    shape_im2col.collapse(3);

    return shape_im2col;
}
inline TensorShape compute_transposed_shape(const ITensorInfo &input)
{
    TensorShape shape_transposed{ input.tensor_shape() };

    shape_transposed.set(0, input.dimension(1));
    shape_transposed.set(1, input.dimension(0));

    return shape_transposed;
}
inline TensorShape compute_depthwise_convolution_shape(const ITensorInfo &input, const ITensorInfo &weights, PadStrideInfo conv_info)
{
    const TensorShape input_shape{ input.tensor_shape() };
    const TensorShape weights_shape{ weights.tensor_shape() };

    unsigned int output_width  = 0;
    unsigned int output_height = 0;
    std::tie(output_width, output_height) = scaled_dimensions(input_shape.x(), input_shape.y(), weights_shape.x(),
                                                              weights_shape.y(), conv_info);

    TensorShape output_shape{ input_shape };
    output_shape.set(0, output_width);
    output_shape.set(1, output_height);

    return output_shape;
}
inline TensorShape compute_deconvolution_shape(const ITensorInfo &input, unsigned int sx, unsigned int sy, unsigned int inner_border_right, unsigned int inner_border_top, const PadStrideInfo &info)
{
    TensorShape        scale_out_shape(input.tensor_shape());
    const unsigned int out_x = input.dimension(0) + (input.dimension(0) - 1) * (sx - 1) + inner_border_right + 2 * info.pad().first;
    const unsigned int out_y = input.dimension(1) + (input.dimension(1) - 1) * (sy - 1) + inner_border_top + 2 * info.pad().second;
    scale_out_shape.set(0, out_x);
    scale_out_shape.set(1, out_y);

    return scale_out_shape;
}
} // namespace shape_calculator
} // namespace misc
} // namespace arm_compute
#endif /* __ARM_COMPUTE_MISC_SHAPE_CALCULATOR_H__ */
