/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_MISC_SHAPE_CALCULATOR_H
#define ARM_COMPUTE_MISC_SHAPE_CALCULATOR_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/FunctionDescriptors.h"

#include "arm_compute/core/utils/helpers/tensor_transform.h"

#include <cmath>

namespace arm_compute
{
namespace misc
{
namespace shape_calculator
{
/** Calculate the output tensor shape for the reduce mean operation
 *
 * @param[in] input          Input tensor shape
 * @param[in] reduction_axis Reduction axis
 * @param[in] keep_dims      Flag to indicate if dimensions are kept
 *
 * @return the calculated shape
 */
inline TensorShape calculate_reduce_mean_shape(ITensorInfo *input, const Coordinates &reduction_axis, bool keep_dims)
{
    const int   reduction_ops = reduction_axis.num_dimensions();
    Coordinates axis_local    = reduction_axis;
    const int   input_dims    = input->num_dimensions();
    convert_negative_axis(axis_local, input_dims);
    TensorShape out_shape = input->tensor_shape();
    // Configure reshape layer if we want to drop the dimensions
    if(!keep_dims)
    {
        // We have to sort the reduction axis vectors in order for remove_dimension
        // to work properly
        std::sort(axis_local.begin(), axis_local.begin() + reduction_ops);
        for(int i = 0; i < reduction_ops; ++i)
        {
            out_shape.remove_dimension(axis_local[i] - i);
        }
        return out_shape;
    }
    else
    {
        for(int i = 0; i < reduction_ops; ++i)
        {
            out_shape.set(axis_local[i], 1);
        }
        return out_shape;
    }
}
/** Calculate the output tensor shape of a vector input given the convolution dimensions
 *
 * @param[in] input       Input tensor shape
 * @param[in] conv_w      Convolution width
 * @param[in] conv_h      Convolution height
 * @param[in] data_layout Data layout
 *
 * @return the calculated shape
 */
inline TensorShape compute_vector_to_tensor_output_shape(const TensorShape &input, size_t conv_w, size_t conv_h, const DataLayout &data_layout)
{
    const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const size_t idx_c = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    TensorShape output_shape(input);
    output_shape.set(idx_w, conv_w);
    output_shape.set(idx_h, conv_h);
    output_shape.set(idx_c, input.x() / (conv_w * conv_h));

    return output_shape;
}

/** Calculate the permuted shape of an input given a permutation vector
 *
 * @param[in] input Input tensor info
 * @param[in] perm  Permutation vector
 *
 * @return the calculated shape
 */
inline TensorShape compute_permutation_output_shape(const ITensorInfo &input, const PermutationVector &perm)
{
    TensorShape output_shape = input.tensor_shape();
    permute(output_shape, perm);
    return output_shape;
}

/** Calculate the output shape of the reorg layer given a stride
 *
 * @param[in] input  Input tensor info
 * @param[in] stride Stride
 *
 * @return the calculated shape
 */
inline TensorShape compute_reorg_output_shape(const ITensorInfo &input, int32_t stride)
{
    const size_t idx_width   = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height  = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::HEIGHT);
    const size_t idx_channel = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_ERROR_ON(stride <= 0);
    ARM_COMPUTE_ERROR_ON_MSG((input.tensor_shape()[idx_width] % stride != 0), "The width of the input tensor must be a multiple of stride");
    ARM_COMPUTE_ERROR_ON_MSG((input.tensor_shape()[idx_height] % stride != 0), "The height of the input tensor must be a multiple of stride");

    TensorShape output_shape{ input.tensor_shape() };

    output_shape.set(idx_width, output_shape[idx_width] / stride);
    output_shape.set(idx_height, output_shape[idx_height] / stride);
    output_shape.set(idx_channel, output_shape[idx_channel] * stride * stride);

    return output_shape;
}

/** Calculate the reshaped shape of the weights
 *
 * @param[in] weights    Weights tensor info
 * @param[in] has_bias   (Optional) Set to true if there is bias
 * @param[in] num_groups (Optional) Number of groups
 *
 * @return the calculated shape of the reshaped weights
 */
inline TensorShape compute_weights_reshaped_shape(const ITensorInfo &weights, bool has_bias = false, unsigned int num_groups = 1)
{
    // Number of groups greater than one are only supported for NCHW data layout, and the number of weights must be a multiple of it.
    ARM_COMPUTE_ERROR_ON(num_groups == 0);
    ARM_COMPUTE_ERROR_ON(weights.data_layout() == DataLayout::NHWC && num_groups > 1);
    ARM_COMPUTE_ERROR_ON((weights.dimension(3) % num_groups) != 0);

    // Calculate output shape
    TensorShape weights_reshaped{ weights.tensor_shape() };
    weights_reshaped.set(3, weights_reshaped[3] / num_groups);

    weights_reshaped.collapse(3);
    const size_t tmp_dim = weights_reshaped[0];
    weights_reshaped.set(0, weights_reshaped[1]);
    weights_reshaped.set(1, tmp_dim + (has_bias ? 1 : 0));
    if(weights.num_dimensions() < 5)
    {
        weights_reshaped.set(2, num_groups);
    }

    return weights_reshaped;
}

/** Calculate the Left Hand Side matrix reshaped shape
 *
 * @param[in] a                       Input tensor info
 * @param[in] lhs_info                Left Hand Side matrix information
 * @param[in] reinterpret_input_as_3d (Optional) Set to true if the input need to be interpreted as 3d
 *
 * @return the calculated shape
 */
inline TensorShape compute_lhs_reshaped_shape(const ITensorInfo &a, const GEMMLHSMatrixInfo &lhs_info, bool reinterpret_input_as_3d = false)
{
    ARM_COMPUTE_ERROR_ON(lhs_info.m0 == 0);
    ARM_COMPUTE_ERROR_ON(lhs_info.k0 == 0);
    ARM_COMPUTE_ERROR_ON(lhs_info.v0 == 0);

    // Input width/height
    const unsigned int input_width  = a.dimension(0);
    const unsigned int input_height = reinterpret_input_as_3d ? a.dimension(1) * a.dimension(2) : a.dimension(1);

    // Number of horizontal/vertical blocks in the input tensor
    const unsigned int num_horiz_blocks = std::ceil(input_width / static_cast<float>(lhs_info.k0));
    const unsigned int num_vert_blocks  = std::ceil(input_height / static_cast<float>(lhs_info.m0));

    // Block size
    const unsigned int block_size = lhs_info.m0 * lhs_info.k0;

    // Output width/height
    const unsigned int output_width  = block_size * num_horiz_blocks * lhs_info.v0;
    const unsigned int output_height = std::ceil(num_vert_blocks / static_cast<float>(lhs_info.v0));

    TensorShape lhs_shape{ a.tensor_shape() };
    lhs_shape.set(0, output_width);
    lhs_shape.set(1, output_height);

    if((reinterpret_input_as_3d) && (lhs_shape.num_dimensions() > 2))
    {
        // When the data format is NHWC and the shapes are Nx1x1
        // the tensor shape num_dimensions is automatically set to 1 instead of 3.
        // To avoid failures by removing a dimension that doesn't exist
        // check if the number of dimensions is greater than 2.
        lhs_shape.remove_dimension(2);
    }

    return lhs_shape;
}

/** Calculate the Right Hand Side matrix reshaped shape
 *
 * @param[in] a        Input tensor info
 * @param[in] rhs_info Right Hand Side matrix information
 *
 * @return the calculated shape
 */
inline TensorShape compute_rhs_reshaped_shape(const ITensorInfo &a, const GEMMRHSMatrixInfo &rhs_info)
{
    ARM_COMPUTE_ERROR_ON(rhs_info.n0 == 0);
    ARM_COMPUTE_ERROR_ON(rhs_info.k0 == 0);
    ARM_COMPUTE_ERROR_ON(rhs_info.h0 == 0);

    // Input width/height
    const unsigned int input_width  = a.dimension(0);
    const unsigned int input_height = a.dimension(1);

    // Number of horizontal/vertical blocks in the input tensor
    const unsigned int num_horiz_blocks = std::ceil(input_width / static_cast<float>(rhs_info.n0));
    const unsigned int num_vert_blocks  = std::ceil(input_height / static_cast<float>(rhs_info.k0));

    // Block size
    const unsigned int block_size = rhs_info.n0 * rhs_info.k0;

    // Output width/height
    const unsigned int output_width  = block_size * num_vert_blocks * rhs_info.h0;
    const unsigned int output_height = std::ceil(num_horiz_blocks / static_cast<float>(rhs_info.h0));

    TensorShape rhs_shape{ a.tensor_shape() };
    rhs_shape.set(0, output_width);
    rhs_shape.set(1, output_height);

    return rhs_shape;
}

/** Calculate the interleaved shape of an input tensor
 *
 * @param[in] a                         Input tensor info
 * @param[in] mult_interleave4x4_height (Optional) Interleave4x4 height
 * @param[in] reinterpret_input_as_3d   (Optional)  Set to true if the input need to be interpreted as 3d
 *
 * @return the calculated shape
 */
inline TensorShape compute_interleaved_shape(const ITensorInfo &a, int mult_interleave4x4_height = 1, bool reinterpret_input_as_3d = false)
{
    // The interleaved output matrix will have the following shape: [ a_height * W, ceil(a_width / W) ] where W = 4 * mult_interleave4x4_height
    ARM_COMPUTE_ERROR_ON(mult_interleave4x4_height < 1);
    const int   interleave_width = 4 * mult_interleave4x4_height;
    TensorShape shape_interleaved_a{ a.tensor_shape() };
    shape_interleaved_a.set(0, a.dimension(0) * interleave_width);
    if(reinterpret_input_as_3d)
    {
        const int M      = a.dimension(1) * a.dimension(2);
        const int height = std::ceil(M / static_cast<float>(interleave_width));
        shape_interleaved_a.set(1, height);

        // When the data format is NHWC and the shapes are Nx1x1
        // the tensor shape num_dimensions is automatically set to 1 instead of 3.
        // To avoid failures by removing a dimension that doesn't exist
        // check if the number of dimensions is greater than 2.
        if(shape_interleaved_a.num_dimensions() > 2)
        {
            shape_interleaved_a.remove_dimension(2);
        }
    }
    else
    {
        shape_interleaved_a.set(1, std::ceil(a.dimension(1) / static_cast<float>(interleave_width)));
    }

    return shape_interleaved_a;
}

/** Calculate the transposed 1xW shape
 *
 * @param[in] b Input tensor info
 *
 * @return the calculated shape
 */
inline TensorShape compute_transpose1xW_shape(const ITensorInfo &b)
{
    // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
    TensorShape shape_transposed1xW_b{ b.tensor_shape() };
    shape_transposed1xW_b.set(0, b.dimension(1) * 16);
    shape_transposed1xW_b.set(1, std::ceil(b.dimension(0) / 16.f));

    return shape_transposed1xW_b;
}

/** Calculate the transposed 1xW width element shape
 *
 * @param[in] b                       Input tensor info
 * @param[in] mult_transpose1xW_width (Optional) Transpose1xW width
 *
 * @return the calculated shape
 */
inline TensorShape compute_transpose1xW_with_element_size_shape(const ITensorInfo &b, int mult_transpose1xW_width = 1)
{
    // Note: mult_transpose1xW_width expresses the number of chunks with size 1x(W) we want to store on the same row
    //       The transpose1xW output matrix will have the following shape:
    //       [ b_height * W, ceil(b_width / W) ] where W = (16 / element size of the tensor) * mult_transpose1xW_width
    ARM_COMPUTE_ERROR_ON(mult_transpose1xW_width < 1);
    TensorShape  shape_transposed1xW_b{ b.tensor_shape() };
    const size_t transpose_width = (16 / b.element_size()) * mult_transpose1xW_width;
    shape_transposed1xW_b.set(0, b.dimension(1) * transpose_width);
    shape_transposed1xW_b.set(1, static_cast<size_t>(std::ceil(b.dimension(0) / static_cast<float>(transpose_width))));

    return shape_transposed1xW_b;
}

/** Calculate the reductionA shape used in GEMMLowp
 *
 * @param[in] b Input tensor info
 *
 * @return the calculated shape
 */
inline TensorShape compute_reductionA_shape(const ITensorInfo &b)
{
    TensorShape shape_vector_sum_col{ b.tensor_shape() };
    if(shape_vector_sum_col.num_dimensions() > 1)
    {
        shape_vector_sum_col.remove_dimension(1);
    }

    return shape_vector_sum_col;
}

/** Calculate the reductionB shape used in GEMMLowp
 *
 * @param[in] a Input tensor info
 *
 * @return the calculated shape
 */
inline TensorShape compute_reductionB_shape(const ITensorInfo &a)
{
    TensorShape shape_vector_sum_row{ a.tensor_shape() };
    shape_vector_sum_row.set(Window::DimX, a.dimension(1));
    if(shape_vector_sum_row.num_dimensions() > 1)
    {
        shape_vector_sum_row.remove_dimension(1);
    }

    return shape_vector_sum_row;
}

/** Calculate the Col2Im shape
 *
 * @param[in] input           Input tensor info
 * @param[in] convolved_dims  Convolved dimensions
 * @param[in] batch_size_on_z True if batch size is on z axis
 * @param[in] num_groups      (Optional)  Number of groups when performing a grouped convolution
 *
 * @return the calculated shape
 */
inline TensorShape compute_col2im_shape(const ITensorInfo &input, const Size2D &convolved_dims, bool batch_size_on_z, unsigned int num_groups = 1)
{
    ARM_COMPUTE_ERROR_ON(num_groups == 0);
    ARM_COMPUTE_ERROR_ON(input.tensor_shape()[1] != (convolved_dims.area()));
    ARM_COMPUTE_ERROR_ON((num_groups > 1) && input.tensor_shape()[2] != num_groups);

    const DataLayout data_layout = input.data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    TensorShape col2im_shape{ input.tensor_shape() };
    // If batches start on 3rd dimension shift dimensions right by 1 to retain upper tensor shape,
    // as first three will be override by H,W,C data
    if(batch_size_on_z && num_groups == 1)
    {
        col2im_shape.shift_right(1);
    }
    col2im_shape.set(width_idx, convolved_dims.width);
    col2im_shape.set(height_idx, convolved_dims.height);
    col2im_shape.set(channel_idx, input.tensor_shape()[0] * num_groups);

    return col2im_shape;
}

/** Calculate the transposed shape of a tensor
 *
 * @param[in] input Input tensor info
 *
 * @return the calculated shape
 */
inline TensorShape compute_transposed_shape(const ITensorInfo &input)
{
    TensorShape shape_transposed{ input.tensor_shape() };

    shape_transposed.set(0, input.dimension(1));
    shape_transposed.set(1, input.dimension(0));

    return shape_transposed;
}

/** Calculate the depthwise convolution output shape of a tensor
 *
 * @param[in] input   Input tensor info
 * @param[in] weights Weights tensor info
 * @param[in] info    Convolution info
 *
 * @return the calculated shape
 */
inline TensorShape compute_depthwise_convolution_shape(const ITensorInfo &input, const ITensorInfo &weights, const ConvolutionInfo &info)
{
    const TensorShape input_shape{ input.tensor_shape() };
    const TensorShape weights_shape{ weights.tensor_shape() };

    const DataLayout data_layout = input.data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const DataLayout weights_data_layout = weights.data_layout();
    const int        weights_width_idx   = get_data_layout_dimension_index(weights_data_layout, DataLayoutDimension::WIDTH);
    const int        weights_height_idx  = get_data_layout_dimension_index(weights_data_layout, DataLayoutDimension::HEIGHT);

    unsigned int output_width  = 0;
    unsigned int output_height = 0;
    std::tie(output_width, output_height) = scaled_dimensions(input_shape[width_idx], input_shape[height_idx],
                                                              weights_shape[weights_width_idx], weights_shape[weights_height_idx],
                                                              info.pad_stride_info, info.dilation);

    TensorShape output_shape{ input_shape };
    output_shape.set(width_idx, output_width);
    output_shape.set(height_idx, output_height);
    output_shape.set(channel_idx, input_shape[channel_idx] * info.depth_multiplier);

    return output_shape;
}

/** Calculate the upsampled output shape used for deconvolution
 *
 * @param[in] input    Input tensor info
 * @param[in] weights  Weights tensor shape
 * @param[in] sx       Stride on x axis
 * @param[in] sy       Stride on y axis
 * @param[in] out_dims Output shape dimensions
 * @param[in] padx     Padding on x axis
 * @param[in] pady     Padding on y axis
 *
 * @return the calculated shape
 */
inline TensorShape compute_deconvolution_upsampled_shape(const ITensorInfo &input, const ITensorInfo &weights, unsigned int sx, unsigned int sy,
                                                         std::pair<unsigned int, unsigned int> &out_dims, uint32_t &padx, uint32_t &pady)
{
    const DataLayout data_layout = input.data_layout();
    const size_t     idx_w       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t     idx_h       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Find the upsampled dimensions
    unsigned int out_x = (input.dimension(idx_w) - 1) * sx + 1;
    unsigned int out_y = (input.dimension(idx_h) - 1) * sy + 1;

    // Find the padding needed for the convolution with stride 1 in order to match output shape
    padx = out_dims.first - (out_x - weights.dimension(idx_w) + 1);
    pady = out_dims.second - (out_y - weights.dimension(idx_h) + 1);
    out_x += padx;
    out_y += pady;

    TensorShape scale_out_shape(input.tensor_shape());
    scale_out_shape.set(idx_w, out_x);
    scale_out_shape.set(idx_h, out_y);

    return scale_out_shape;
}

/** Calculate the output shape of the deconvolution layer
 *
 * @param[in] out_dims Output x and y shape dimensions
 * @param[in] input    Input tensor info
 * @param[in] weights  Weights tensor shape
 *
 * @return the calculated shape
 */
inline TensorShape compute_deconvolution_output_shape(const std::pair<unsigned int, unsigned int> &out_dims, const ITensorInfo &input, const ITensorInfo &weights)
{
    const TensorShape input_shape{ input.tensor_shape() };
    const TensorShape weights_shape{ weights.tensor_shape() };

    const DataLayout data_layout = input.data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    const int        batch_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    TensorShape out_shape{ input_shape };
    out_shape.set(width_idx, out_dims.first);
    out_shape.set(height_idx, out_dims.second);
    out_shape.set(channel_idx, weights_shape[batch_idx]);
    return out_shape;
}

/** Calculate the im2col output shape of a tensor
 *
 * @param[in] input           Input tensor info
 * @param[in] kernel_dims     The kernel dimensions (width and height).
 * @param[in] conv_info       Contains padding and stride information
 * @param[in] has_bias        In case biases are provided expands the matrix with 1
 * @param[in] dilation        Dilation, in elements, across x and y
 * @param[in] batch_size_on_z True if batch size is on z axis
 * @param[in] num_groups      (Optional)  Number of groups when performing a grouped convolution
 *
 * @return the calculated shape
 */
inline TensorShape compute_im2col_conv_shape(const ITensorInfo *input, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation, bool batch_size_on_z,
                                             unsigned int num_groups = 1)
{
    // The output shape will be the 3D shape [ out_channels * kernel_area, num_elems_per_out_channel, batches ]                           if batch_size_on_z == true
    //                       or the 4D shape [ out_channels * kernel_area / num_groups, num_elems_per_out_channel, num_groups, batches ]  if batch_size_on_z == false

    ARM_COMPUTE_ERROR_ON(num_groups == 0);
    ARM_COMPUTE_ERROR_ON(num_groups > 1 && input->data_layout() != DataLayout::NCHW);
    ARM_COMPUTE_ERROR_ON(num_groups > 1 && batch_size_on_z);

    TensorShape output_shape{ input->tensor_shape() };

    const DataLayout data_layout = input->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    std::pair<unsigned int, unsigned int> out_dims = scaled_dimensions(output_shape[width_idx], output_shape[height_idx], kernel_dims.width, kernel_dims.height, conv_info, dilation);
    output_shape.set(0, (output_shape[channel_idx] / num_groups * kernel_dims.area() + (has_bias ? 1 : 0))); // NOLINT
    output_shape.set(1, (out_dims.first * out_dims.second));
    if(batch_size_on_z && output_shape.num_dimensions() >= 3)
    {
        output_shape.remove_dimension(2);
    }
    else
    {
        output_shape.set(2, num_groups);
    }

    return output_shape;
}

/** Calculate the flattened output shape of a tensor
 *
 * @param[in] input Input tensor info
 *
 * @return the calculated shape
 */
inline TensorShape compute_flatten_shape(const ITensorInfo *input)
{
    // The output shape will be the flatten version of the input (i.e. [ width * height * channels, num_batches, ... ] ). Used for FlattenLayer and FullyConnectedLayer.

    TensorShape output_shape{ input->tensor_shape() };

    output_shape.collapse(3);

    return output_shape;
}

/** Calculate the softmax output shape of a tensor
 *
 * @param[in] input Input tensor info
 * @param[in] axis  (Optional) Softmax axis
 *
 * @return the calculated shape
 */
inline TensorShape compute_softmax_shape(const ITensorInfo *input, size_t axis = 1)
{
    // The output shape will be a 2D version of the input. For instance:
    // - [x,y,z] and axis 1 will return [x, y*z]
    // - [x,y,z,w] and axis 2 will return [x*y, w*z]
    // - [x,y,z,w] and axis 3 will return [x*y*z, w]
    TensorShape shape2D = input->tensor_shape();

    if(axis < input->num_dimensions())
    {
        // Collapse from axis onward (this changes the shape)
        shape2D.collapse_from(axis);

        // Collapse the rest (collapse is inclusive)
        shape2D.collapse(shape2D.num_dimensions() - 1);
    }
    else
    {
        // Collapse everything
        shape2D.collapse(shape2D.num_dimensions());
    }

    if(axis == 0)
    {
        // If axis is zero the first dim should be one. Since
        // collapse is an inclusive operation we need to shift
        shape2D.shift_right(1);
    }

    return shape2D;
}

/** Calculate the winograd filter transform shape
 *
 * @param[in] input         Input tensor info
 * @param[in] winograd_info Winograd information
 *
 * @return the calculated shape
 */
inline TensorShape compute_winograd_filter_transform_shape(const ITensorInfo &input, const WinogradInfo &winograd_info)
{
    TensorShape tensor_shape{ input.tensor_shape() };

    const Size2D kernel_size      = winograd_info.kernel_size;
    const Size2D output_tile_size = winograd_info.output_tile_size;
    const Size2D input_tile_size  = Size2D(output_tile_size.width + kernel_size.width - 1, output_tile_size.height + kernel_size.height - 1);

    tensor_shape.remove_dimension(get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::WIDTH));
    tensor_shape.set(Window::DimX, input.dimension(3));
    tensor_shape.set(Window::DimY, input.dimension(get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::CHANNEL)));
    tensor_shape.set(Window::DimZ, input_tile_size.area());

    return tensor_shape;
}

/** Calculate the winograd input transform shape
 *
 * @param[in] input         Input tensor info
 * @param[in] winograd_info Winograd information
 *
 * @return the calculated shape
 */
inline TensorShape compute_winograd_input_transform_shape(const ITensorInfo &input, const WinogradInfo &winograd_info)
{
    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        kernel_size      = winograd_info.kernel_size;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const Size2D        input_tile_size  = Size2D(output_tile_size.width + kernel_size.width - 1, output_tile_size.height + kernel_size.height - 1);

    const size_t idx_w = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::HEIGHT);
    const size_t idx_c = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::CHANNEL);

    // Compute the number of output tiles along the x and y direction of size "output_tile_size"
    const Size2D num_tiles = compute_winograd_convolution_tiles(Size2D(input.tensor_shape()[idx_w], input.tensor_shape()[idx_h]),
                                                                kernel_size,
                                                                output_tile_size,
                                                                conv_info);

    const unsigned int width  = input.tensor_shape()[idx_c];
    const unsigned int height = num_tiles.area();
    const unsigned int depth  = input_tile_size.area();

    TensorShape output_shape{ input.tensor_shape() };
    output_shape.set(0, width);
    output_shape.set(1, height);
    output_shape.set(2, depth);

    return output_shape;
}

/** Calculate the winograd output transform shape
 *
 * @param[in] input         Input tensor info
 * @param[in] winograd_info Winograd information
 *
 * @return the calculated shape
 */
inline TensorShape compute_winograd_output_transform_shape(const ITensorInfo &input, const WinogradInfo &winograd_info)
{
    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        kernel_size      = winograd_info.kernel_size;
    const Size2D        input_dimensions = winograd_info.input_dimensions;
    const DataLayout    data_layout      = winograd_info.output_data_layout;

    // Compute output shape
    unsigned int output_width  = 0;
    unsigned int output_height = 0;
    std::tie(output_width, output_height) = scaled_dimensions(input_dimensions.width, input_dimensions.height,
                                                              kernel_size.width, kernel_size.height, conv_info);

    TensorShape tensor_shape{ input.tensor_shape() };

    // Output dimension
    const unsigned int out_w = output_width;
    const unsigned int out_h = output_height;
    const unsigned int out_c = input.dimension(0);

    tensor_shape.set(get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH), out_w);
    tensor_shape.set(get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT), out_h);
    tensor_shape.set(get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL), out_c);

    return tensor_shape;
}

/** Calculate the deep convolution shape output shape of a tensor
 *
 * @param[in] input_shape       Input tensor shape
 * @param[in] input_data_layout Input data layout
 * @param[in] weights_shape     Weights tensor shape
 * @param[in] conv_info         Contains padding and stride information
 *
 * @return the calculated shape
 */
inline TensorShape compute_deep_convolution_shape(const TensorShape &input_shape, DataLayout input_data_layout, const TensorShape &weights_shape, const PadStrideInfo &conv_info)
{
    const size_t idx_width   = get_data_layout_dimension_index(input_data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_height  = get_data_layout_dimension_index(input_data_layout, DataLayoutDimension::HEIGHT);
    const size_t idx_channel = get_data_layout_dimension_index(input_data_layout, DataLayoutDimension::CHANNEL);

    const unsigned int input_width         = input_shape[idx_width];
    const unsigned int input_height        = input_shape[idx_height];
    const unsigned int weights_width       = weights_shape[idx_width];
    const unsigned int weights_height      = weights_shape[idx_height];
    const unsigned int weights_out_channel = weights_shape[3];
    unsigned int       output_width        = 0;
    unsigned int       output_height       = 0;
    std::tie(output_width, output_height) = scaled_dimensions(input_width, input_height, weights_width, weights_height, conv_info);

    TensorShape output_shape{ input_shape };
    output_shape.set(idx_width, output_width);
    output_shape.set(idx_height, output_height);
    output_shape.set(idx_channel, weights_out_channel);

    return output_shape;
}

/** Calculate the deep convolution shape output shape of a tensor
 *
 * @param[in] input     Input tensor info
 * @param[in] weights   Weights tensor info
 * @param[in] conv_info Contains padding and stride information
 *
 * @return the calculated shape
 */
inline TensorShape compute_deep_convolution_shape(const ITensorInfo &input, const ITensorInfo &weights, const PadStrideInfo &conv_info)
{
    return compute_deep_convolution_shape(input.tensor_shape(), input.data_layout(), weights.tensor_shape(), conv_info);
}

/** Calculate the min/max shape output shape of a tensor
 *
 * @param[in] input Input tensor info
 *
 * @return the calculated shape
 */
inline TensorShape compute_min_max_shape(const ITensorInfo *input)
{
    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(Window::DimX, 2);
    output_shape.remove_dimension(1);
    output_shape.remove_dimension(1);

    return output_shape;
}

/** Calculate the output pool shape of a tensor
 *
 * @param[in] input     Input tensor info
 * @param[in] pool_info Pooling layer info
 *
 * @return the calculated shape
 */
inline TensorShape compute_pool_shape(const ITensorInfo &input, PoolingLayerInfo pool_info)
{
    int pooled_w = 0;
    int pooled_h = 0;

    TensorShape output_shape{ input.tensor_shape() };

    const bool is_global_pooling = pool_info.is_global_pooling;
    const int  idx_width         = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::WIDTH);
    const int  idx_height        = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::HEIGHT);
    const int  input_width       = input.tensor_shape()[idx_width];
    const int  input_height      = input.tensor_shape()[idx_height];
    const int  pool_size_x       = is_global_pooling ? output_shape[idx_width] : pool_info.pool_size.width;
    const int  pool_size_y       = is_global_pooling ? output_shape[idx_height] : pool_info.pool_size.height;

    std::tie(pooled_w, pooled_h) = scaled_dimensions_signed(input_width, input_height,
                                                            pool_size_x, pool_size_y,
                                                            pool_info.pad_stride_info);

    ARM_COMPUTE_ERROR_ON_MSG((pooled_w < 1 || pooled_h < 1), "Calculated output dimension size is invalid");

    output_shape.set(idx_width, static_cast<size_t>(pooled_w));
    output_shape.set(idx_height, static_cast<size_t>(pooled_h));

    return output_shape;
}

/** Calculate the output unpool shape of a tensor
 *
 * @param[in] input     Input tensor info
 * @param[in] pool_info Pooling layer info
 *
 * @return the calculated shape
 */
inline TensorShape compute_unpool_shape(const ITensorInfo &input, PoolingLayerInfo pool_info)
{
    const unsigned int idx_width   = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::WIDTH);
    const unsigned int idx_height  = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::HEIGHT);
    const TensorShape  input_shape = input.tensor_shape();
    ARM_COMPUTE_ERROR_ON(input_shape[idx_height] <= 1 || input_shape[idx_width] <= 1);
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;
    const unsigned int  stride_x        = pad_stride_info.stride().first;
    const unsigned int  stride_y        = pad_stride_info.stride().second;

    const int pad_left   = pad_stride_info.pad_left();
    const int pad_top    = pad_stride_info.pad_top();
    const int pad_right  = pad_stride_info.pad_right();
    const int pad_bottom = pad_stride_info.pad_bottom();

    TensorShape        output_shape = input_shape;
    const unsigned int out_width    = (input_shape[idx_width] - 1) * stride_x - pad_left - pad_right + pool_info.pool_size.width;
    const unsigned int out_height   = (input_shape[idx_height] - 1) * stride_y - pad_top - pad_bottom + pool_info.pool_size.height;

    output_shape.set(idx_width, out_width);
    output_shape.set(idx_height, out_height);
    return output_shape;
}

/** Calculate the output roi align shape of a tensor
 *
 * @param[in] input     Input tensor info
 * @param[in] rois      Rois tensor info
 * @param[in] pool_info Pooling layer info
 *
 * @return the calculated shape
 */
inline TensorShape compute_roi_align_shape(const ITensorInfo &input, const ITensorInfo &rois, ROIPoolingLayerInfo pool_info)
{
    TensorShape output_shape{ input.tensor_shape() };

    const unsigned int idx_width  = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::WIDTH);
    const unsigned int idx_height = get_data_layout_dimension_index(input.data_layout(), DataLayoutDimension::HEIGHT);

    output_shape.set(idx_width, pool_info.pooled_width());
    output_shape.set(idx_height, pool_info.pooled_height());
    output_shape.set(3, rois.dimension(1));

    return output_shape;
}

/** Calculate the RNN shape of a tensor
 *
 * @param[in] input      Input tensor info
 * @param[in] batch_size Batch size
 *
 * @return the calculated shape
 */
inline TensorShape compute_rnn_shape(const ITensorInfo *input, const unsigned int batch_size)
{
    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(1, batch_size);

    return output_shape;
}

/** Calculate the matrix multiplication output shape of two tensors
 *
 * @param[in] input0                    First input tensor info
 * @param[in] input1                    Second input tensor info
 * @param[in] is_interleaved_transposed True if the input is interleaved transposed
 * @param[in] reshape_info              GEMM reshape info
 *
 * @return the calculated shape
 */
inline TensorShape compute_mm_shape(const ITensorInfo &input0, const ITensorInfo &input1, bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_ERROR_ON_MSG(input0.num_dimensions() > 4, "The number of dimensions for the matrix A must be <= 4");
    ARM_COMPUTE_ERROR_ON_MSG(is_interleaved_transposed && reshape_info.reinterpret_input_as_3d(), "The first input tensor cannot be reinterpreted as 3D if is_interleaved_transposed is true");

    const bool reinterpret_input_as_3d  = reshape_info.reinterpret_input_as_3d();
    const bool reinterpret_output_as_3d = reshape_info.depth_output_gemm3d() != 0;
    const int  depth_output_gemm3d      = reinterpret_output_as_3d ? reshape_info.depth_output_gemm3d() : 1;
    const int  m                        = reshape_info.reinterpret_input_as_3d() ? input0.dimension(1) * input0.dimension(2) : input0.dimension(1);

    // If the output of GEMM has to be reinterpreted as 3D, the number of input0 rows (M) is obtained collapsing the second and third
    // dimension of the output tensor
    const int dim0 = is_interleaved_transposed ? reshape_info.n() : input1.dimension(0);
    const int dim1 = is_interleaved_transposed ? reshape_info.m() / depth_output_gemm3d : m / depth_output_gemm3d;
    const int dim2 = reinterpret_input_as_3d ? input0.tensor_shape()[3] : input0.tensor_shape()[2];
    const int dim3 = reinterpret_input_as_3d ? 1 : input0.tensor_shape()[3];

    TensorShape output_shape{ input0.tensor_shape() };

    output_shape.set(0, dim0);
    output_shape.set(1, dim1);
    output_shape.set(2, reinterpret_output_as_3d ? depth_output_gemm3d : dim2);
    output_shape.set(3, reinterpret_output_as_3d ? dim2 : dim3);
    output_shape.set(4, reinterpret_output_as_3d ? dim3 : 1);

    return output_shape;
}

/** Calculate the matrix multiplication output shape of two tensors
 *
 * @param[in] input0    First input tensor info
 * @param[in] input1    Second input tensor info
 * @param[in] gemm_info GEMM reshape info
 *
 * @return the calculated shape
 */
inline TensorShape compute_mm_shape(const ITensorInfo &input0, const ITensorInfo &input1, const GEMMReshapeInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(input1);
    ARM_COMPUTE_ERROR_ON_MSG(input0.num_dimensions() > 4, "The number of dimensions for the matrix A must be <= 4");

    const bool reinterpret_input_as_3d  = gemm_info.reinterpret_input_as_3d();
    const bool reinterpret_output_as_3d = gemm_info.depth_output_gemm3d() != 0;
    const int  depth_output_gemm3d      = reinterpret_output_as_3d ? gemm_info.depth_output_gemm3d() : 1;

    TensorShape output_shape{ input0.tensor_shape() };

    if(!reinterpret_input_as_3d && !reinterpret_output_as_3d)
    {
        output_shape.set(0, gemm_info.n());
        output_shape.set(1, gemm_info.m());
    }
    else
    {
        // If the output of GEMM has to be reinterpreted as 3D, the number of input0 rows (M) is obtained collapsing the second and third
        // dimension of the output tensor
        const int batch_size = reinterpret_input_as_3d ? input0.tensor_shape()[3] : input0.tensor_shape()[2];
        output_shape.set(0, gemm_info.n());
        output_shape.set(1, gemm_info.m() / depth_output_gemm3d);
        output_shape.set(2, reinterpret_output_as_3d ? depth_output_gemm3d : batch_size);
        output_shape.set(3, reinterpret_output_as_3d ? batch_size : 1);
    }

    return output_shape;
}

/** Calculate the matrix multiplication output shape of two tensors
 *
 * @param[in] input0    First input tensor info
 * @param[in] input1    Second input tensor info
 * @param[in] gemm_info GEMM kernel info used to retrieve the original dimensions of the input matrices
 *
 * @return the calculated shape
 */
inline TensorShape compute_mm_shape(const ITensorInfo &input0, const ITensorInfo &input1, const GEMMKernelInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(input1);
    ARM_COMPUTE_ERROR_ON_MSG(input0.num_dimensions() > 4, "The number of dimensions for the matrix A must be <= 4");

    const bool         reinterpret_input_as_3d  = gemm_info.reinterpret_input_as_3d;
    const bool         reinterpret_output_as_3d = gemm_info.depth_output_gemm3d != 0;
    const unsigned int depth_output_gemm3d      = reinterpret_output_as_3d ? gemm_info.depth_output_gemm3d : 1;

    TensorShape output_shape{ input0.tensor_shape() };

    if(!reinterpret_input_as_3d && !reinterpret_output_as_3d)
    {
        output_shape.set(0, gemm_info.n);
        output_shape.set(1, gemm_info.m);
    }
    else
    {
        // If the output of GEMM has to be reinterpreted as 3D, the number of input0 rows (M) is obtained collapsing the second and third
        // dimension of the output tensor
        const unsigned int batch_size = reinterpret_input_as_3d ? input0.tensor_shape()[3] : input0.tensor_shape()[2];
        output_shape.set(0, gemm_info.n);
        output_shape.set(1, gemm_info.m / depth_output_gemm3d);
        output_shape.set(2, reinterpret_output_as_3d ? depth_output_gemm3d : batch_size);
        output_shape.set(3, reinterpret_output_as_3d ? batch_size : 1);
    }

    return output_shape;
}

/** Calculate the matrix multiplication output shape of two tensors
 *
 * @param[in] input           Input tensor info
 * @param[in] gemm_3d_depth   (Optional)  GEMM 3d depth
 * @param[in] batch_size_on_z (Optional) True if batch size is on z axis
 *
 * @return the calculated shape
 */
inline TensorShape compute_output_stage_shape(const ITensorInfo &input, unsigned int gemm_3d_depth = 1, bool batch_size_on_z = false)
{
    ARM_COMPUTE_ERROR_ON(input.data_layout() != DataLayout::NHWC && gemm_3d_depth > 1);

    TensorShape output_shape = input.tensor_shape();
    if(gemm_3d_depth > 1)
    {
        if(batch_size_on_z)
        {
            output_shape.shift_right(1);
        }
        output_shape.set(0, input.tensor_shape().x());
        output_shape.set(1, input.tensor_shape().y() / gemm_3d_depth);
        output_shape.set(2, gemm_3d_depth);
    }

    return output_shape;
}

/** Calculate the strided slice output shape of a tensor
 *
 * @param[in] input            Input tensor info
 * @param[in] starts           The starts of the dimensions of the input tensor to be sliced
 * @param[in] ends             The ends of the dimensions of the input tensor to be sliced
 * @param[in] strides          The strides of the dimensions of the input tensor to be sliced
 * @param[in] begin_mask       If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
 * @param[in] end_mask         If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
 * @param[in] shrink_axis_mask If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1
 *
 * @return the calculated shape
 */
inline TensorShape compute_strided_slice_shape(const ITensorInfo &input,
                                               const Coordinates &starts, const Coordinates &ends, const Coordinates &strides,
                                               int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    using namespace arm_compute::helpers::tensor_transform;
    return compute_strided_slice_output_shape(input.tensor_shape(), starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
}

/** Calculate the slice output shape of a tensor
 *
 * @param[in] input_shape Input tensor info
 * @param[in] starts      The starts of the dimensions of the input tensor to be sliced
 * @param[in] ends        The ends of the dimensions of the input tensor to be sliced
 *
 * @return the calculated shape
 */
inline TensorShape compute_slice_shape(const TensorShape &input_shape, const Coordinates &starts, const Coordinates &ends)
{
    using namespace arm_compute::helpers::tensor_transform;

    return compute_strided_slice_output_shape(input_shape,
                                              starts, ends, BiStrides(),
                                              0, construct_slice_end_mask(ends), 0);
}

/** Calculate the batch to space output shape of a tensor
 *
 * @param[in] input   Input tensor info
 * @param[in] block_x Block shape x value
 * @param[in] block_y Block shape y value
 *
 * @return the calculated shape
 */
inline TensorShape compute_batch_to_space_shape(const ITensorInfo *input, const int block_x, const int block_y)
{
    ARM_COMPUTE_ERROR_ON(block_x <= 0 || block_y <= 0);

    const DataLayout data_layout = input->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_batch   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(idx_width, input->tensor_shape()[idx_width] * block_x);
    output_shape.set(idx_height, input->tensor_shape()[idx_height] * block_y);
    output_shape.set(idx_batch, input->tensor_shape()[idx_batch] / (block_x * block_y));

    return output_shape;
}

/** Calculate the depth to space output shape of a tensor
 *
 * @param[in] input_shape Input tensor shape
 * @param[in] data_layout Operation data layout
 * @param[in] block       Block shape value
 *
 * @return the calculated shape
 */
inline TensorShape compute_depth_to_space_shape(const TensorShape &input_shape, DataLayout data_layout, int block)
{
    ARM_COMPUTE_ERROR_ON(block < 2);

    const int idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    TensorShape output_shape{ input_shape };
    output_shape.set(idx_width, input_shape[idx_width] * block);
    output_shape.set(idx_height, input_shape[idx_height] * block);
    output_shape.set(idx_channel, input_shape[idx_channel] / (block * block));

    return output_shape;
}

/** Calculate the split output shape of a tensor
 *
 * @param[in] input      Input tensor info
 * @param[in] axis       Axis on which to split the input
 * @param[in] num_splits Number of splits
 *
 * @return the calculated shape
 */
inline TensorShape compute_split_shape(const ITensorInfo *input, unsigned int axis, unsigned int num_splits)
{
    TensorShape empty_shape;
    empty_shape.set(0, 0);

    TensorShape out_shape{ input->tensor_shape() };

    // Return empty shape if axis is invalid
    if(axis > input->tensor_shape().num_dimensions())
    {
        return empty_shape;
    }

    size_t axis_size = out_shape[axis];

    // Return empty shape if num_split is not valid
    if(axis_size % num_splits)
    {
        return empty_shape;
    }

    out_shape[axis] = axis_size / num_splits;
    return out_shape;
}

/** Calculate the space to batch output shape of a tensor
 *
 * @param[in] input         Input tensor info
 * @param[in] block_x       Block shape x value
 * @param[in] block_y       Block shape y value
 * @param[in] padding_left  Left padding values
 * @param[in] padding_right Right padding values
 *
 * @return the calculated shape
 */
inline TensorShape compute_space_to_batch_shape(const ITensorInfo *input, const int block_x, const int block_y, const Size2D &padding_left, const Size2D &padding_right)
{
    TensorShape output_shape{ input->tensor_shape() };

    const DataLayout data_layout = input->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_batch   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    ARM_COMPUTE_ERROR_ON((input->tensor_shape()[idx_width] + padding_left.x() + padding_right.x()) % block_x != 0);
    ARM_COMPUTE_ERROR_ON((input->tensor_shape()[idx_height] + padding_left.y() + padding_right.y()) % block_y != 0);

    output_shape.set(idx_width, (input->tensor_shape()[idx_width] + padding_left.x() + padding_right.x()) / block_x);
    output_shape.set(idx_height, (input->tensor_shape()[idx_height] + padding_left.y() + padding_right.y()) / block_y);
    output_shape.set(idx_batch, input->tensor_shape()[idx_batch] * block_x * block_y);

    return output_shape;
}

/** Calculate the space to batch output shape of a tensor
 *
 * @param[in] input       Input tensor info
 * @param[in] block_shape Block shape value
 *
 * @return the calculated shape
 */
inline TensorShape compute_space_to_depth_shape(const ITensorInfo *input, int32_t block_shape)
{
    TensorShape output_shape{ input->tensor_shape() };

    const DataLayout data_layout = input->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_depth   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    output_shape.set(idx_width, input->tensor_shape()[idx_width] * block_shape);
    output_shape.set(idx_height, input->tensor_shape()[idx_height] * block_shape);
    output_shape.set(idx_depth, input->tensor_shape()[idx_depth] / (block_shape * block_shape));

    return output_shape;
}

/** Calculate the prior box output shape of a tensor
 *
 * @param[in] input Input tensor info
 * @param[in] info  PriorBoxLayer info
 *
 * @return the calculated shape
 */
inline TensorShape compute_prior_box_shape(const ITensorInfo &input, const PriorBoxLayerInfo &info)
{
    DataLayout   data_layout = input.data_layout();
    const size_t idx_w       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_h       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int    num_priors  = info.aspect_ratios().size() * info.min_sizes().size() + info.max_sizes().size();

    TensorShape output_shape{};
    output_shape.set(0, input.dimension(idx_w) * input.dimension(idx_h) * num_priors * 4);
    output_shape.set(1, 2);

    return output_shape;
}

/** Calculate the padded shape of a tensor
 *
 * @param[in] input_shape Input tensor shape
 * @param[in] padding     Paddings list
 *
 * @return the calculated shape
 */
inline TensorShape compute_padded_shape(const TensorShape &input_shape, const PaddingList &padding)
{
    TensorShape padded_shape = input_shape;
    for(size_t dim = 0; dim < padding.size(); ++dim)
    {
        const auto    &padding_pair   = padding[dim];
        const uint32_t shape_on_index = (padded_shape.num_dimensions() <= dim) ? 1 : input_shape[dim];
        padded_shape.set(dim, padding_pair.first + shape_on_index + padding_pair.second);
    }
    return padded_shape;
}

/** Calculate the tiled shape of a tensor
 *
 * @param[in] input_shape Input tensor shape
 * @param[in] multiples   Paddings list
 *
 * @return the calculated shape
 */
inline TensorShape compute_tiled_shape(const TensorShape &input_shape, const Multiples &multiples)
{
    TensorShape tiled_shape = input_shape;
    for(size_t dim = 0; dim < multiples.size(); ++dim)
    {
        tiled_shape.set(dim, input_shape[dim] * multiples[dim]);
    }
    return tiled_shape;
}

/** Calculate the reduced shape of a tensor given an axis
 *
 * @param[in] input     Input tensor info
 * @param[in] axis      Axis on which to perform reduction
 * @param[in] keep_dims (Optional) Whether to keep the dimension after reduction operation. Defaults to true.
 *
 * @return the calculated shape
 */
inline TensorShape compute_reduced_shape(const TensorShape &input, unsigned int axis, bool keep_dims = true)
{
    TensorShape output_shape{ input };

    if(!keep_dims)
    {
        output_shape.remove_dimension(axis);
    }
    else
    {
        output_shape.set(axis, 1);
    }

    return output_shape;
}

/** Calculate the upsampled shape of a tensor
 *
 * @param[in] input Input tensor info
 * @param[in] info  Contains stride information (x and y)
 *
 * @return the calculated shape
 */
inline TensorShape compute_upsample_shape(const ITensorInfo &input, const Size2D &info)
{
    const DataLayout data_layout = input.data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    TensorShape        scale_out_shape(input.tensor_shape());
    const unsigned int out_x = input.dimension(idx_width) * info.x();
    const unsigned int out_y = input.dimension(idx_height) * info.y();
    scale_out_shape.set(idx_width, out_x);
    scale_out_shape.set(idx_height, out_y);

    return scale_out_shape;
}

/** Get the tensor shape
 *
 * @param[in] data Input data
 *
 * @return the extracted tensor shape
 */
template <typename T>
inline TensorShape extract_shape(T *data)
{
    return data->info()->tensor_shape();
}

inline TensorShape extract_shape(ITensorInfo *data)
{
    return data->tensor_shape();
}
inline TensorShape extract_shape(const ITensorInfo *data)
{
    return data->tensor_shape();
}

inline TensorShape extract_shape(const TensorShape *data)
{
    return *data;
}

inline TensorShape extract_shape(TensorShape *data)
{
    return *data;
}

/** Calculate the unstack shape of a tensor
 *
 * @param[in] input_shape Input tensor shape
 * @param[in] axis        Axis on which to perform the unstack operation
 *
 * @return the calculated shape
 */
inline TensorShape calculate_unstack_shape(TensorShape input_shape, unsigned int axis)
{
    ARM_COMPUTE_ERROR_ON(axis > input_shape.num_dimensions());
    input_shape.remove_dimension(axis);
    return input_shape;
}

/** Calculate the concatenate output shape of the concatenate operation along a single axis
 *
 * @param[in] input Vector containing the shapes of the inputs
 * @param[in] axis  Axis along which to concatenate the input tensors
 *
 * @return the calculated shape
 */
template <typename T>
inline TensorShape calculate_concatenate_shape(const std::vector<T *> &input, size_t axis)
{
    TensorShape out_shape = extract_shape(input[0]);

#if defined(ARM_COMPUTE_ASSERTS_ENABLED)
    // All dimensions must match except the axis one
    for(unsigned int i = 0; i < MAX_DIMS; ++i)
    {
        if(i == axis)
        {
            continue;
        }

        for(const auto &tensor : input)
        {
            ARM_COMPUTE_ERROR_ON(tensor == nullptr);
            const TensorShape shape = extract_shape(tensor);
            ARM_COMPUTE_ERROR_ON(out_shape[i] != shape[i]);
        }
    }
#endif // defined(ARM_COMPUTE_ASSERTS_ENABLED)

    // Calculate output shape
    size_t new_size = 0;
    for(const auto &tensor : input)
    {
        const TensorShape shape = extract_shape(tensor);
        new_size += shape[axis];
    }

    out_shape.set(axis, new_size);

    return out_shape;
}
/** Calculate the stack output shape of a tensor
 *
 * @param[in] a           Input tensor info
 * @param[in] axis        Axis on which to perform the stack operation
 * @param[in] num_tensors Number of tensors to stack
 *
 * @return the calculated shape
 */
inline TensorShape compute_stack_shape(const ITensorInfo &a, unsigned int axis, unsigned int num_tensors)
{
    ARM_COMPUTE_ERROR_ON(axis > a.num_dimensions());
    ARM_COMPUTE_ERROR_ON(a.num_dimensions() > 4);

    TensorShape shape_out{ a.tensor_shape() };
    shape_out.set(axis, num_tensors);

    unsigned int i_shift = 0;

    for(unsigned int i = 0; i < a.num_dimensions(); ++i)
    {
        if(i == axis)
        {
            i_shift++;
        }

        shape_out.set(i + i_shift, a.tensor_shape()[i]);
    }
    return shape_out;
}

/** Calculate the output shape of 3d Convolution
 *
 * @param[in] src         Input tensor shape
 * @param[in] weights     Weights tensor shape
 * @param[in] conv3d_info 3d Convolution Parameters object
 *
 * @return the calculated shape
 */
inline TensorShape compute_conv3d_shape(const TensorShape &src, const TensorShape &weights, const Conv3dInfo &conv3d_info)
{
    // Weight tensor shape indices (D H W Cin Cout)
    constexpr unsigned int weights_depth_dim  = 4u;
    constexpr unsigned int weights_height_dim = 3u;
    constexpr unsigned int weights_width_dim  = 2u;
    constexpr unsigned int weights_CHout_dim  = 0u;

    // Source/Destination Tensor shape indices (N D H W C)
    constexpr unsigned int batch_dim   = 4u;
    constexpr unsigned int depth_dim   = 3u;
    constexpr unsigned int height_dim  = 2u;
    constexpr unsigned int width_dim   = 1u;
    constexpr unsigned int channel_dim = 0u;

    TensorShape  output_shape{ src };
    const size_t pad_left   = conv3d_info.padding.left;
    const size_t pad_right  = conv3d_info.padding.right;
    const size_t pad_top    = conv3d_info.padding.top;
    const size_t pad_bottom = conv3d_info.padding.bottom;
    const size_t pad_front  = conv3d_info.padding.front;
    const size_t pad_back   = conv3d_info.padding.back;
    const size_t dilation_x = conv3d_info.dilation.width;
    const size_t dilation_y = conv3d_info.dilation.height;
    const size_t dilation_z = conv3d_info.dilation.depth;
    const size_t stride_x   = conv3d_info.stride.x();
    const size_t stride_y   = conv3d_info.stride.y();
    const size_t stride_z   = conv3d_info.stride.z();

    int output_width_size  = 0;
    int output_height_size = 0;
    int output_depth_size  = 0;

    switch(conv3d_info.round_type)
    {
        case DimensionRoundingType::FLOOR:
            output_width_size  = static_cast<int>(std::floor((static_cast<float>(src[width_dim] + pad_left + pad_right - (dilation_x * (weights[weights_width_dim] - 1) + 1)) / stride_x) + 1));
            output_height_size = static_cast<int>(std::floor((static_cast<float>(src[height_dim] + pad_top + pad_bottom - (dilation_y * (weights[weights_height_dim] - 1) + 1)) / stride_y) + 1));
            output_depth_size  = static_cast<int>(std::floor((static_cast<float>(src[depth_dim] + pad_front + pad_back - (dilation_z * (weights[weights_depth_dim] - 1) + 1)) / stride_z) + 1));
            break;
        case DimensionRoundingType::CEIL:
            output_width_size  = static_cast<int>(std::ceil((static_cast<float>(src[width_dim] + pad_left + pad_right - (dilation_x * (weights[weights_width_dim] - 1) + 1)) / stride_x) + 1));
            output_height_size = static_cast<int>(std::ceil((static_cast<float>(src[height_dim] + pad_top + pad_bottom - (dilation_y * (weights[weights_height_dim] - 1) + 1)) / stride_y) + 1));
            output_depth_size  = static_cast<int>(std::ceil((static_cast<float>(src[depth_dim] + pad_front + pad_back - (dilation_z * (weights[weights_depth_dim] - 1) + 1)) / stride_z) + 1));
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported rounding type");
    }

    output_shape.set(batch_dim, src[batch_dim]);
    output_shape.set(width_dim, output_width_size);
    output_shape.set(height_dim, output_height_size);
    output_shape.set(depth_dim, output_depth_size);
    output_shape.set(channel_dim, weights[weights_CHout_dim]);
    return output_shape;
}

/** Calculate the output pool3d shape of a tensor
 *
 * @param[in] src         Input tensor info
 * @param[in] pool3d_info Pooling layer info
 *
 * @return the calculated shape
 */
inline TensorShape compute_pool3d_shape(const TensorShape &src, Pooling3dLayerInfo pool3d_info)
{
    TensorShape output_shape{ src };

    const auto data_layout      = DataLayout::NDHWC;
    const int  idx_width        = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int  idx_depth        = get_data_layout_dimension_index(data_layout, DataLayoutDimension::DEPTH);
    const int  pool_size_width  = pool3d_info.is_global_pooling ? src[idx_width] : pool3d_info.pool_size.width;
    const int  pool_size_height = pool3d_info.is_global_pooling ? src[idx_height] : pool3d_info.pool_size.height;
    const int  pool_size_depth  = pool3d_info.is_global_pooling ? src[idx_depth] : pool3d_info.pool_size.depth;
    int        output_width     = 0;
    int        output_height    = 0;
    int        output_depth     = 0;

    std::tie(output_width, output_height, output_depth) = scaled_3d_dimensions_signed(src[idx_width], src[idx_height], src[idx_depth], pool_size_width, pool_size_height,
                                                                                      pool_size_depth, pool3d_info);

    ARM_COMPUTE_ERROR_ON_MSG((output_width < 1 || output_height < 1 || output_depth < 1), "Calculated output dimension size is invalid");

    output_shape.set(idx_width, static_cast<size_t>(output_width));
    output_shape.set(idx_height, static_cast<size_t>(output_height));
    output_shape.set(idx_depth, static_cast<size_t>(output_depth));

    return output_shape;
}

inline TensorShape compute_gather_shape(const TensorShape &input_shape, const TensorShape &indices_shape, uint32_t actual_axis)
{
    ARM_COMPUTE_ERROR_ON(input_shape.num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON(actual_axis >= input_shape.num_dimensions());

    TensorShape output_shape = input_shape;
    if(indices_shape.num_dimensions() == 1u)
    {
        output_shape[actual_axis] = indices_shape[0];
    }
    else
    {
        const auto inddims{ indices_shape.num_dimensions() };
        output_shape.shift_right(indices_shape.num_dimensions() - 1);
        output_shape[0] = input_shape[0];
        for(size_t idx(1); (idx - 1) < inddims; ++idx)
        {
            output_shape.set(actual_axis + idx, indices_shape[idx - 1], false);
        }
    }
    return output_shape;
}
} // namespace shape_calculator
} // namespace misc
} // namespace arm_compute
#endif /* ARM_COMPUTE_MISC_SHAPE_CALCULATOR_H */
