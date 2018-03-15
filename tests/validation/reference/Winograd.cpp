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
#include "Winograd.h"

#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Utils.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
template <typename T>
void winograd_filter_transform3x3(const SimpleTensor<T> &in, SimpleTensor<T> &out, const Size2D &output_tile)
{
    const bool         is_2x2      = (output_tile.width == 2);
    const unsigned int transf_side = is_2x2 ? 4u : 6u;

    // Simple tensor for the 3x3 input tile
    SimpleTensor<T> input_tile{ TensorShape(3u, 3u), in.data_type(), 1 };

    // Simple tensor for the transformation matrix
    SimpleTensor<T> trans_matrix{ TensorShape(3u, transf_side), in.data_type(), 1 };

    // Simple tensor for the transformation matrix transpose
    SimpleTensor<T> trans_matrix_transposed{ TensorShape(transf_side, 3u), in.data_type(), 1 };

    // Simple tensor for the 3xSide temporary tile
    SimpleTensor<T> tmp_tile{ TensorShape(3u, transf_side), in.data_type(), 1 };

    // Simple tensor for the SidexSide output tile
    SimpleTensor<T> transf_tile{ TensorShape(transf_side, transf_side), in.data_type(), 1 };

    if(is_2x2)
    {
        // Initialize 3x4 transformation matrix
        // 1   | 0   | 0
        // 0.5 | 0.5 | 0.5
        // 0.5 |-0.5 | 0.5
        // 0   | 0   | 1
        trans_matrix[0 + 0 * 3] = 1.0f;
        trans_matrix[1 + 0 * 3] = 0.0f;
        trans_matrix[2 + 0 * 3] = 0.0f;
        trans_matrix[0 + 1 * 3] = 0.5f;
        trans_matrix[1 + 1 * 3] = 0.5f;
        trans_matrix[2 + 1 * 3] = 0.5f;
        trans_matrix[0 + 2 * 3] = 0.5f;
        trans_matrix[1 + 2 * 3] = -0.5f;
        trans_matrix[2 + 2 * 3] = 0.5f;
        trans_matrix[0 + 3 * 3] = 0.0f;
        trans_matrix[1 + 3 * 3] = 0.0f;
        trans_matrix[2 + 3 * 3] = 1.0f;
    }
    else
    {
        // Initialize 3x6 transformation matrix
        //   1/4  |    0   |   0
        //  -1/6  |  -1/6  | -1/6
        //  -1/6  |   1/6  | -1/6
        //  1/24  |  1/12  |  1/6
        //  1/24  | -1/12  |  1/6
        //    0   |    0   |   1
        trans_matrix[0 + 0 * 3] = 1.0f / 4.0f;
        trans_matrix[1 + 0 * 3] = 0.0f;
        trans_matrix[2 + 0 * 3] = 0.0f;
        trans_matrix[0 + 1 * 3] = -1.0f / 6.0f;
        trans_matrix[1 + 1 * 3] = -1.0f / 6.0f;
        trans_matrix[2 + 1 * 3] = -1.0f / 6.0f;
        trans_matrix[0 + 2 * 3] = -1.0f / 6.0f;
        trans_matrix[1 + 2 * 3] = 1.0f / 6.0f;
        trans_matrix[2 + 2 * 3] = -1.0f / 6.0f;
        trans_matrix[0 + 3 * 3] = 1.0f / 24.0f;
        trans_matrix[1 + 3 * 3] = 1.0f / 12.0f;
        trans_matrix[2 + 3 * 3] = 1.0f / 6.0f;
        trans_matrix[0 + 4 * 3] = 1.0f / 24.0f;
        trans_matrix[1 + 4 * 3] = -1.0f / 12.0f;
        trans_matrix[2 + 4 * 3] = 1.0f / 6.0f;
        trans_matrix[0 + 5 * 3] = 0.0f;
        trans_matrix[1 + 5 * 3] = 0.0f;
        trans_matrix[2 + 5 * 3] = 1.0f;
    }

    // Transpose the transformation matrix
    transpose_matrix(trans_matrix, trans_matrix_transposed);

    const int num_channels = in.shape()[2];
    const int num_filters  = in.shape()[3];
    const int num_batches  = in.shape().total_size() / (9 * num_channels * num_filters);

    for(int n = 0; n < num_batches; ++n)
    {
        for(int w = 0; w < num_filters; ++w)
        {
            for(int z = 0; z < num_channels; ++z)
            {
                // Load the 3x3 tile from the input tensor
                get_tile(in, input_tile, Coordinates(0, 0, z, w, n));

                // First transformation
                matrix_multiply(trans_matrix, input_tile, tmp_tile);

                // Second transformation
                matrix_multiply(tmp_tile, trans_matrix_transposed, transf_tile);

                // Store the 4x4 output tile across the 16 channels
                const int output_offset = w + z * num_filters;

                for(unsigned int out_h = 0, out_pos = 0; out_h < transf_side; ++out_h)
                {
                    for(unsigned int out_w = 0; out_w < transf_side; ++out_w, ++out_pos)
                    {
                        out[output_offset + out_pos * num_filters * num_channels] = transf_tile[out_w + out_h * transf_side];
                    }
                }
            }
        }
    }
}

template <typename T>
void winograd_input_transform3x3(const SimpleTensor<T> &src, SimpleTensor<T> &dst, const PadStrideInfo &conv_info)
{
    TensorShape shape4x4(4u, 4u);

    // Simple tensor for the 4x4 input tile
    SimpleTensor<T> src_tile{ shape4x4, src.data_type() };

    // Simple tensor for the 4x4 temporary tile
    SimpleTensor<T> tmp_tile{ shape4x4, src.data_type() };

    // Simple tensor for the 4x4 output tile
    SimpleTensor<T> dst_tile{ shape4x4, src.data_type() };

    // Simple tensor for the transformation matrix
    SimpleTensor<T> matrix{ shape4x4, src.data_type() };

    // Simple tensor for the transformation matrix transposed
    SimpleTensor<T> matrix_transposed{ shape4x4, src.data_type() };

    const float matrix_values[] = { 1.f, 0.f, -1.f, 0.f,
                                    0.f, 1.f, 1.f, 0.f,
                                    0.f, -1.f, 1.f, 0.f,
                                    0.f, 1.f, 0.f, -1.f
                                  };

    for(int i = 0; i < matrix.num_elements(); ++i)
    {
        matrix[i] = matrix_values[i];
    }

    transpose_matrix(matrix, matrix_transposed);

    const int in_w        = src.shape().x();
    const int in_h        = src.shape().y();
    const int in_d        = src.shape().z();
    const int num_batches = src.shape().total_size() / (in_w * in_h * in_d);
    const int num_tiles_x = std::ceil((in_w - 2 + conv_info.pad_left() + conv_info.pad_right()) / 2.0f);
    const int num_tiles_y = std::ceil((in_h - 2 + conv_info.pad_top() + conv_info.pad_bottom()) / 2.0f);

    ARM_COMPUTE_ERROR_ON((num_tiles_x * num_tiles_y) != static_cast<int>(dst.shape().y()));

    for(int b = 0; b < num_batches; ++b)
    {
        for(int z = 0; z < in_d; ++z)
        {
            for(int y = 0; y < num_tiles_y; ++y)
            {
                for(int x = 0; x < num_tiles_x; ++x)
                {
                    int xi = x * 2 - conv_info.pad_left();
                    int yi = y * 2 - conv_info.pad_top();

                    // Get the 4x4 tile from the input tensor
                    get_tile(src, src_tile, Coordinates(xi, yi, z, b));

                    // Compute the transformation
                    matrix_multiply(matrix, src_tile, tmp_tile);
                    matrix_multiply(tmp_tile, matrix_transposed, dst_tile);

                    // Store the 4x4 output tile across the 16 channels
                    for(int i = 0; i < 16; ++i)
                    {
                        int xo = z;
                        int yo = x + y * num_tiles_x;
                        dst[coords2index(dst.shape(), Coordinates(xo, yo, i, b))] = dst_tile[i];
                    }
                }
            }
        }
    }
}

template <typename T>
void winograd_output_transform3x3(const SimpleTensor<T> &in, SimpleTensor<T> &out, int num_tiles_x)
{
    ARM_COMPUTE_ERROR_ON(in.shape()[2] != 16);
    ARM_COMPUTE_ERROR_ON(in.shape()[0] != out.shape()[2]);

    // Simple tensor for the 3x3 input tile
    SimpleTensor<T> input_tile{ TensorShape(4u, 4u), in.data_type(), 1 };

    // Simple tensor for the transformation matrix
    SimpleTensor<T> trans_matrix{ TensorShape(4u, 2u), in.data_type(), 1 };

    // Simple tensor for the transformation matrix transpose
    SimpleTensor<T> trans_matrix_transposed{ TensorShape(2u, 4u), in.data_type(), 1 };

    // Simple tensor for the 4x3 temporary tile
    SimpleTensor<T> tmp_tile{ TensorShape(4u, 2u), in.data_type(), 1 };

    // Simple tensor for the 4x4 output tile
    SimpleTensor<T> output_tile{ TensorShape(2u, 2u), in.data_type(), 1 };

    // Initialize transformation matrix
    // 1   | 1   | 1   | 1
    // 0   | 1   | -1  | -1
    trans_matrix[0 + 0 * 4] = 1.0f;
    trans_matrix[1 + 0 * 4] = 1.0f;
    trans_matrix[2 + 0 * 4] = 1.0f;
    trans_matrix[3 + 0 * 4] = 0.0f;
    trans_matrix[0 + 1 * 4] = 0.0f;
    trans_matrix[1 + 1 * 4] = 1.0f;
    trans_matrix[2 + 1 * 4] = -1.0f;
    trans_matrix[3 + 1 * 4] = -1.0f;

    // Transpose the transformation matrix
    transpose_matrix(trans_matrix, trans_matrix_transposed);

    const int w_in        = in.shape()[0];
    const int h_in        = in.shape()[1];
    const int c_in        = in.shape()[2];
    const int w_out       = out.shape()[0];
    const int h_out       = out.shape()[1];
    const int c_out       = out.shape()[2];
    const int num_batches = in.shape().total_size() / (w_in * h_in * c_in);

    // Input strides
    const int stridey_in = w_in;
    const int stridez_in = stridey_in * h_in;
    const int stridew_in = stridez_in * c_in;

    // Output strides
    const int stridey_out = w_out;
    const int stridez_out = stridey_out * h_out;
    const int stridew_out = stridez_out * c_out;

    for(int n = 0; n < num_batches; ++n)
    {
        for(int y = 0; y < h_in; ++y)
        {
            for(int x = 0; x < w_in; ++x)
            {
                // Load the 4x4 tile across the 16 channels of the input tensor
                for(int z = 0; z < c_in; ++z)
                {
                    input_tile[z] = in[x + (y * stridey_in) + (z * stridez_in) + (n * stridew_in)];
                }

                // First transformation
                matrix_multiply(trans_matrix, input_tile, tmp_tile);

                // Second transformation
                matrix_multiply(tmp_tile, trans_matrix_transposed, output_tile);

                // Store the 2x2 output tile
                const int xo = (y % num_tiles_x) * 2;
                const int yo = (y / num_tiles_x) * 2;
                const int zo = x;

                const int output_offset                  = xo + (yo * stridey_out) + (zo * stridez_out) + (n * stridew_out);
                out[output_offset + 0 * stridey_out + 0] = output_tile[0 + 0 * 2];

                // Check out-of-bound writes
                if(xo + 1 < w_out)
                {
                    out[output_offset + 0 * stridey_out + 1] = output_tile[1 + 0 * 2];
                }

                if(yo + 1 < h_out)
                {
                    out[output_offset + 1 * stridey_out + 0] = output_tile[0 + 1 * 2];
                }

                if((yo + 1 < h_out) && (xo + 1 < w_out))
                {
                    out[output_offset + 1 * stridey_out + 1] = output_tile[1 + 1 * 2];
                }
            }
        }
    }
}
} // namespace

template <typename T>
SimpleTensor<T> winograd_input_transform(const SimpleTensor<T> &src, const TensorShape &dst_shape, const PadStrideInfo &conv_info, const Size2D &kernel_dims)
{
    ARM_COMPUTE_ERROR_ON(kernel_dims.width != kernel_dims.height);
    ARM_COMPUTE_ERROR_ON(src.data_layout() != DataLayout::NCHW);

    SimpleTensor<T> dst{ dst_shape, src.data_type() };

    switch(kernel_dims.width)
    {
        case 3:
            winograd_input_transform3x3(src, dst, conv_info);
            break;
        default:
            ARM_COMPUTE_ERROR("Only 3x3 kernels are supported");
    }

    return dst;
}

template <typename T>
SimpleTensor<T> winograd_filter_transform(const SimpleTensor<T> &in, const TensorShape &output_shape, const Size2D &output_tile)
{
    ARM_COMPUTE_ERROR_ON_MSG(in.data_layout() != DataLayout::NCHW, "Only supported NCHW data format");

    // Create reference
    SimpleTensor<T> out{ output_shape, in.data_type(), 1 };

    switch(in.shape()[0])
    {
        case 3:
            winograd_filter_transform3x3(in, out, output_tile);
            break;
        default:
            ARM_COMPUTE_ERROR("Only supported 3x3 kernel");
            break;
    }

    return out;
}

template <typename T>
SimpleTensor<T> winograd_output_transform(const SimpleTensor<T> &in, const TensorShape &output_shape, const Size2D &kernel_dims, const Size2D &num_tiles)
{
    ARM_COMPUTE_ERROR_ON_MSG(in.data_layout() != DataLayout::NCHW, "Only supported NCHW data format");
    ARM_COMPUTE_ERROR_ON(kernel_dims.width != kernel_dims.height);
    ARM_COMPUTE_ERROR_ON(in.shape()[1] != num_tiles.area());

    // Create reference
    SimpleTensor<T> out{ output_shape, in.data_type(), 1 };

    switch(kernel_dims.width)
    {
        case 3:
            winograd_output_transform3x3(in, out, num_tiles.width);
            break;
        default:
            ARM_COMPUTE_ERROR("Only supported 3x3 kernel");
            break;
    }

    return out;
}

template SimpleTensor<float> winograd_input_transform(const SimpleTensor<float> &src, const TensorShape &dst_shape, const PadStrideInfo &conv_info, const Size2D &kernel_dims);
template SimpleTensor<float> winograd_filter_transform(const SimpleTensor<float> &in, const TensorShape &output_shape, const Size2D &output_tile);
template SimpleTensor<float> winograd_output_transform(const SimpleTensor<float> &in, const TensorShape &output_shape, const Size2D &kernel_dims, const Size2D &num_tiles);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
