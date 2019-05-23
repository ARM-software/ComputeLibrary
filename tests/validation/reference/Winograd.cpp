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
#include "Winograd.h"

#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Utils.h"

#include "arm_compute/core/Types.h"

#include <algorithm>
#include <cmath>

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
void initialize_matrix_transform(SimpleTensor<T> &src, const Size2D &output_tile_size, const Size2D &kernel_size, WinogradTransformType winograd_transform_type)
{
    // Winograd input transform matrices
    static const std::array<float, 16> imatrix2x2_3x3 =
    {
        1.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
        0.0f, -1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, -1.0f
    };

    static const std::array<float, 36> imatrix4x4_3x3 =
    {
        4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f,
        0.0f, -4.0f, -4.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 4.0f, -4.0f, -1.0f, 1.0f, 0.0f,
        0.0f, -2.0f, -1.0f, 2.0f, 1.0f, 0.0f,
        0.0f, 2.0f, -1.0f, -2.0f, 1.0f, 0.0f,
        0.0f, 4.0f, 0.0f, -5.0f, 0.0f, 1.0f,
    };

    static const std::array<float, 64> imatrix4x4_5x5 =
    {
        1.f, 0.f, -21.f / 4.f, 0.f, 21.f / 4.f, 0.f, -1.f, 0.f,
        0.f, 1.f, 1.f, -17.f / 4.f, -17.f / 4.f, 1.f, 1.f, 0.f,
        0.f, -1.f, 1.f, 17.f / 4.f, -17.f / 4.f, -1.f, 1.f, 0.f,
        0.f, 1.f / 2.f, 1.f / 4.f, -5.f / 2.f, -5.f / 4.f, 2.f, 1.f, 0.f,
        0.f, -1.f / 2.f, 1.f / 4.f, 5.f / 2.f, -5.f / 4.f, -2.f, 1.f, 0.f,
        0.f, 2.f, 4.f, -5.f / 2.f, -5.f, 1.f / 2.f, 1.f, 0.f,
        0.f, -2.f, 4.f, 5.f / 2.f, -5.f, -1.f / 2.f, 1.f, 0.f,
        0.f, -1.f, 0.f, 21.f / 4.f, 0.f, -21.f / 4.f, 0.f, 1.f
    };

    static const std::array<float, 64> imatrix2x1_7x7 =
    {
        -36.0f, 0.0f, 49.0f, 0.0f, -14.0f, 0.0f, 1.0f, 0.0f,
        0.0f, -36.0f, 36.0f, 13.0f, -13.0f, -1.0f, 1.0f, 0.0f,
        0.0f, 36.0f, 36.0f, -13.0f, -13.0f, 1.0f, 1.0f, 0.0f,
        0.0f, -18.0f, 9.0f, 20.0f, -10.0f, -2.0f, 1.0f, 0.0f,
        0.0f, 18.0f, 9.0f, -20.0f, -10.0f, 2.0f, 1.0f, 0.0f,
        0.0f, -12.0f, 4.0f, 15.0f, -5.0f, -3.0f, 1.0f, 0.0f,
        0.0f, 12.0f, 4.0f, -15.0f, -5.0f, 3.0f, 1.0f, 0.0f,
        0.0f, -36.0f, 0.0f, 49.0f, 0.0f, -14.0f, 0.0f, 1.0f
    };

    // ------------------------------------------

    // Winograd filter transform matrices
    static const std::array<float, 12> fmatrix2x2_3x3 =
    {
        1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f,
        0.5f, -0.5f, 0.5f,
        0.0f, 0.0f, 1.0f
    };

    static const std::array<float, 18> fmatrix4x4_3x3 =
    {
        0.25f, 0.0f, 0.0f,
        -1.0f / 6.0f, -1.0f / 6.0f, -1.0f / 6.0f,
        -1.0f / 6.0f, 1.0f / 6.0f, -1.0f / 6.0f,
        1.0f / 24.0f, 1.0f / 12.0f, 1.0f / 6.0f,
        1.0f / 24.0f, -1.0f / 12.0f, 1.0f / 6.0f,
        0.0f, 0.0f, 1.0f
    };

    static const std::array<float, 40> fmatrix4x4_5x5 =
    {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        -2.0f / 9.0f, -2.0f / 9.0f, -2.0f / 9.0f, -2.0f / 9.0f, -2.0f / 9.0f,
        -2.0f / 9.0f, 2.0f / 9.0f, -2.0f / 9.0f, 2.0f / 9.0f, -2.0f / 9.0f,
        1.0f / 90.0f, 1.0f / 45.0f, 2.0f / 45.0f, 4.0f / 45.0f, 8.0f / 45.0f,
        1.0f / 90.0f, -1.0f / 45.0f, 2.0f / 45.0f, -4.0f / 45.0f, 8.0f / 45.0f,
        4.0f / 45.0f, 2.0f / 45.0f, 1.0f / 45.0f, 1.0f / 90.0f, 1.0f / 180.0f,
        4.0f / 45.0f, -2.0f / 45.0f, 1.0f / 45.0f, -1.0f / 90.0f, 1.0f / 180.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f

    };

    static const std::array<float, 56> fmatrix2x1_7x7 =
    {
        -1.0f / 36.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        1.0f / 48.0f, -1.0f / 48.0f, 1.0f / 48.0f, -1.0f / 48.0f, 1.0f / 48.0f, -1.0f / 48.0f, 1.0f / 48.0f,
        1.0f / 48.0f, 1.0f / 48.0f, 1.0f / 48.0f, 1.0f / 48.0f, 1.0f / 48.0f, 1.0f / 48.0f, 1.0f / 48.0f,
        -1.0f / 120.0f, 1.0f / 60.0f, -1.0f / 30.0f, 1.0f / 15.0f, -2.0f / 15.0f, 4.0f / 15.0f, -8.0f / 15.0f,
        -1.0f / 120.0f, -1.0f / 60.0f, -1.0f / 30.0f, -1.0f / 15.0f, -2.0f / 15.0f, -4.0f / 15.0f, -8.0f / 15.0f,
        1.0f / 720.0f, -1.0f / 240.0f, 1.0f / 80.0f, -3.0f / 80.0f, 9.0f / 80.0f, -27.0f / 80.0f, 81.0f / 80.0f,
        1.0f / 720.0f, 1.0f / 240.0f, 1.0f / 80.0f, 3.0f / 80.0f, 9.0f / 80.0f, 27.0f / 80.0f, 81.0f / 80.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f
    };

    // ------------------------------------------

    // Winograd output transform matrices
    static const std::array<float, 8> omatrix2x2_3x3 =
    {
        1.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, -1.0f, -1.0f
    };

    static const std::array<float, 24> omatrix4x4_3x3 =
    {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 4.0f, 4.0f, 0.0f,
        0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f
    };

    static const std::array<float, 36> omatrix4x4_5x5 =
    {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 8.0f, 0.0f,
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 4.0f, 4.0f, 2.0f, 2.0f, 0.0f,
        0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f, -1.0f, 1.0f
    };

    static const std::array<float, 16> omatrix2x1_7x7 =
    {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        0.0f, -1.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 1.0f
    };

    // ------------------------------------------

    using WinogradKey = std::tuple<std::pair<int, int>, std::pair<int, int>, WinogradTransformType>;

    // Key = (Output tile size, Kernel size, Winograd transform type)
    static std::map<WinogradKey, const float *> matrix_map =
    {
        { WinogradKey(std::pair<int, int>(2, 2), std::pair<int, int>(3, 3), WinogradTransformType::INPUT), imatrix2x2_3x3.data() },
        { WinogradKey(std::pair<int, int>(4, 4), std::pair<int, int>(3, 3), WinogradTransformType::INPUT), imatrix4x4_3x3.data() },
        { WinogradKey(std::pair<int, int>(2, 1), std::pair<int, int>(3, 1), WinogradTransformType::INPUT), imatrix2x2_3x3.data() },
        { WinogradKey(std::pair<int, int>(4, 1), std::pair<int, int>(3, 1), WinogradTransformType::INPUT), imatrix4x4_3x3.data() },
        { WinogradKey(std::pair<int, int>(1, 2), std::pair<int, int>(1, 3), WinogradTransformType::INPUT), imatrix2x2_3x3.data() },
        { WinogradKey(std::pair<int, int>(1, 4), std::pair<int, int>(1, 3), WinogradTransformType::INPUT), imatrix4x4_3x3.data() },
        { WinogradKey(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5), WinogradTransformType::INPUT), imatrix4x4_5x5.data() },
        { WinogradKey(std::pair<int, int>(4, 1), std::pair<int, int>(5, 1), WinogradTransformType::INPUT), imatrix4x4_5x5.data() },
        { WinogradKey(std::pair<int, int>(2, 1), std::pair<int, int>(7, 1), WinogradTransformType::INPUT), imatrix2x1_7x7.data() },
        { WinogradKey(std::pair<int, int>(1, 2), std::pair<int, int>(1, 7), WinogradTransformType::INPUT), imatrix2x1_7x7.data() },
        { WinogradKey(std::pair<int, int>(2, 2), std::pair<int, int>(7, 7), WinogradTransformType::INPUT), imatrix2x1_7x7.data() },
        { WinogradKey(std::pair<int, int>(1, 4), std::pair<int, int>(1, 5), WinogradTransformType::INPUT), imatrix4x4_5x5.data() },
        { WinogradKey(std::pair<int, int>(2, 2), std::pair<int, int>(3, 3), WinogradTransformType::FILTER), fmatrix2x2_3x3.data() },
        { WinogradKey(std::pair<int, int>(4, 4), std::pair<int, int>(3, 3), WinogradTransformType::FILTER), fmatrix4x4_3x3.data() },
        { WinogradKey(std::pair<int, int>(2, 1), std::pair<int, int>(3, 1), WinogradTransformType::FILTER), fmatrix2x2_3x3.data() },
        { WinogradKey(std::pair<int, int>(4, 1), std::pair<int, int>(3, 1), WinogradTransformType::FILTER), fmatrix4x4_3x3.data() },
        { WinogradKey(std::pair<int, int>(1, 2), std::pair<int, int>(1, 3), WinogradTransformType::FILTER), fmatrix2x2_3x3.data() },
        { WinogradKey(std::pair<int, int>(1, 4), std::pair<int, int>(1, 3), WinogradTransformType::FILTER), fmatrix4x4_3x3.data() },
        { WinogradKey(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5), WinogradTransformType::FILTER), fmatrix4x4_5x5.data() },
        { WinogradKey(std::pair<int, int>(4, 1), std::pair<int, int>(5, 1), WinogradTransformType::FILTER), fmatrix4x4_5x5.data() },
        { WinogradKey(std::pair<int, int>(2, 1), std::pair<int, int>(7, 1), WinogradTransformType::FILTER), fmatrix2x1_7x7.data() },
        { WinogradKey(std::pair<int, int>(1, 2), std::pair<int, int>(1, 7), WinogradTransformType::FILTER), fmatrix2x1_7x7.data() },
        { WinogradKey(std::pair<int, int>(2, 2), std::pair<int, int>(7, 7), WinogradTransformType::FILTER), fmatrix2x1_7x7.data() },
        { WinogradKey(std::pair<int, int>(1, 4), std::pair<int, int>(1, 5), WinogradTransformType::FILTER), fmatrix4x4_5x5.data() },
        { WinogradKey(std::pair<int, int>(2, 2), std::pair<int, int>(3, 3), WinogradTransformType::OUTPUT), omatrix2x2_3x3.data() },
        { WinogradKey(std::pair<int, int>(4, 4), std::pair<int, int>(3, 3), WinogradTransformType::OUTPUT), omatrix4x4_3x3.data() },
        { WinogradKey(std::pair<int, int>(2, 1), std::pair<int, int>(3, 1), WinogradTransformType::OUTPUT), omatrix2x2_3x3.data() },
        { WinogradKey(std::pair<int, int>(4, 1), std::pair<int, int>(3, 1), WinogradTransformType::OUTPUT), omatrix4x4_3x3.data() },
        { WinogradKey(std::pair<int, int>(1, 2), std::pair<int, int>(1, 3), WinogradTransformType::OUTPUT), omatrix2x2_3x3.data() },
        { WinogradKey(std::pair<int, int>(1, 4), std::pair<int, int>(1, 3), WinogradTransformType::OUTPUT), omatrix4x4_3x3.data() },
        { WinogradKey(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5), WinogradTransformType::OUTPUT), omatrix4x4_5x5.data() },
        { WinogradKey(std::pair<int, int>(4, 1), std::pair<int, int>(5, 1), WinogradTransformType::OUTPUT), omatrix4x4_5x5.data() },
        { WinogradKey(std::pair<int, int>(2, 1), std::pair<int, int>(7, 1), WinogradTransformType::OUTPUT), omatrix2x1_7x7.data() },
        { WinogradKey(std::pair<int, int>(1, 2), std::pair<int, int>(1, 7), WinogradTransformType::OUTPUT), omatrix2x1_7x7.data() },
        { WinogradKey(std::pair<int, int>(2, 2), std::pair<int, int>(7, 7), WinogradTransformType::OUTPUT), omatrix2x1_7x7.data() },
        { WinogradKey(std::pair<int, int>(1, 4), std::pair<int, int>(1, 5), WinogradTransformType::OUTPUT), omatrix4x4_5x5.data() },
    };

    // Find transformation matrix
    std::map<WinogradKey, const float *>::iterator it;

    it = matrix_map.find(WinogradKey(std::pair<int, int>(output_tile_size.width, output_tile_size.height),
                                     std::pair<int, int>(kernel_size.width, kernel_size.height),
                                     winograd_transform_type));

    float const *matrix_values = nullptr;
    if(it != matrix_map.end())
    {
        // Get matrix pointer
        matrix_values = it->second;
    }
    else
    {
        ARM_COMPUTE_ERROR("Winograd configuration not supported");
    }

    // Copy values
    std::copy(&matrix_values[0], &matrix_values[0] + src.num_elements(), &src[0]);
}
} // namespace

template <typename T>
SimpleTensor<T> winograd_input_transform(const SimpleTensor<T> &in, const TensorShape &output_shape, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_ERROR_ON(in.data_layout() != DataLayout::NCHW);

    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const Size2D        kernel_size      = winograd_info.kernel_size;

    SimpleTensor<T> out{ output_shape, in.data_type() };

    // Calculate dimensions for the tile
    const unsigned int tile_w = output_tile_size.width + kernel_size.width - 1;
    const unsigned int tile_h = output_tile_size.height + kernel_size.height - 1;

    // Get the maximum dimension from the tile size
    const unsigned int tile_max_dim = std::max(tile_w, tile_h);

    TensorShape tile_dims(tile_max_dim, tile_max_dim);

    // Simple tensor for the input tile
    SimpleTensor<T> src_tile{ tile_dims, in.data_type() };

    // Simple tensor for the temporary tile
    SimpleTensor<T> tmp_tile{ tile_dims, in.data_type() };

    // Simple tensor for the output tile
    SimpleTensor<T> dst_tile{ tile_dims, in.data_type() };

    // Simple tensor for the transformation matrix
    SimpleTensor<T> matrix{ tile_dims, in.data_type() };

    // Simple tensor for the transformation matrix transposed
    SimpleTensor<T> matrix_transposed{ tile_dims, in.data_type() };

    // Initialize matrix for the input transform
    initialize_matrix_transform(matrix, output_tile_size, kernel_size, WinogradTransformType::INPUT);

    // Transpose matrix
    transpose_matrix<T>(matrix, matrix_transposed);

    const int in_w        = in.shape().x();
    const int in_h        = in.shape().y();
    const int in_d        = in.shape().z();
    const int out_d       = out.shape().z();
    const int num_batches = in.shape().total_size() / (in_w * in_h * in_d);
    const int step_x      = output_tile_size.width;
    const int step_y      = output_tile_size.height;

    // Compute the number of output tiles along the x and y direction of size "output_tile_size"
    const Size2D num_tiles = compute_winograd_convolution_tiles(Size2D(in_w, in_h),
                                                                kernel_size,
                                                                output_tile_size,
                                                                conv_info);

    const int num_tiles_x = num_tiles.width;
    const int num_tiles_y = num_tiles.height;

    // In case of 1D convolution, the input tile has to be partially filled with zeros
    int start_x_zero = 0;
    int start_y_zero = 0;
    int end_x_zero   = 0;
    int end_y_zero   = 0;

    if(output_tile_size.width == 1)
    {
        start_x_zero = 1;
        start_y_zero = 0;
        end_x_zero   = tile_max_dim - 1;
        end_y_zero   = tile_max_dim;
    }
    else if(output_tile_size.height == 1)
    {
        start_x_zero = 0;
        start_y_zero = 1;
        end_x_zero   = tile_max_dim;
        end_y_zero   = tile_max_dim - 1;
    }

    // Set the anchor and shape of the zeros area
    const Coordinates anchor_zeros(start_x_zero, start_y_zero);
    const TensorShape shape_zeros(end_x_zero, end_y_zero);

    // If we have a vertical filter (i.e. 1x3, 1x5,..), we need to take the elements along the y direction (step = width of the output tile)
    const int step_y_transf_tile = kernel_size.width == 1 ? tile_max_dim : 1;

    ARM_COMPUTE_ERROR_ON((num_tiles_x * num_tiles_y) != static_cast<int>(out.shape().y()));

    for(int b = 0; b < num_batches; ++b)
    {
        for(int z = 0; z < in_d; ++z)
        {
            for(int y = 0; y < num_tiles_y; ++y)
            {
                for(int x = 0; x < num_tiles_x; ++x)
                {
                    int xi = x * step_x - conv_info.pad_left();
                    int yi = y * step_y - conv_info.pad_top();

                    // Get the tile from the input tensor
                    get_tile<T>(in, src_tile, Coordinates(xi, yi, z, b));

                    // Fill partially with zeros in case of 1D convolution
                    zeros<T>(src_tile, anchor_zeros, shape_zeros);

                    // Compute the transformation
                    matrix_multiply<T>(matrix, src_tile, tmp_tile);
                    matrix_multiply<T>(tmp_tile, matrix_transposed, dst_tile);

                    // Store the output tile across the channels
                    for(int i = 0; i < out_d; ++i)
                    {
                        int xo = z;
                        int yo = x + y * num_tiles_x;
                        out[coords2index(out.shape(), Coordinates(xo, yo, i, b))] = dst_tile[i * step_y_transf_tile];
                    }
                }
            }
        }
    }

    return out;
}

template <typename T>
SimpleTensor<T> winograd_filter_transform(const SimpleTensor<T> &in, const TensorShape &output_shape, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_ERROR_ON_MSG(in.data_layout() != DataLayout::NCHW, "Only supported NCHW data format");

    // Create reference
    SimpleTensor<T> out{ output_shape, in.data_type(), 1 };

    const Size2D output_tile_size = winograd_info.output_tile_size;
    const Size2D kernel_size      = winograd_info.kernel_size;

    // Calculate dimensions for the tile
    const unsigned int input_tile_w    = output_tile_size.width + kernel_size.width - 1;
    const unsigned int input_tile_h    = output_tile_size.height + kernel_size.height - 1;
    const unsigned int input_tile_area = input_tile_w * input_tile_h;

    // Get the maximum dimension from the filter size
    const unsigned int kernel_max_dim = std::max(kernel_size.width, kernel_size.height);

    // Get the maximum dimension from the input tile
    const unsigned int input_tile_max_dim = std::max(input_tile_w, input_tile_h);

    // Simple tensor for the input tile
    SimpleTensor<T> input_tile{ TensorShape(kernel_max_dim, kernel_max_dim), in.data_type(), 1 };

    // Simple tensor for the transformation matrix
    SimpleTensor<T> trans_matrix{ TensorShape(kernel_max_dim, input_tile_max_dim), in.data_type(), 1 };

    // Simple tensor for the transformation matrix transpose
    SimpleTensor<T> trans_matrix_transposed{ TensorShape(input_tile_max_dim, kernel_max_dim), in.data_type(), 1 };

    // Simple tensor for the temporary tile
    SimpleTensor<T> tmp_tile{ TensorShape(kernel_max_dim, input_tile_max_dim), in.data_type(), 1 };

    // Simple tensor for the output tile
    SimpleTensor<T> transf_tile{ TensorShape(input_tile_max_dim, input_tile_max_dim), in.data_type(), 1 };

    // Initialize matrix for the filter transform
    initialize_matrix_transform(trans_matrix, output_tile_size, kernel_size, WinogradTransformType::FILTER);

    // Transpose the transformation matrix
    transpose_matrix<T>(trans_matrix, trans_matrix_transposed);

    const int num_channels = in.shape()[2];
    const int num_filters  = in.shape()[3];
    const int num_batches  = in.shape().total_size() / (kernel_size.area() * num_channels * num_filters);

    // If we have a vertical filter (i.e. 1x3, 1x5,..), we need to take the elements along the y direction (step_y_transf_tile = width of the output tile)
    const int step_y_transf_tile = kernel_size.width == 1 ? input_tile_max_dim : 1;

    for(int n = 0; n < num_batches; ++n)
    {
        for(int w = 0; w < num_filters; ++w)
        {
            for(int z = 0; z < num_channels; ++z)
            {
                // Load the tile from the input tensor
                get_tile<T>(in, input_tile, Coordinates(0, 0, z, w, n));

                // First transformation
                matrix_multiply<T>(trans_matrix, input_tile, tmp_tile);

                // Second transformation
                matrix_multiply<T>(tmp_tile, trans_matrix_transposed, transf_tile);

                // Store the output tile across the channels
                const int output_offset = w + z * num_filters;

                // Store the values across the channels
                for(unsigned int i = 0; i < input_tile_area; ++i)
                {
                    out[output_offset + i * num_filters * num_channels] = transf_tile[i * step_y_transf_tile];
                }
            }
        }
    }

    return out;
}

template <typename T>
SimpleTensor<T> winograd_output_transform(const SimpleTensor<T> &in, const SimpleTensor<T> &b, const TensorShape &output_shape, const WinogradInfo &winograd_info)
{
    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        input_dimensions = winograd_info.input_dimensions;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const Size2D        kernel_size      = winograd_info.kernel_size;

    // Create reference
    SimpleTensor<T> out{ output_shape, in.data_type(), 1 };

    // Calculate dimensions for the tiles
    const unsigned int in_tile_w  = output_tile_size.width + kernel_size.width - 1;
    const unsigned int in_tile_h  = output_tile_size.height + kernel_size.height - 1;
    const unsigned int out_tile_w = output_tile_size.width;
    const unsigned int out_tile_h = output_tile_size.height;

    ARM_COMPUTE_ERROR_ON(in.shape()[2] != (in_tile_w * in_tile_h));
    ARM_COMPUTE_ERROR_ON(in.shape()[0] != out.shape()[get_data_layout_dimension_index(winograd_info.output_data_layout, DataLayoutDimension::CHANNEL)]);

    // Get the maximum dimension from the tile size
    const unsigned int in_tile_max_dim  = std::max(in_tile_w, in_tile_h);
    const unsigned int out_tile_max_dim = std::max(output_tile_size.width, output_tile_size.height);

    // Compute tile dimensions
    // Input tile dimensions
    TensorShape in_tile_dims(in_tile_max_dim, in_tile_max_dim);

    // Output tile dimensions
    TensorShape out_tile_dims(out_tile_max_dim, out_tile_max_dim);

    // Transformation matrix dimensions
    TensorShape tr_tile_dims(in_tile_max_dim, out_tile_max_dim);

    // Create tensors
    // Simple tensor for the input tile
    SimpleTensor<T> input_tile{ in_tile_dims, in.data_type(), 1 };

    // Simple tensor for the transformation matrix
    SimpleTensor<T> trans_matrix{ tr_tile_dims, in.data_type(), 1 };

    // Simple tensor for the transformation matrix transpose
    SimpleTensor<T> trans_matrix_transposed{ TensorShape(tr_tile_dims[1], tr_tile_dims[0]), in.data_type(), 1 };

    // Simple tensor for the temporary tile
    SimpleTensor<T> tmp_tile{ tr_tile_dims, in.data_type(), 1 };

    // Simple tensor for the output tile
    SimpleTensor<T> output_tile{ out_tile_dims, in.data_type(), 1 };

    // Initialize matrix for the output transform
    initialize_matrix_transform(trans_matrix, output_tile_size, kernel_size, WinogradTransformType::OUTPUT);

    // Transpose the transformation matrix
    transpose_matrix<T>(trans_matrix, trans_matrix_transposed);

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

    // Compute the number of output tiles along the x and y direction of size "output_tile_size"
    const Size2D num_tiles = compute_winograd_convolution_tiles(Size2D(input_dimensions.width, input_dimensions.height),
                                                                kernel_size,
                                                                output_tile_size,
                                                                conv_info);

    const int num_tiles_x = num_tiles.width;
    const int num_tiles_y = num_tiles.height;

    ARM_COMPUTE_UNUSED(num_tiles_y);
    ARM_COMPUTE_ERROR_ON(in.shape()[1] != static_cast<unsigned int>(num_tiles_x * num_tiles_y));

    // If we have a vertical filter (i.e. 1x3, 1x5,..), we still need to take the elements along the x direction (step_y_transf_tile = 1)
    const int step_y_transf_tile = kernel_size.width == 1 ? 1 : output_tile.shape()[0];

    // Initialize with zeros the input tile
    zeros<T>(input_tile, Coordinates(0, 0), input_tile.shape());

    for(int n = 0; n < num_batches; ++n)
    {
        for(int y = 0; y < h_in; ++y)
        {
            for(int x = 0; x < w_in; ++x)
            {
                // Load the input tile tile across the channels of the input tensor
                for(int z = 0; z < c_in; ++z)
                {
                    input_tile[z] = in[x + (y * stridey_in) + (z * stridez_in) + (n * stridew_in)];
                }

                // First transformation
                matrix_multiply<T>(trans_matrix, input_tile, tmp_tile);

                // Second transformation
                matrix_multiply<T>(tmp_tile, trans_matrix_transposed, output_tile);

                // Store the output tile
                const int xo = (y % num_tiles_x) * out_tile_w;
                const int yo = (y / num_tiles_x) * out_tile_h;
                const int zo = x;

                const int output_offset = xo + (yo * stridey_out) + (zo * stridez_out) + (n * stridew_out);

                for(int yi = 0; yi < static_cast<int>(out_tile_h); ++yi)
                {
                    for(int xi = 0; xi < static_cast<int>(out_tile_w); ++xi)
                    {
                        // Check out-of-bound writes
                        if((xo + xi < w_out) && (yo + yi < h_out))
                        {
                            out[output_offset + yi * stridey_out + xi] = output_tile[xi + yi * step_y_transf_tile];

                            // Add bias
                            out[output_offset + yi * stridey_out + xi] += b[zo];
                        }
                    }
                }
            }
        }
    }

    return out;
}

template SimpleTensor<float> winograd_filter_transform(const SimpleTensor<float> &in, const TensorShape &output_shape, const WinogradInfo &winograd_info);
template SimpleTensor<float> winograd_input_transform(const SimpleTensor<float> &in, const TensorShape &output_shape, const WinogradInfo &winograd_info);
template SimpleTensor<float> winograd_output_transform(const SimpleTensor<float> &in, const SimpleTensor<float> &b, const TensorShape &output_shape, const WinogradInfo &winograd_info);
template SimpleTensor<half> winograd_filter_transform(const SimpleTensor<half> &in, const TensorShape &output_shape, const WinogradInfo &winograd_info);
template SimpleTensor<half> winograd_input_transform(const SimpleTensor<half> &in, const TensorShape &output_shape, const WinogradInfo &winograd_info);
template SimpleTensor<half> winograd_output_transform(const SimpleTensor<half> &in, const SimpleTensor<half> &b, const TensorShape &output_shape, const WinogradInfo &winograd_info);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
