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
#include "GEMMReshapeRHSMatrix.h"

#include "arm_compute/core/Types.h"

#include "tests/validation/Helpers.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> gemm_reshape_rhs_matrix(const SimpleTensor<T> &in, const TensorShape &output_shape, const GEMMRHSMatrixInfo &rhs_info)
{
    ARM_COMPUTE_ERROR_ON(in.shape().num_dimensions() > 3);

    SimpleTensor<T> out{ output_shape, in.data_type() };

    // Initialize the output tensor with zero
    std::memset(&out[0], 0, out.num_elements() * sizeof(T));

    const unsigned int N = in.shape()[0];
    const unsigned int K = in.shape()[1];
    const unsigned int B = in.shape()[2];

    const unsigned int num_tiles_x = std::ceil(N / static_cast<float>(rhs_info.n0));
    const unsigned int num_tiles_y = std::ceil(K / static_cast<float>(rhs_info.k0));

    const TensorShape tile_dims(rhs_info.n0, rhs_info.k0);
    const TensorShape tile_dims_transposed(rhs_info.k0, rhs_info.n0);

    // Simple tensor for the input tile
    SimpleTensor<T> src_tile{ tile_dims, in.data_type() };

    // Simple tensor for the input tile
    SimpleTensor<T> src_tile_transposed{ tile_dims_transposed, in.data_type() };

    // Simple tensor to use when storing the values
    SimpleTensor<T> *tile_to_use = rhs_info.transpose ? &src_tile_transposed : &src_tile;

    const unsigned int offset_output_x = rhs_info.interleave ? tile_to_use->shape()[0] : tile_to_use->shape()[0] * tile_to_use->shape()[1];
    const unsigned int step_output_x   = rhs_info.interleave ? tile_to_use->shape()[0] * rhs_info.h0 : tile_to_use->shape()[0];

    for(unsigned int z = 0; z < B; ++z)
    {
        for(unsigned int y = 0; y < num_tiles_y; ++y)
        {
            for(unsigned int x = 0; x < num_tiles_x; ++x)
            {
                // Get the tile from the input tensor
                get_tile<T>(in, src_tile, Coordinates(x * rhs_info.n0, y * rhs_info.k0, z, 0));

                if(rhs_info.transpose)
                {
                    // Transpose matrix
                    transpose_matrix<T>(src_tile, src_tile_transposed);
                }

                // Store
                const unsigned int offset_output = (y * rhs_info.k0 * rhs_info.n0 * rhs_info.h0) + ((x % rhs_info.h0) * offset_output_x) + ((x / rhs_info.h0) * out.shape()[0]) + (z * out.shape()[0] * out.shape()[1]);

                for(unsigned int i = 0; i < tile_to_use->shape()[1]; ++i)
                {
                    const unsigned int offset_tile = i * tile_to_use->shape()[0];

                    // Copy per row
                    std::copy(&(*tile_to_use)[offset_tile], &(*tile_to_use)[offset_tile + tile_to_use->shape()[0]], &out[offset_output + i * step_output_x]);
                }
            }
        }
    }

    return out;
}
template SimpleTensor<int> gemm_reshape_rhs_matrix(const SimpleTensor<int> &in, const TensorShape &output_shape, const GEMMRHSMatrixInfo &rhs_info);
template SimpleTensor<short> gemm_reshape_rhs_matrix(const SimpleTensor<short> &in, const TensorShape &output_shape, const GEMMRHSMatrixInfo &rhs_info);
template SimpleTensor<char> gemm_reshape_rhs_matrix(const SimpleTensor<char> &in, const TensorShape &output_shape, const GEMMRHSMatrixInfo &rhs_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute