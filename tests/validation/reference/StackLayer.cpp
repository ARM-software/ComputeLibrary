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
#include "StackLayer.h"

#include "arm_compute/core/Types.h"

#include "tests/validation/Helpers.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> stack_layer(const std::vector<SimpleTensor<T>> &in, const TensorShape &output_shape, DataType data_type, unsigned int axis)
{
    ARM_COMPUTE_ERROR_ON(output_shape.num_dimensions() > 5);
    ARM_COMPUTE_ERROR_ON(in.size() < 2);
    ARM_COMPUTE_ERROR_ON(axis > in[0].shape().num_dimensions());

    SimpleTensor<T> out{ output_shape, data_type };

    const int width       = in[0].shape()[0];
    const int height      = in[0].shape()[1];
    const int depth       = in[0].shape()[2];
    const int batch_size  = in[0].shape()[3];
    const int num_tensors = in.size();

    // Array to store the input coordinates
    // i_coordinates[0] = xi, i_coordinates[1] = yi, i_coordinates[2] = zi
    // i_coordinates[3] = bi, i_coordinates[4] = i, i_coordinates[5] = 0
    // i_coordinates[5] will be always zero and used for not incrementing the output when the input has less than 4 dimensions
    std::array<int, 6> i_coordinates{ 0 };

    // Array of pointers used to map the output coordinates to the input ones accordingly with the axis
    // This array is initialized with &i_coordinates[5] since this will be always zero
    std::array<int *, 5> o_coordinates = { &i_coordinates[5], &i_coordinates[5], &i_coordinates[5], &i_coordinates[5], &i_coordinates[5] };

    // Set the axis coordinate
    o_coordinates[axis] = &i_coordinates[4];

    unsigned int k_shift = 0;

    // Map the output coordinates
    for(unsigned int k = 0; k < in[0].shape().num_dimensions(); ++k)
    {
        if(k == axis)
        {
            k_shift++;
        }

        o_coordinates[k + k_shift] = &i_coordinates[k];
    }

    // Use alias for the input coordinates
    int &xi = i_coordinates[0];
    int &yi = i_coordinates[1];
    int &zi = i_coordinates[2];
    int &bi = i_coordinates[3];
    int &i  = i_coordinates[4];

    // Use alias for the output coordinates
    int &xo = *(o_coordinates[0]);
    int &yo = *(o_coordinates[1]);
    int &zo = *(o_coordinates[2]);
    int &bo = *(o_coordinates[3]);
    int &wo = *(o_coordinates[4]);

    // Stack tensors
    for(; i < num_tensors; ++(i))
    {
        bi = 0;
        for(; bi < batch_size; ++(bi))
        {
            zi = 0;
            for(; zi < depth; ++(zi))
            {
                yi = 0;
                for(; yi < height; ++(yi))
                {
                    xi = 0;
                    for(; xi < width; ++(xi))
                    {
                        *(reinterpret_cast<T *>(out(Coordinates(xo, yo, zo, bo, wo)))) = *(reinterpret_cast<const T *>(in[i](Coordinates(xi, yi, zi, bi))));
                    }
                }
            }
        }
    }

    return out;
}
template SimpleTensor<int> stack_layer(const std::vector<SimpleTensor<int>> &in, const TensorShape &output_shape, DataType data_type, unsigned int axis);
template SimpleTensor<short> stack_layer(const std::vector<SimpleTensor<short>> &in, const TensorShape &output_shape, DataType data_type, unsigned int axis);
template SimpleTensor<char> stack_layer(const std::vector<SimpleTensor<char>> &in, const TensorShape &output_shape, DataType data_type, unsigned int axis);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
