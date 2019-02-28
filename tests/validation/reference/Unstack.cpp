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
#include "Unstack.h"

#include "tests/validation/Helpers.h"

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
inline Coordinates expand_coordinates(Coordinates in_coord, size_t axis, size_t slice, size_t num_dimensions)
{
    /*
        Reconstruct input_coord to read the corresponding value from the correct slice. This is done by adding an extra dimension
        to the coordinates and shuffling around the values based on the info below.

        For example, if input tensor shape is (X, Y, Z, W);

        If axis == 0, each slice will have the shape (Y, Z, W) and there will be X slices

        If axis == 1, each slice will have the shape (X, Z, W) and there will be Y slices.
    */
    Coordinates expanded_coord;
    expanded_coord.set_num_dimensions(num_dimensions);
    expanded_coord.set(axis, slice);
    for(size_t k = 0; k < axis; ++k)
    {
        expanded_coord.set(k, in_coord[k]);
    }
    for(size_t k = axis + 1; k < num_dimensions; ++k)
    {
        expanded_coord.set(k, in_coord[k - 1]);
    }
    return expanded_coord;
}

template <typename T>
SimpleTensor<T> get_slice(const SimpleTensor<T> &input_tensor, size_t axis, size_t slice)
{
    TensorShape out_shape = input_tensor.shape();
    out_shape.remove_dimension(axis);

    const size_t unpacked_num_dimensions(input_tensor.shape().num_dimensions());

    SimpleTensor<T> output{ out_shape, input_tensor.data_type() };

    Window win;
    win.use_tensor_dimensions(out_shape);
    execute_window_loop(win, [&](const Coordinates & id)
    {
        const Coordinates input_coords     = expand_coordinates(id, axis, slice, unpacked_num_dimensions);
        *reinterpret_cast<T *>(output(id)) = *reinterpret_cast<const T *>(input_tensor(input_coords));
    });

    return output;
}
} // namespace

template <typename T>
std::vector<SimpleTensor<T>> unstack(const SimpleTensor<T> &input_tensor, std::vector<SimpleTensor<T>> &output_tensors, int axis)
{
    // Wrap around negative values
    const unsigned int axis_u = wrap_around(axis, static_cast<int>(input_tensor.shape().num_dimensions()));
    ARM_COMPUTE_ERROR_ON(axis_u >= input_tensor.shape().num_dimensions());
    for(size_t k = 0; k < output_tensors.size(); ++k)
    {
        SimpleTensor<T>      &output    = output_tensors[k];
        const SimpleTensor<T> kth_slice = get_slice(input_tensor, axis_u, k);
        output                          = copy_tensor<T>(kth_slice);
    }
    return output_tensors;
}

template std::vector<SimpleTensor<float>> unstack(const SimpleTensor<float> &input_tensor, std::vector<SimpleTensor<float>> &output_tensors, int axis);
template std::vector<SimpleTensor<half>> unstack(const SimpleTensor<half> &input_tensor, std::vector<SimpleTensor<half>> &output_tensors, int axis);
template std::vector<SimpleTensor<uint8_t>> unstack(const SimpleTensor<uint8_t> &input_tensor, std::vector<SimpleTensor<uint8_t>> &output_tensors, int axis);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
