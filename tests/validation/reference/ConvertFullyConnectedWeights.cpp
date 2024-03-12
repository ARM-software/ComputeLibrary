/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "ConvertFullyConnectedWeights.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> convert_fully_connected_weights(const SimpleTensor<T> &src, const TensorShape &original_input_shape, const DataLayout training_data_layout)
{
    SimpleTensor<T> dst(src.shape(), src.data_type());

    const DataLayout original_input_data_layout = (training_data_layout == DataLayout::NCHW) ? DataLayout::NHWC : DataLayout::NCHW;

    const int width_idx   = get_data_layout_dimension_index(original_input_data_layout, DataLayoutDimension::WIDTH);
    const int height_idx  = get_data_layout_dimension_index(original_input_data_layout, DataLayoutDimension::HEIGHT);
    const int channel_idx = get_data_layout_dimension_index(original_input_data_layout, DataLayoutDimension::CHANNEL);

    const bool         is_nchw_to_nhwc           = training_data_layout == DataLayout::NCHW;
    const unsigned int num_elems_per_input_plane = original_input_shape[width_idx] * original_input_shape[height_idx];
    const unsigned int num_channels              = original_input_shape[channel_idx];
    const unsigned int factor_1                  = is_nchw_to_nhwc ? num_elems_per_input_plane : num_channels;
    const unsigned int factor_2                  = is_nchw_to_nhwc ? num_channels : num_elems_per_input_plane;

    const uint32_t num_elements = src.num_elements();
#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(uint32_t i = 0; i < num_elements; ++i)
    {
        const Coordinates coords_in = index2coords(src.shape(), i);
        const Coordinates coords_out(coords_in.x(), coords_in.y() % factor_1 * factor_2 + coords_in.y() / factor_1);

        dst[coords2index(dst.shape(), coords_out)] = src[i];
    }

    return dst;
}

template SimpleTensor<uint8_t> convert_fully_connected_weights(const SimpleTensor<uint8_t> &src, const TensorShape &original_input_shape,
                                                               const DataLayout training_data_layout);
template SimpleTensor<half> convert_fully_connected_weights(const SimpleTensor<half> &src, const TensorShape &original_input_shape,
                                                            const DataLayout training_data_layout);
template SimpleTensor<float> convert_fully_connected_weights(const SimpleTensor<float> &src, const TensorShape &original_input_shape,
                                                             const DataLayout training_data_layout);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
