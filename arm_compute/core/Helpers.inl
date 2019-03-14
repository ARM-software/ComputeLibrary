/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"

#include <cmath>
#include <numeric>

namespace arm_compute
{
inline uint8_t pixel_area_c1u8_clamp(const uint8_t *first_pixel_ptr, size_t stride, size_t width, size_t height, float wr, float hr, int x, int y)
{
    ARM_COMPUTE_ERROR_ON(first_pixel_ptr == nullptr);

    // Calculate sampling position
    float in_x = (x + 0.5f) * wr - 0.5f;
    float in_y = (y + 0.5f) * hr - 0.5f;

    // Get bounding box offsets
    int x_from = std::floor(x * wr - 0.5f - in_x);
    int y_from = std::floor(y * hr - 0.5f - in_y);
    int x_to   = std::ceil((x + 1) * wr - 0.5f - in_x);
    int y_to   = std::ceil((y + 1) * hr - 0.5f - in_y);

    // Clamp position to borders
    in_x = std::max(-1.f, std::min(in_x, static_cast<float>(width)));
    in_y = std::max(-1.f, std::min(in_y, static_cast<float>(height)));

    // Clamp bounding box offsets to borders
    x_from = ((in_x + x_from) < -1) ? -1 : x_from;
    y_from = ((in_y + y_from) < -1) ? -1 : y_from;
    x_to   = ((in_x + x_to) > width) ? (width - in_x) : x_to;
    y_to   = ((in_y + y_to) > height) ? (height - in_y) : y_to;

    // Get pixel index
    const int xi = std::floor(in_x);
    const int yi = std::floor(in_y);

    // Bounding box elements in each dimension
    const int x_elements = (x_to - x_from + 1);
    const int y_elements = (y_to - y_from + 1);
    ARM_COMPUTE_ERROR_ON(x_elements == 0 || y_elements == 0);

    // Sum pixels in area
    int sum = 0;
    for(int j = yi + y_from, je = yi + y_to; j <= je; ++j)
    {
        const uint8_t *ptr = first_pixel_ptr + j * stride + xi + x_from;
        sum                = std::accumulate(ptr, ptr + x_elements, sum);
    }

    // Return average
    return sum / (x_elements * y_elements);
}

template <size_t dimension>
struct IncrementIterators
{
    template <typename T, typename... Ts>
    static void unroll(T &&it, Ts &&... iterators)
    {
        auto increment = [](T && it)
        {
            it.increment(dimension);
        };
        utility::for_each(increment, std::forward<T>(it), std::forward<Ts>(iterators)...);
    }
    static void unroll()
    {
        // End of recursion
    }
};

template <size_t dim>
struct ForEachDimension
{
    template <typename L, typename... Ts>
    static void unroll(const Window &w, Coordinates &id, L &&lambda_function, Ts &&... iterators)
    {
        const auto &d = w[dim - 1];

        for(auto v = d.start(); v < d.end(); v += d.step(), IncrementIterators < dim - 1 >::unroll(iterators...))
        {
            id.set(dim - 1, v);
            ForEachDimension < dim - 1 >::unroll(w, id, lambda_function, iterators...);
        }
    }
};

template <>
struct ForEachDimension<0>
{
    template <typename L, typename... Ts>
    static void unroll(const Window &w, Coordinates &id, L &&lambda_function, Ts &&... iterators)
    {
        lambda_function(id);
    }
};

template <typename L, typename... Ts>
inline void execute_window_loop(const Window &w, L &&lambda_function, Ts &&... iterators)
{
    w.validate();

    for(unsigned int i = 0; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(w[i].step() == 0);
    }

    Coordinates id;
    ForEachDimension<Coordinates::num_max_dimensions>::unroll(w, id, std::forward<L>(lambda_function), std::forward<Ts>(iterators)...);
}

inline constexpr Iterator::Iterator()
    : _ptr(nullptr), _dims()
{
}

inline Iterator::Iterator(const ITensor *tensor, const Window &win)
    : Iterator()
{
    ARM_COMPUTE_ERROR_ON(tensor == nullptr);
    ARM_COMPUTE_ERROR_ON(tensor->info() == nullptr);

    const ITensorInfo *info    = tensor->info();
    const Strides     &strides = info->strides_in_bytes();

    _ptr = tensor->buffer() + info->offset_first_element_in_bytes();

    //Initialize the stride for each dimension and calculate the position of the first element of the iteration:
    for(unsigned int n = 0; n < info->num_dimensions(); ++n)
    {
        _dims[n]._stride = win[n].step() * strides[n];
        std::get<0>(_dims)._dim_start += strides[n] * win[n].start();
    }

    //Copy the starting point to all the dimensions:
    for(unsigned int n = 1; n < Coordinates::num_max_dimensions; ++n)
    {
        _dims[n]._dim_start = std::get<0>(_dims)._dim_start;
    }

    ARM_COMPUTE_ERROR_ON_WINDOW_DIMENSIONS_GTE(win, info->num_dimensions());
}

inline void Iterator::increment(const size_t dimension)
{
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions);

    _dims[dimension]._dim_start += _dims[dimension]._stride;

    for(unsigned int n = 0; n < dimension; ++n)
    {
        _dims[n]._dim_start = _dims[dimension]._dim_start;
    }
}

inline constexpr int Iterator::offset() const
{
    return _dims.at(0)._dim_start;
}

inline constexpr uint8_t *Iterator::ptr() const
{
    return _ptr + _dims.at(0)._dim_start;
}

inline void Iterator::reset(const size_t dimension)
{
    ARM_COMPUTE_ERROR_ON(dimension >= Coordinates::num_max_dimensions - 1);

    _dims[dimension]._dim_start = _dims[dimension + 1]._dim_start;

    for(unsigned int n = 0; n < dimension; ++n)
    {
        _dims[n]._dim_start = _dims[dimension]._dim_start;
    }
}

inline bool auto_init_if_empty(ITensorInfo       &info,
                               const TensorShape &shape,
                               int                num_channels,
                               DataType           data_type,
                               QuantizationInfo   quantization_info)
{
    if(info.tensor_shape().total_size() == 0)
    {
        info.set_data_type(data_type);
        info.set_num_channels(num_channels);
        info.set_tensor_shape(shape);
        info.set_quantization_info(quantization_info);
        return true;
    }

    return false;
}

inline bool auto_init_if_empty(ITensorInfo &info_sink, const ITensorInfo &info_source)
{
    if(info_sink.tensor_shape().total_size() == 0)
    {
        info_sink.set_data_type(info_source.data_type());
        info_sink.set_num_channels(info_source.num_channels());
        info_sink.set_tensor_shape(info_source.tensor_shape());
        info_sink.set_quantization_info(info_source.quantization_info());
        info_sink.set_data_layout(info_source.data_layout());
        return true;
    }

    return false;
}

inline bool set_shape_if_empty(ITensorInfo &info, const TensorShape &shape)
{
    if(info.tensor_shape().total_size() == 0)
    {
        info.set_tensor_shape(shape);
        return true;
    }

    return false;
}

inline bool set_format_if_unknown(ITensorInfo &info, Format format)
{
    if(info.data_type() == DataType::UNKNOWN)
    {
        info.set_format(format);
        return true;
    }

    return false;
}

inline bool set_data_type_if_unknown(ITensorInfo &info, DataType data_type)
{
    if(info.data_type() == DataType::UNKNOWN)
    {
        info.set_data_type(data_type);
        return true;
    }

    return false;
}

inline bool set_data_layout_if_unknown(ITensorInfo &info, DataLayout data_layout)
{
    if(info.data_layout() == DataLayout::UNKNOWN)
    {
        info.set_data_layout(data_layout);
        return true;
    }

    return false;
}

inline bool set_quantization_info_if_empty(ITensorInfo &info, QuantizationInfo quantization_info)
{
    if(info.quantization_info().empty() && (is_data_type_quantized_asymmetric(info.data_type())))
    {
        info.set_quantization_info(quantization_info);
        return true;
    }

    return false;
}

inline Coordinates index2coords(const TensorShape &shape, int index)
{
    int num_elements = shape.total_size();

    ARM_COMPUTE_ERROR_ON_MSG(index < 0 || index >= num_elements, "Index has to be in [0, num_elements]!");
    ARM_COMPUTE_ERROR_ON_MSG(num_elements == 0, "Cannot create coordinate from empty shape!");

    Coordinates coord{ 0 };

    for(int d = shape.num_dimensions() - 1; d >= 0; --d)
    {
        num_elements /= shape[d];
        coord.set(d, index / num_elements);
        index %= num_elements;
    }

    return coord;
}

inline int coords2index(const TensorShape &shape, const Coordinates &coord)
{
    int num_elements = shape.total_size();
    ARM_COMPUTE_UNUSED(num_elements);
    ARM_COMPUTE_ERROR_ON_MSG(num_elements == 0, "Cannot create linear index from empty shape!");

    int index  = 0;
    int stride = 1;

    for(unsigned int d = 0; d < coord.num_dimensions(); ++d)
    {
        index += coord[d] * stride;
        stride *= shape[d];
    }

    return index;
}

inline size_t get_data_layout_dimension_index(const DataLayout data_layout, const DataLayoutDimension data_layout_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(data_layout == DataLayout::UNKNOWN, "Cannot retrieve the dimension index for an unknown layout!");

    /* Return the index based on the data layout
     * [N C H W]
     * [3 2 1 0]
     * [N H W C]
    */
    switch(data_layout_dimension)
    {
        case DataLayoutDimension::CHANNEL:
            return (data_layout == DataLayout::NCHW) ? 2 : 0;
            break;
        case DataLayoutDimension::HEIGHT:
            return (data_layout == DataLayout::NCHW) ? 1 : 2;
            break;
        case DataLayoutDimension::WIDTH:
            return (data_layout == DataLayout::NCHW) ? 0 : 1;
            break;
        case DataLayoutDimension::BATCHES:
            return 3;
            break;
        default:
            ARM_COMPUTE_ERROR("Data layout index not supported!");
            break;
    }
}

inline DataLayoutDimension get_index_data_layout_dimension(const DataLayout data_layout, const size_t index)
{
    ARM_COMPUTE_ERROR_ON_MSG(data_layout == DataLayout::UNKNOWN, "Cannot retrieve the dimension index for an unknown layout!");

    /* Return the index based on the data layout
    * [N C H W]
    * [3 2 1 0]
    * [N H W C]
    */
    switch(index)
    {
        case 0:
            return (data_layout == DataLayout::NCHW) ? DataLayoutDimension::WIDTH : DataLayoutDimension::CHANNEL;
            break;
        case 1:
            return (data_layout == DataLayout::NCHW) ? DataLayoutDimension::HEIGHT : DataLayoutDimension::WIDTH;
            break;
        case 2:
            return (data_layout == DataLayout::NCHW) ? DataLayoutDimension::CHANNEL : DataLayoutDimension::HEIGHT;
            break;
        case 3:
            return DataLayoutDimension::BATCHES;
            break;
        default:
            ARM_COMPUTE_ERROR("Index value not supported!");
            break;
    }
}
} // namespace arm_compute
