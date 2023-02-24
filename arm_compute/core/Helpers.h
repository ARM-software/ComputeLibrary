/*
 * Copyright (c) 2016-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_HELPERS_H
#define ARM_COMPUTE_HELPERS_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace arm_compute
{
class IKernel;
class ITensor;
class ITensorInfo;

/** Iterator updated by @ref execute_window_loop for each window element */
class Iterator
{
public:
    /** Default constructor to create an empty iterator */
    constexpr Iterator();
    /** Create a container iterator for the metadata and allocation contained in the ITensor
     *
     * @param[in] tensor The tensor to associate to the iterator.
     * @param[in] window The window which will be used to iterate over the tensor.
     */
    Iterator(const ITensor *tensor, const Window &window);

    /** Create a container iterator for the tensor with the specified number of dimensions, stride, buffer pointer and window.
     *
     * @param[in] num_dims The number of dimensions.
     * @param[in] strides  The strides in bytes.
     * @param[in] buffer   The data buffer.
     * @param[in] offset   The offset in bytes from the beginning of the buffer to the first element of the tensor.
     * @param[in] window   The window which will be used to iterate over the tensor.
     */
    Iterator(size_t num_dims, const Strides &strides, uint8_t *buffer, size_t offset, const Window &window);

    /** Increment the iterator along the specified dimension of the step value associated to the dimension.
     *
     * @warning It is the caller's responsibility to call increment(dimension+1) when reaching the end of a dimension, the iterator will not check for overflow.
     *
     * @note When incrementing a dimension 'n' the coordinates of all the dimensions in the range (0,n-1) are reset. For example if you iterate over a 2D image, everytime you change row (dimension 1), the iterator for the width (dimension 0) is reset to its start.
     *
     * @param[in] dimension Dimension to increment
     */
    void increment(size_t dimension);

    /** Return the offset in bytes from the first element to the current position of the iterator
     *
     * @return The current position of the iterator in bytes relative to the first element.
     */
    constexpr size_t offset() const;

    /** Return a pointer to the current pixel.
     *
     * @warning Only works if the iterator was created with an ITensor.
     *
     * @return equivalent to  buffer() + offset()
     */
    constexpr uint8_t *ptr() const;

    /** Move the iterator back to the beginning of the specified dimension.
     *
     * @param[in] dimension Dimension to reset
     */
    void reset(size_t dimension);

private:

    /** Initialize a container iterator for the tensor with the specified number of dimensions, stride, buffer pointer and window.
     *
     * @param[in] num_dims The number of dimensions.
     * @param[in] strides  The strides in bytes.
     * @param[in] buffer   The data buffer.
     * @param[in] offset   The offset in bytes from the beginning of the buffer to the first element of the tensor.
     * @param[in] window   The window which will be used to iterate over the tensor.
     */
    void initialize(size_t num_dims, const Strides &strides, uint8_t *buffer, size_t offset, const Window &window);

    uint8_t *_ptr;

    class Dimension
    {
    public:
        constexpr Dimension()
            : _dim_start(0), _stride(0)
        {
        }

        size_t _dim_start;
        size_t _stride;
    };

    std::array<Dimension, Coordinates::num_max_dimensions> _dims;
};

/** Iterate through the passed window, automatically adjusting the iterators and calling the lambda_functino for each element.
 *  It passes the x and y positions to the lambda_function for each iteration
 *
 * @param[in]     w               Window to iterate through.
 * @param[in]     lambda_function The function of type void(function)( const Coordinates & id ) to call at each iteration.
 *                                Where id represents the absolute coordinates of the item to process.
 * @param[in,out] iterators       Tensor iterators which will be updated by this function before calling lambda_function.
 */
template <typename L, typename... Ts>
inline void execute_window_loop(const Window &w, L &&lambda_function, Ts &&... iterators);

/** Permutes given Dimensions according to a permutation vector
 *
 * @warning Validity of permutation is not checked
 *
 * @param[in, out] dimensions Dimensions to permute
 * @param[in]      perm       Permutation vector
 */
template <typename T>
inline void permute(Dimensions<T> &dimensions, const PermutationVector &perm)
{
    auto dimensions_copy = utility::make_array<Dimensions<T>::num_max_dimensions>(dimensions.begin(), dimensions.end());
    for(unsigned int i = 0; i < perm.num_dimensions(); ++i)
    {
        T dimension_val = (perm[i] < dimensions.num_dimensions()) ? dimensions_copy[perm[i]] : 0;
        dimensions.set(i, dimension_val);
    }
}

/** Permutes given TensorShape according to a permutation vector
 *
 * @warning Validity of permutation is not checked
 *
 * @param[in, out] shape Shape to permute
 * @param[in]      perm  Permutation vector
 */
inline void permute(TensorShape &shape, const PermutationVector &perm)
{
    TensorShape shape_copy = shape;
    for(unsigned int i = 0; i < perm.num_dimensions(); ++i)
    {
        size_t dimension_val = (perm[i] < shape.num_dimensions()) ? shape_copy[perm[i]] : 1;
        shape.set(i, dimension_val, false, false); // Avoid changes in _num_dimension
    }
}

/** Helper function to calculate the Valid Region for Scale.
 *
 * @param[in] src_info           Input tensor info used to check.
 * @param[in] dst_shape          Shape of the output.
 * @param[in] interpolate_policy Interpolation policy.
 * @param[in] sampling_policy    Sampling policy.
 * @param[in] border_undefined   True if the border is undefined.
 *
 * @return The corresponding valid region
 */
ValidRegion calculate_valid_region_scale(const ITensorInfo &src_info, const TensorShape &dst_shape,
                                         InterpolationPolicy interpolate_policy, SamplingPolicy sampling_policy, bool border_undefined);

/** Convert a linear index into n-dimensional coordinates.
 *
 * @param[in] shape Shape of the n-dimensional tensor.
 * @param[in] index Linear index specifying the i-th element.
 *
 * @return n-dimensional coordinates.
 */
inline Coordinates index2coords(const TensorShape &shape, int index);

/** Convert n-dimensional coordinates into a linear index.
 *
 * @param[in] shape Shape of the n-dimensional tensor.
 * @param[in] coord N-dimensional coordinates.
 *
 * @return linead index
 */
inline int coords2index(const TensorShape &shape, const Coordinates &coord);

/** Returns a static map used to find an index or dimension based on a data layout
  *
  * *** Layouts ***
  *
  * *** 4D ***
  * [N C H W]
  * [3 2 1 0]
  * [N H W C]
  *
  * * *** 5D ***
  * [N C D H W]
  * [4 3 2 1 0]
  * [N D H W C]
  */
const std::map<DataLayout, std::vector<DataLayoutDimension>> &get_layout_map();

/** Get the index of the given dimension.
 *
 * @param[in] data_layout           The data layout.
 * @param[in] data_layout_dimension The dimension which this index is requested for.
 *
 * @return The int conversion of the requested data layout index.
 */
inline size_t get_data_layout_dimension_index(const DataLayout &data_layout, const DataLayoutDimension &data_layout_dimension);

/** Get the DataLayoutDimension of a given index and layout.
 *
 * @param[in] data_layout The data layout.
 * @param[in] index       The data layout index.
 *
 * @return The dimension which this index is requested for.
 */
inline DataLayoutDimension get_index_data_layout_dimension(const DataLayout &data_layout, const size_t index);

/** Calculate the number of output tiles required by Winograd Convolution layer. This utility function can be used by the Winograd input transform
 *  to know the number of tiles on the x and y direction
 *
 * @param[in] in_dims          Spatial dimensions of the input tensor of convolution layer
 * @param[in] kernel_size      Kernel size
 * @param[in] output_tile_size Size of a single output tile
 * @param[in] conv_info        Convolution info (i.e. pad, stride,...)
 *
 * @return the number of output tiles along the x and y directions of size "output_tile_size"
 */
inline Size2D compute_winograd_convolution_tiles(const Size2D &in_dims, const Size2D &kernel_size, const Size2D &output_tile_size, const PadStrideInfo &conv_info)
{
    int num_tiles_x = std::ceil((in_dims.width - (kernel_size.width - 1) + conv_info.pad_left() + conv_info.pad_right()) / static_cast<float>(output_tile_size.width));
    int num_tiles_y = std::ceil((in_dims.height - (kernel_size.height - 1) + conv_info.pad_top() + conv_info.pad_bottom()) / static_cast<float>(output_tile_size.height));

    // Clamp in case we provide paddings but we have 1D convolution
    num_tiles_x = std::min(num_tiles_x, static_cast<int>(in_dims.width));
    num_tiles_y = std::min(num_tiles_y, static_cast<int>(in_dims.height));

    return Size2D(num_tiles_x, num_tiles_y);
}

/** Wrap-around a number within the range 0 <= x < m
 *
 * @param[in] x Input value
 * @param[in] m Range
 *
 * @return the wrapped-around number
 */
template <typename T>
inline T wrap_around(T x, T m)
{
    return x >= 0 ? x % m : (x % m + m) % m;
}

/** Convert negative coordinates to positive in the range [0, num_dims_input]
 *
 * @param[out] coords    Array of coordinates to be converted.
 * @param[in]  max_value Maximum value to be used when wrapping the negative values in coords
 */
inline Coordinates &convert_negative_axis(Coordinates &coords, int max_value)
{
    for(unsigned int i = 0; i < coords.num_dimensions(); ++i)
    {
        coords[i] = wrap_around(coords[i], max_value);
    }
    return coords;
}
} // namespace arm_compute

#include "arm_compute/core/Helpers.inl"
#endif /*ARM_COMPUTE_HELPERS_H */
