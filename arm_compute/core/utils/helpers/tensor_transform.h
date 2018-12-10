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
#ifndef __ARM_COMPUTE_UTILS_HELPERS_TENSOR_TRANSFORM_H__
#define __ARM_COMPUTE_UTILS_HELPERS_TENSOR_TRANSFORM_H__

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace helpers
{
namespace tensor_transform
{
/** Computes stride of a given index
 *
 * @param[in] index   Index of tensor to calculate absolute start position
 * @param[in] strides Slice strides
 *
 * @return Stride at a given index
 */
int calculate_stride_on_index(int index, Coordinates strides);

/** Computes absolute start position of a given index for a strided slice operation
 *
 * @param[in] input_shape Input tensor shape
 * @param[in] index       Index of tensor to calculate absolute start position
 * @param[in] starts      Start coordinates
 * @param[in] strides     Slice strides
 * @param[in] begin_mask  (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and
 *                        the fullest possible range in that dimension is used instead.
 *
 * @return Absolute start position of a given index
 */
int calculate_start_on_index(TensorShape input_shape, int index, Coordinates starts, Coordinates strides, int32_t begin_mask);

/** Returns the absolute end position of a given index for a strided slice operation
 *
 * @param[in] input_shape      Input tensor shape
 * @param[in] index            Index of tensor to calculate absolute start position
 * @param[in] start_on_index   Absolute start coordinate for given index
 * @param[in] ends             End coordinates
 * @param[in] strides          Slice strides
 * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, end[i] is ignored and
 *                             the fullest possible range in that dimension is used instead.
 * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
 *                             A slice of size 1 starting from starts[i] in the dimension must be preserved.
 *
 * @return Absolute end position of a given index
 */
int calculate_end_on_index(TensorShape input_shape, int index, int start_on_index, Coordinates ends, Coordinates strides,
                           int32_t end_mask = 0, int32_t shrink_axis_mask = 0);

/** Calculate start, end and stride coordinates for a strided slice
 *
 * @param[in] input_shape      Input tensor shape
 * @param[in] starts           Start coordinates
 * @param[in] ends             End coordinates
 * @param[in] strides          Slice strides
 * @param[in] begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and
 *                             the fullest possible range in that dimension is used instead.
 * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, end[i] is ignored and
 *                             the fullest possible range in that dimension is used instead.
 * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
 *                             A slice of size 1 starting from starts[i] in the dimension must be preserved.
 *
 * @return A tuple with <Start,End,Strides>
 */
std::tuple<Coordinates, Coordinates, Coordinates> calculate_strided_slice_coords(TensorShape input_shape,
                                                                                 Coordinates starts, Coordinates ends, Coordinates strides,
                                                                                 int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_axis_mask = 0);

/** Computes output shape of strided slice
 *
 * @warning Starts and ends must be non-negative
 * @warning Starts, ends and final strides should have the same dimensions as the input shape
 *
 * @param[in] input_shape       Input tensor shape
 * @param[in] starts            Absolute start coordinates
 * @param[in] ends              Absolute end coordinates
 * @param[in] strides           Slice strides
 * @param[in] begin_mask        (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and
 *                              the fullest possible range in that dimension is used instead.
 * @param[in] end_mask          (Optional) If the ith bit of end_mask is set, end[i] is ignored and
 *                              the fullest possible range in that dimension is used instead.
 * @param[in] shrink_axis_mask  (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
 *                              A slice of size 1 starting from starts[i] in the dimension must be preserved.
 * @param[in] return_unshrinked (Optional) Returns un-shrinked shape
 *
 * @return The output tensor shape
 */
TensorShape compute_strided_slice_output_shape(TensorShape input_shape, Coordinates starts, Coordinates ends, Coordinates strides,
                                               int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_axis_mask = 0,
                                               bool return_unshrinked = false);

/** Constructs end mask in case we want to perform a slice operation using the strided slice interface
 *
 * @note Ends are inclusive in slice operations that is why construction an end mask is needed
 *
 * @param[in] ends End coordinates
 *
 * @return End mask
 */
int32_t construct_slice_end_mask(Coordinates ends);
} // namespace tensor_tranform
} // namespace helpers
} // namespace arm_compute
#endif /* __ARM_COMPUTE_UTILS_HELPERS_TENSOR_TRANSFORM_H__ */
