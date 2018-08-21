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
/** Returns the absolute start coordinates of strided slice
 *
 * @param[in] input_shape Input tensor shape
 * @param[in] starts      Start coordinates
 * @param[in] strides     Slice strides
 * @param[in] begin_mask  (Optional) If the ith bit of begin_mask is set, begin[i] is ignored and
 *                        the fullest possible range in that dimension is used instead.
 *
 * @return Absolute start coordinates
 */
Coordinates strided_slice_absolute_start_coords(TensorShape input_shape, Coordinates starts, Coordinates strides, int32_t begin_mask = 0);

/** Returns the absolute ends coordinates of strided slice
 *
 * @param[in] input_shape      Input tensor shape
 * @param[in] starts_abs       Absolute start coordinates
 * @param[in] ends             End coordinates
 * @param[in] strides          Slice strides
 * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, end[i] is ignored and
 *                             the fullest possible range in that dimension is used instead.
 * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
 *                             A slice of size 1 starting from begin[i] in the dimension must be preserved.
 *
 * @return Absolute end coordinates
 */
Coordinates strided_slice_absolute_end_coords(TensorShape input_shape, Coordinates starts_abs, Coordinates ends, Coordinates strides,
                                              int32_t end_mask = 0, int32_t shrink_axis_mask = 0);
/** Returns the final strides of strided slice
 *
 * @param[in] input_shape Input tensor shape
 * @param[in] strides     Slice strides
 *
 * @return The final strides need by strided slice
 */
Coordinates strided_slice_strides(TensorShape input_shape, Coordinates strides);

/** Computes output shape of a strided slice
 *
 * @param[in] input_shape   Input tensor shape
 * @param[in] starts_abs    Absolute start coordinates
 * @param[in] ends_abs      Absolute end coordinates
 * @param[in] final_strides Slice strides
 *
 * @return The output tensor shape
 */
TensorShape compute_strided_slice_output_shape(TensorShape input_shape, Coordinates starts_abs, Coordinates ends_abs, Coordinates final_strides);
} // namespace tensor_tranform
} // namespace helpers
} // namespace arm_compute
#endif /* __ARM_COMPUTE_UTILS_HELPERS_TENSOR_TRANSFORM_H__ */
