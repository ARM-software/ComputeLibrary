/*
* Copyright (c) 2020-2021 Arm Limited.
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
#ifndef SRC_CORE_HELPERS_UTILS_H
#define SRC_CORE_HELPERS_UTILS_H

#include "arm_compute/core/ITensorInfo.h"

namespace arm_compute
{
/** Create a strides object based on the provided strides and the tensor dimensions.
 *
 * @param[in] info          Tensor info object providing the shape of the tensor for unspecified strides.
 * @param[in] stride_x      Stride to be used in X dimension (in bytes).
 * @param[in] fixed_strides Strides to be used in higher dimensions starting at Y (in bytes).
 *
 * @return Strides object based on the specified strides. Missing strides are
 *         calculated based on the tensor shape and the strides of lower dimensions.
 */
template <typename T, typename... Ts>
inline Strides compute_strides(const ITensorInfo &info, T stride_x, Ts &&... fixed_strides)
{
    const TensorShape &shape = info.tensor_shape();

    // Create strides object
    Strides strides(stride_x, fixed_strides...);

    for(size_t i = 1 + sizeof...(Ts); i < info.num_dimensions(); ++i)
    {
        strides.set(i, shape[i - 1] * strides[i - 1]);
    }

    return strides;
}

/** Create a strides object based on the tensor dimensions.
 *
 * @param[in] info Tensor info object used to compute the strides.
 *
 * @return Strides object based on element size and tensor shape.
 */
template <typename... Ts>
inline Strides compute_strides(const ITensorInfo &info)
{
    return compute_strides(info, info.element_size());
}

/** Given an integer value, this function returns the next power of two
 *
 * @param[in] x Input value
 *
 * @return the next power of two
 */
inline unsigned int get_next_power_two(unsigned int x)
{
    // Decrement by 1
    x--;

    // Shift right by 1
    x |= x >> 1u;
    // Shift right by 2
    x |= x >> 2u;
    // Shift right by 4
    x |= x >> 4u;
    // Shift right by 8
    x |= x >> 8u;
    // Shift right by 16
    x |= x >> 16u;

    // Increment by 1
    x++;

    return x;
}
} // namespace arm_compute

#endif /* SRC_CORE_HELPERS_UTILS_H */
