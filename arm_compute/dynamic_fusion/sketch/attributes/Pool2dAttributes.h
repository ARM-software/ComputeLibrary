/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_POOL2DATTRIBUTES
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_POOL2DATTRIBUTES

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Attributes are backend-agnostic parameters (in addition to the input/output tensors) of an operator.
 */

class Pool2dAttributes
{
public:
    /* Get Pooling Type */
    PoolingType pool_type() const;

    /* Set Pooling Type */
    Pool2dAttributes pool_type(PoolingType pool_type);

    /* Get 2D Pool Size */
    Size2D pool_size() const;

    /* Set 2D Pool size */
    Pool2dAttributes pool_size(const Size2D &pool_size);

    /* Get Padding */
    Padding2D pad() const;

    /* Set Padding */
    Pool2dAttributes pad(const Padding2D &padding);

    /* Get Stride */
    Size2D stride() const;

    /* Set Stride */
    Pool2dAttributes stride(const Size2D &stride);

    /* Get exclude padding */
    bool exclude_padding() const;

    /* Set exclude padding */
    Pool2dAttributes exclude_padding(bool exclude_padding);

private:
    PoolingType _pool_type{};
    Padding2D   _pad{};
    Size2D      _pool_size{};
    Size2D      _stride{ 1U, 1U };
    bool        _exclude_padding{ true };
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_POOL2DATTRIBUTES */
