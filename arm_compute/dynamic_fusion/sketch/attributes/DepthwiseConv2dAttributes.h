/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_DEPTHWISECONV2DATTRIBUTES
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_DEPTHWISECONV2DATTRIBUTES

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Types.h"
#include <cstdint>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Attributes are backend-agnostic parameters (in addition to the input/output tensors) of an operator.
 */

/** Depthwise Conv2d attributes */
class DepthwiseConv2dAttributes
{
public:
    /** Set padding */
    DepthwiseConv2dAttributes &pad(const Padding2D &pad);
    /** Get padding */
    Padding2D pad() const;
    /** Set stride */
    DepthwiseConv2dAttributes &stride(const Size2D &stride);
    /** Get stride */
    Size2D stride() const;
    /** Set dilation */
    DepthwiseConv2dAttributes &dilation(const Size2D &dilation);
    /** Get dilation */
    Size2D dilation() const;
    /** Set depth multiplier */
    DepthwiseConv2dAttributes &depth_multiplier(const uint32_t &depth_multiplier);
    /** Get depth multiplier */
    uint32_t depth_multiplier() const;
    /** Set Dimension rounding type */
    DepthwiseConv2dAttributes &dimension_rounding_type(const DimensionRoundingType &dimension_rounding_type);
    /** Get Dimension rounding type */
    DimensionRoundingType dimension_rounding_type() const;

private:
    Padding2D             _pad{};                                                   /**< Padding */
    Size2D                _stride{ 1U, 1U };                                        /**< Stride */
    Size2D                _dilation{ 1U, 1U };                                      /**< Dilation */
    uint32_t              _depth_multiplier{ 1U };                                  /**< Depth multiplier */
    DimensionRoundingType _dimension_rounding_type{ DimensionRoundingType::FLOOR }; /**< Dimension rounding type */
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_DEPTHWISECONV2DATTRIBUTES */
