/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_DILATED_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_DILATED_CONVOLUTION_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/datasets/DepthwiseConvolutionLayerDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
/** Dataset containing small, generic depthwise convolution shapes with dilation. */
class SmallDepthwiseDilatedConvolutionLayerDataset final : public DepthwiseConvolutionLayerDataset
{
public:
    SmallDepthwiseDilatedConvolutionLayerDataset()
    {
        // Different strides and dilations
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 2U), PadStrideInfo(1, 1, 0, 0), Size2D(2U, 2U));
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 2U), PadStrideInfo(1, 2, 0, 0), Size2D(2U, 1U));
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 2U), PadStrideInfo(1, 1, 0, 0), Size2D(2U, 1U));
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 2U), PadStrideInfo(2, 1, 0, 0), Size2D(2U, 2U));
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 2U), PadStrideInfo(2, 2, 0, 0), Size2D(1U, 2U));

        add_config(TensorShape(7U, 8U, 1U), Size2D(2U, 3U), PadStrideInfo(1, 2, 0, 0), Size2D(2U, 2U));
        add_config(TensorShape(23U, 27U, 5U), Size2D(3U, 5U), PadStrideInfo(2, 1, 0, 0), Size2D(2U, 1U));
        add_config(TensorShape(33U, 27U, 7U), Size2D(7U, 3U), PadStrideInfo(3, 2, 1, 0), Size2D(1U, 2U));
        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 1, 1, 2, 0, DimensionRoundingType::FLOOR), Size2D(2U, 2U));
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 1, 1, 0, 2, DimensionRoundingType::FLOOR), Size2D(2U, 2U));
    }
};

/** Dataset containing small, 3x3 depthwise convolution shapes with dilation. */
class SmallDepthwiseDilatedConvolutionLayerDataset3x3 final : public DepthwiseConvolutionLayerDataset
{
public:
    SmallDepthwiseDilatedConvolutionLayerDataset3x3()
    {
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 0), Size2D(2U, 2U));
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 3U), PadStrideInfo(1, 1, 2, 0), Size2D(2U, 2U));
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 3U), PadStrideInfo(1, 1, 3, 0), Size2D(2U, 2U));

        // Different strides and dilations
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 0), Size2D(2U, 2U));
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 3U), PadStrideInfo(1, 2, 0, 0), Size2D(2U, 1U));
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 3U), PadStrideInfo(2, 1, 0, 0), Size2D(2U, 2U));
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 0), Size2D(1U, 2U));

        add_config(TensorShape(11U, 11U, 1U), Size2D(3U, 3U), PadStrideInfo(3, 3, 0, 0), Size2D(2U, 2U));
        add_config(TensorShape(7U, 7U, 3U, 2U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 0), Size2D(2U, 2U));

        add_config(TensorShape(21U, 31U, 9U, 4U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 0), Size2D(1U, 1U));

        add_config(TensorShape(21U, 31U, 9U, 4U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 0), Size2D(2U, 2U));
        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 11U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), Size2D(2U, 2U));
    }
};

/** Dataset containing large, generic depthwise convolution shapes with dilation. */
class LargeDepthwiseDilatedConvolutionLayerDataset final : public DepthwiseConvolutionLayerDataset
{
public:
    LargeDepthwiseDilatedConvolutionLayerDataset()
    {
        add_config(TensorShape(33U, 27U, 11U), Size2D(3U, 3U), PadStrideInfo(1, 2, 0, 1), Size2D(2U, 1U));
        add_config(TensorShape(17U, 31U, 2U), Size2D(5U, 9U), PadStrideInfo(1, 2, 1, 1), Size2D(1U, 2U));
        add_config(TensorShape(23U, 27U, 5U), Size2D(11U, 3U), PadStrideInfo(1, 2, 0, 0), Size2D(1U, 3U));
        add_config(TensorShape(17U, 31U, 2U, 3U), Size2D(5U, 9U), PadStrideInfo(1, 2, 1, 1), Size2D(2U, 2U));
        add_config(TensorShape(233U, 277U, 55U), Size2D(3U, 3U), PadStrideInfo(2, 1, 0, 0), Size2D(2U, 2U));
        add_config(TensorShape(333U, 277U, 77U), Size2D(3U, 3U), PadStrideInfo(3, 2, 1, 0), Size2D(3U, 2U));
        add_config(TensorShape(177U, 311U, 22U), Size2D(3U, 3U), PadStrideInfo(1, 2, 1, 1), Size2D(2U, 2U));
        add_config(TensorShape(233U, 277U, 55U), Size2D(3U, 3U), PadStrideInfo(1, 2, 0, 0), Size2D(5U, 2U));
        add_config(TensorShape(333U, 277U, 77U), Size2D(3U, 3U), PadStrideInfo(2, 3, 0, 1), Size2D(2U, 2U));
        add_config(TensorShape(177U, 311U, 22U), Size2D(3U, 3U), PadStrideInfo(2, 1, 1, 1), Size2D(2U, 5U));
        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 2, 1, 2, 0, DimensionRoundingType::FLOOR), Size2D(3U, 2U));
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 1, 3, 0, 2, DimensionRoundingType::FLOOR), Size2D(4U, 4U));
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 1, 0, 1, 0, DimensionRoundingType::FLOOR), Size2D(2U, 2U));
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), Size2D(3U, 3U));
    }
};

/** Dataset containing large, 3x3 depthwise convolution shapes with dilation. */
class LargeDepthwiseDilatedConvolutionLayerDataset3x3 final : public DepthwiseConvolutionLayerDataset
{
public:
    LargeDepthwiseDilatedConvolutionLayerDataset3x3()
    {
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 1), Size2D(2U, 1U));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 1), Size2D(2U, 2U));
        add_config(TensorShape(21U, 31U, 9U, 4U), Size2D(3U, 3U), PadStrideInfo(1, 2, 1, 0), Size2D(2U, 2U));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(1, 2, 0, 1), Size2D(2U, 1U));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(1, 2, 1, 1), Size2D(2U, 3U));
        add_config(TensorShape(21U, 31U, 9U, 4U), Size2D(3U, 3U), PadStrideInfo(2, 1, 1, 0), Size2D(2U, 1U));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 1, 0, 1), Size2D(3U, 3U));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 1, 1, 1), Size2D(2U, 2U));
        add_config(TensorShape(21U, 31U, 9U, 4U), Size2D(3U, 3U), PadStrideInfo(2, 2, 1, 0), Size2D(2U, 2U));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 1), Size2D(4U, 4U));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 2, 1, 1), Size2D(2U, 5U));
        add_config(TensorShape(233U, 277U, 55U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 1, 0, 0), Size2D(3U, 3U));
        add_config(TensorShape(177U, 311U, 22U), Size2D(3U, 3U), PadStrideInfo(1, 2, 1, 1), Size2D(4U, 4U));
        add_config(TensorShape(233U, 277U, 55U), Size2D(3U, 3U), PadStrideInfo(1, 2, 0, 0), Size2D(5U, 5U));
        add_config(TensorShape(333U, 277U, 77U, 5U), Size2D(3U, 3U), PadStrideInfo(2, 3, 0, 1), Size2D(4U, 4U));
        add_config(TensorShape(177U, 311U, 22U), Size2D(3U, 3U), PadStrideInfo(2, 1, 1, 1), Size2D(3U, 3U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DILATED_CONVOLUTION_LAYER_DATASET */
