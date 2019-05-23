/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_SMALL_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_SMALL_CONVOLUTION_LAYER_DATASET

#include "tests/datasets/ConvolutionLayerDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class SmallWinogradConvolutionLayer3x3Dataset final : public ConvolutionLayerDataset
{
public:
    SmallWinogradConvolutionLayer3x3Dataset()
    {
        // Channel size big enough to force multithreaded execution of the input transform
        add_config(TensorShape(8U, 8U, 32U), TensorShape(3U, 3U, 32U, 1U), TensorShape(1U), TensorShape(6U, 6U, 1U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 1
        add_config(TensorShape(8U, 8U, 2U), TensorShape(3U, 3U, 2U, 1U), TensorShape(1U), TensorShape(6U, 6U, 1U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 4
        add_config(TensorShape(23U, 27U, 5U, 4U), TensorShape(3U, 3U, 5U, 21U), TensorShape(21U), TensorShape(21U, 25U, 21U, 4U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(8U, 8U, 2U), TensorShape(3U, 3U, 2U, 1U), TensorShape(1U), TensorShape(8U, 8U, 1U), PadStrideInfo(1, 1, 1, 1));
    }
};

class SmallWinogradConvolutionLayer3x1Dataset final : public ConvolutionLayerDataset
{
public:
    SmallWinogradConvolutionLayer3x1Dataset()
    {
        // Channel size big enough to force multithreaded execution of the input transform
        add_config(TensorShape(8U, 8U, 32U), TensorShape(3U, 1U, 32U, 1U), TensorShape(1U), TensorShape(6U, 8U, 1U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 1
        add_config(TensorShape(8U, 8U, 2U), TensorShape(3U, 1U, 2U, 1U), TensorShape(1U), TensorShape(6U, 8U, 1U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 4
        add_config(TensorShape(23U, 27U, 5U, 4U), TensorShape(3U, 1U, 5U, 21U), TensorShape(21U), TensorShape(21U, 27U, 21U, 4U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(8U, 8U, 2U), TensorShape(3U, 1U, 2U, 1U), TensorShape(1U), TensorShape(8U, 8U, 1U), PadStrideInfo(1, 1, 1, 0));
    }
};

class SmallWinogradConvolutionLayer1x3Dataset final : public ConvolutionLayerDataset
{
public:
    SmallWinogradConvolutionLayer1x3Dataset()
    {
        // Channel size big enough to force multithreaded execution of the input transform
        add_config(TensorShape(8U, 8U, 32U), TensorShape(1U, 3U, 32U, 1U), TensorShape(1U), TensorShape(8U, 6U, 1U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 1
        add_config(TensorShape(8U, 8U, 2U), TensorShape(1U, 3U, 2U, 1U), TensorShape(1U), TensorShape(8U, 6U, 1U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 4
        add_config(TensorShape(23U, 27U, 5U, 4U), TensorShape(1U, 3U, 5U, 21U), TensorShape(21U), TensorShape(23U, 25U, 21U, 4U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(8U, 8U, 2U), TensorShape(1U, 3U, 2U, 1U), TensorShape(1U), TensorShape(8U, 8U, 1U), PadStrideInfo(1, 1, 0, 1));
    }
};

class SmallWinogradConvolutionLayer5x5Dataset final : public ConvolutionLayerDataset
{
public:
    SmallWinogradConvolutionLayer5x5Dataset()
    {
        add_config(TensorShape(8U, 8U, 2U), TensorShape(5U, 5U, 2U, 1U), TensorShape(1U), TensorShape(4U, 4U, 1U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(8U, 8U, 2U), TensorShape(5U, 5U, 2U), TensorShape(1U), TensorShape(8U, 8U, 1U), PadStrideInfo(1, 1, 2, 2));
    }
};

class SmallWinogradConvolutionLayer5x1Dataset final : public ConvolutionLayerDataset
{
public:
    SmallWinogradConvolutionLayer5x1Dataset()
    {
        add_config(TensorShape(8U, 8U, 2U), TensorShape(5U, 1U, 2U, 1U), TensorShape(1U), TensorShape(4U, 8U, 1U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(8U, 8U, 2U), TensorShape(5U, 1U, 2U), TensorShape(1U), TensorShape(8U, 8U, 1U), PadStrideInfo(1, 1, 2, 0));
    }
};

class SmallWinogradConvolutionLayer1x5Dataset final : public ConvolutionLayerDataset
{
public:
    SmallWinogradConvolutionLayer1x5Dataset()
    {
        add_config(TensorShape(8U, 8U, 2U), TensorShape(1U, 5U, 2U, 1U), TensorShape(1U), TensorShape(8U, 4U, 1U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(8U, 8U, 2U), TensorShape(1U, 5U, 2U), TensorShape(1U), TensorShape(8U, 8U, 1U), PadStrideInfo(1, 1, 0, 2));
    }
};

class SmallWinogradConvolutionLayer7x1Dataset final : public ConvolutionLayerDataset
{
public:
    SmallWinogradConvolutionLayer7x1Dataset()
    {
        add_config(TensorShape(14U, 14U, 2U), TensorShape(7U, 1U, 2U, 1U), TensorShape(1U), TensorShape(8U, 14U, 1U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(14U, 14U, 2U), TensorShape(7U, 1U, 2U), TensorShape(1U), TensorShape(14U, 14U, 1U), PadStrideInfo(1, 1, 3, 0));
    }
};

class SmallWinogradConvolutionLayer1x7Dataset final : public ConvolutionLayerDataset
{
public:
    SmallWinogradConvolutionLayer1x7Dataset()
    {
        add_config(TensorShape(14U, 14U, 2U), TensorShape(1U, 7U, 2U, 1U), TensorShape(1U), TensorShape(14U, 8U, 1U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(14U, 14U, 2U), TensorShape(1U, 7U, 2U), TensorShape(1U), TensorShape(14U, 14U, 1U), PadStrideInfo(1, 1, 0, 3));
    }
};

class SmallFFTConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    SmallFFTConvolutionLayerDataset()
    {
        add_config(TensorShape(8U, 7U, 3U), TensorShape(3U, 3U, 3U, 2U), TensorShape(2U), TensorShape(8U, 7U, 2U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(64U, 32U, 5U), TensorShape(5U, 5U, 5U, 10U), TensorShape(10U), TensorShape(64U, 32U, 10U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(192U, 128U, 8U), TensorShape(9U, 9U, 8U, 3U), TensorShape(3U), TensorShape(192U, 128U, 3U), PadStrideInfo(1, 1, 4, 4));
    }
};

class SmallConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    SmallConvolutionLayerDataset()
    {
        add_config(TensorShape(224U, 224U, 3U), TensorShape(3U, 3U, 3U, 32U), TensorShape(32U), TensorShape(112U, 112U, 32U),
                   PadStrideInfo(2, 2, /*left*/ 0, /*right*/ 1, /*top*/ 0, /*bottom*/ 1, DimensionRoundingType::FLOOR));

        // 1D Kernel
        add_config(TensorShape(1U, 5U, 2U), TensorShape(1U, 3U, 2U, 3U), TensorShape(3U), TensorShape(1U, 7U, 3U), PadStrideInfo(1, 1, 0, 0, 2, 2, DimensionRoundingType::FLOOR));

        // Batch size 1
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 3U, 5U, 21U), TensorShape(21U), TensorShape(11U, 25U, 21U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 5U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(5U, 5U, 2U, 19U), TensorShape(19U), TensorShape(15U, 15U, 19U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 1U, 5U, 21U), TensorShape(21U), TensorShape(11U, 27U, 21U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 11U, 16U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(5U, 3U, 2U, 19U), TensorShape(19U), TensorShape(15U, 16U, 19U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(3U, 3U, 1U), TensorShape(2U, 2U, 1U, 11U), TensorShape(11U), TensorShape(2U, 2U, 11U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 4
        add_config(TensorShape(23U, 27U, 5U, 4U), TensorShape(3U, 3U, 5U, 21U), TensorShape(21U), TensorShape(11U, 25U, 21U, 4U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U, 4U), TensorShape(5U, 5U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 4U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(17U, 31U, 2U, 4U), TensorShape(5U, 5U, 2U, 19U), TensorShape(19U), TensorShape(15U, 15U, 19U, 4U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(23U, 27U, 5U, 4U), TensorShape(3U, 1U, 5U, 21U), TensorShape(21U), TensorShape(11U, 27U, 21U, 4U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U, 4U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 11U, 16U, 4U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(17U, 31U, 2U, 4U), TensorShape(5U, 3U, 2U, 19U), TensorShape(19U), TensorShape(15U, 16U, 19U, 4U), PadStrideInfo(1, 2, 1, 1));
        // Arbitrary batch size
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 11U, 16U, 5U), PadStrideInfo(3, 2, 1, 0));
        // FC convolution
        add_config(TensorShape(1U, 1U, 1024U), TensorShape(1U, 1U, 1024U, 1001U), TensorShape(1001U), TensorShape(1U, 1U, 1001U), PadStrideInfo(1, 1, 0, 0));
        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 1, 1, 2, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 1, 1, 0, 2, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 2, 1, 2, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 1, 3, 0, 2, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(10U, 11U, 16U, 5U), PadStrideInfo(3, 2, 1, 0, 1, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(10U, 11U, 16U, 5U), PadStrideInfo(3, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR));

        add_config(TensorShape(5U, 4U, 3U, 2U), TensorShape(4U, 4U, 3U, 1U), TensorShape(1U), TensorShape(2U, 1U, 1U, 2U), PadStrideInfo(1, 1, 0, 0, 0, 0, DimensionRoundingType::FLOOR));
    }
};

// TODO (COMPMID-1749)
class SmallConvolutionLayerReducedDataset final : public ConvolutionLayerDataset
{
public:
    SmallConvolutionLayerReducedDataset()
    {
        // Batch size 1
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 3U, 5U, 21U), TensorShape(21U), TensorShape(11U, 25U, 21U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 5U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(5U, 5U, 2U, 19U), TensorShape(19U), TensorShape(15U, 15U, 19U), PadStrideInfo(1, 2, 1, 1));

        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 1, 1, 2, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 1, 1, 0, 2, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U, 5U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 2, 1, 2, 0, DimensionRoundingType::FLOOR));
    }
};

class SmallGroupedConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    SmallGroupedConvolutionLayerDataset()
    {
        // Batch size 1
        // Number of groups = 2
        add_config(TensorShape(23U, 27U, 8U), TensorShape(1U, 1U, 4U, 24U), TensorShape(24U), TensorShape(12U, 27U, 24U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 12U), TensorShape(5U, 5U, 6U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U), PadStrideInfo(3, 2, 1, 0));
        // Number of groups = 4
        add_config(TensorShape(23U, 27U, 8U), TensorShape(1U, 1U, 2U, 24U), TensorShape(24U), TensorShape(12U, 27U, 24U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 12U), TensorShape(5U, 5U, 4U, 15U), TensorShape(15U), TensorShape(11U, 12U, 15U), PadStrideInfo(3, 2, 1, 0));

        // Batch size 4
        // Number of groups = 2
        add_config(TensorShape(23U, 27U, 8U, 4U), TensorShape(1U, 1U, 4U, 24U), TensorShape(24U), TensorShape(12U, 27U, 24U, 4U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 12U, 4U), TensorShape(5U, 5U, 6U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 4U), PadStrideInfo(3, 2, 1, 0));
        // Number of groups = 4
        add_config(TensorShape(23U, 27U, 8U, 4U), TensorShape(1U, 1U, 2U, 24U), TensorShape(24U), TensorShape(12U, 27U, 24U, 4U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 12U, 4U), TensorShape(5U, 5U, 4U, 15U), TensorShape(15U), TensorShape(11U, 12U, 15U, 4U), PadStrideInfo(3, 2, 1, 0));

        // Arbitrary batch size
        add_config(TensorShape(23U, 27U, 8U, 5U), TensorShape(1U, 1U, 4U, 24U), TensorShape(24U), TensorShape(12U, 27U, 24U, 5U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 12U, 3U), TensorShape(5U, 5U, 6U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 3U), PadStrideInfo(3, 2, 1, 0));
        // Number of groups = 4
        add_config(TensorShape(23U, 27U, 8U, 2U), TensorShape(1U, 1U, 2U, 24U), TensorShape(24U), TensorShape(12U, 27U, 24U, 2U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 12U, 5U), TensorShape(5U, 5U, 4U, 15U), TensorShape(15U), TensorShape(11U, 12U, 15U, 5U), PadStrideInfo(3, 2, 1, 0));

        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 8U, 5U), TensorShape(5U, 7U, 2U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 1, 1, 2, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 8U, 5U), TensorShape(5U, 7U, 4U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 1, 1, 0, 2, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 6U, 5U), TensorShape(5U, 7U, 3U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U, 5U), PadStrideInfo(3, 2, 2, 1, 2, 0, DimensionRoundingType::FLOOR));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SMALL_CONVOLUTION_LAYER_DATASET */
