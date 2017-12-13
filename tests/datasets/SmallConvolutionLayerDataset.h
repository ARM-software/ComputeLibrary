/*
 * Copyright (c) 2017 ARM Limited.
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
class SmallWinogradLayerDataset final : public ConvolutionLayerDataset
{
public:
    SmallWinogradLayerDataset()
    {
        // Batch size 1
        add_config(TensorShape(8U, 8U, 2U), TensorShape(3U, 3U, 2U, 1U), TensorShape(1U), TensorShape(6U, 6U, 1U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 4
        add_config(TensorShape(23U, 27U, 5U, 4U), TensorShape(3U, 3U, 5U, 21U), TensorShape(21U), TensorShape(21U, 25U, 21U, 4U), PadStrideInfo(1, 1, 0, 0));
    }
};

class SmallConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    SmallConvolutionLayerDataset()
    {
        // Batch size 1
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 3U, 5U, 21U), TensorShape(21U), TensorShape(11U, 25U, 21U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 5U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(5U, 5U, 2U, 19U), TensorShape(19U), TensorShape(15U, 15U, 19U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 1U, 5U, 21U), TensorShape(21U), TensorShape(11U, 27U, 21U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 11U, 16U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(5U, 3U, 2U, 19U), TensorShape(19U), TensorShape(15U, 16U, 19U), PadStrideInfo(1, 2, 1, 1));
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
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SMALL_CONVOLUTION_LAYER_DATASET */
