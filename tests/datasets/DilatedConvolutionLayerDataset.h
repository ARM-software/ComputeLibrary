/*
 * Copyright (c) 2018 Arm Limited.
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
#include "tests/datasets/ConvolutionLayerDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class TinyDilatedConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    TinyDilatedConvolutionLayerDataset()
    {
        // Batch size 1
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 3U, 5U, 21U), TensorShape(21U), TensorShape(10U, 23U, 21U), PadStrideInfo(2, 1, 0, 0), Size2D(2U, 2U));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 5U, 7U, 16U), TensorShape(16U), TensorShape(11U, 10U, 16U), PadStrideInfo(3, 2, 1, 0), Size2D(1U, 2U));
        // Batch size 4
        add_config(TensorShape(17U, 31U, 2U, 4U), TensorShape(5U, 5U, 2U, 19U), TensorShape(19U), TensorShape(11U, 13U, 19U, 4U), PadStrideInfo(1, 2, 1, 1), Size2D(2U, 2U));
    }
};

class SmallDilatedConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    SmallDilatedConvolutionLayerDataset()
    {
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 3U, 5U, 21U), TensorShape(21U), TensorShape(10U, 23U, 21U), PadStrideInfo(2, 1, 0, 0), Size2D(2U, 2U));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 5U, 7U, 16U), TensorShape(16U), TensorShape(11U, 10U, 16U), PadStrideInfo(3, 2, 1, 0), Size2D(1U, 2U));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(5U, 5U, 2U, 19U), TensorShape(19U), TensorShape(11U, 15U, 19U), PadStrideInfo(1, 2, 1, 1), Size2D(2U, 1U));
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 1U, 5U, 21U), TensorShape(21U), TensorShape(9U, 27U, 21U), PadStrideInfo(2, 1, 0, 0), Size2D(3U, 1U));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(5U, 3U, 2U, 19U), TensorShape(19U), TensorShape(11U, 15U, 19U), PadStrideInfo(1, 2, 1, 1), Size2D(2U, 2U));
    }
};

class LargeDilatedConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    LargeDilatedConvolutionLayerDataset()
    {
        // Batch size 1
        add_config(TensorShape(27U, 27U, 96U), TensorShape(5U, 5U, 96U, 256U), TensorShape(256U), TensorShape(15U, 15U, 256U), PadStrideInfo(1, 1, 2, 2), Size2D(4U, 4U));
        add_config(TensorShape(13U, 13U, 256U), TensorShape(3U, 3U, 256U, 384U), TensorShape(384U), TensorShape(11U, 9U, 384U), PadStrideInfo(1, 1, 1, 1), Size2D(2U, 3U));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 384U), TensorShape(384U), TensorShape(9U, 11U, 384U), PadStrideInfo(1, 1, 1, 1), Size2D(3U, 2U));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 256U), TensorShape(256U), TensorShape(7U, 7U, 256U), PadStrideInfo(1, 1, 1, 1), Size2D(4U, 4U));
        add_config(TensorShape(224U, 224U, 3U), TensorShape(7U, 7U, 3U, 64U), TensorShape(64U), TensorShape(109U, 112U, 64U), PadStrideInfo(2, 2, 3, 3), Size2D(2U, 1U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DILATED_CONVOLUTION_LAYER_DATASET */
