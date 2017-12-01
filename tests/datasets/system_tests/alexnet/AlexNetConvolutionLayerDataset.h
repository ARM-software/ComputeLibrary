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
#ifndef ARM_COMPUTE_TEST_ALEXNET_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_ALEXNET_CONVOLUTION_LAYER_DATASET

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
class AlexNetWinogradLayerDataset final : public ConvolutionLayerDataset
{
public:
    AlexNetWinogradLayerDataset()
    {
        add_config(TensorShape(13U, 13U, 256U), TensorShape(3U, 3U, 256U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 192U), TensorShape(3U, 3U, 192U, 192U), TensorShape(192U), TensorShape(13U, 13U, 192U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 192U), TensorShape(3U, 3U, 192U, 128U), TensorShape(128U), TensorShape(13U, 13U, 128U), PadStrideInfo(1, 1, 1, 1));
    }
};

class AlexNetConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    AlexNetConvolutionLayerDataset()
    {
        add_config(TensorShape(227U, 227U, 3U), TensorShape(11U, 11U, 3U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U), PadStrideInfo(4, 4, 0, 0));
        add_config(TensorShape(27U, 27U, 48U), TensorShape(5U, 5U, 48U, 128U), TensorShape(128U), TensorShape(27U, 27U, 128U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(13U, 13U, 256U), TensorShape(3U, 3U, 256U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 192U), TensorShape(3U, 3U, 192U, 192U), TensorShape(192U), TensorShape(13U, 13U, 192U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 192U), TensorShape(3U, 3U, 192U, 128U), TensorShape(128U), TensorShape(13U, 13U, 128U), PadStrideInfo(1, 1, 1, 1));
    }
};

class AlexNetDirectConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    AlexNetDirectConvolutionLayerDataset()
    {
        add_config(TensorShape(27U, 27U, 48U), TensorShape(5U, 5U, 48U, 128U), TensorShape(128U), TensorShape(27U, 27U, 128U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(13U, 13U, 256U), TensorShape(3U, 3U, 256U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 192U), TensorShape(3U, 3U, 192U, 192U), TensorShape(192U), TensorShape(13U, 13U, 192U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 192U), TensorShape(3U, 3U, 192U, 128U), TensorShape(128U), TensorShape(13U, 13U, 128U), PadStrideInfo(1, 1, 1, 1));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ALEXNET_CONVOLUTION_LAYER_DATASET */
