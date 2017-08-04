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
#ifndef ARM_COMPUTE_TEST_LARGE_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_LARGE_CONVOLUTION_LAYER_DATASET

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
class LargeConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    LargeConvolutionLayerDataset()
    {
        // Batch size 1
        add_config(TensorShape(227U, 227U, 3U), TensorShape(11U, 11U, 3U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U), PadStrideInfo(4, 4, 0, 0));
        add_config(TensorShape(27U, 27U, 96U), TensorShape(5U, 5U, 96U, 256U), TensorShape(256U), TensorShape(27U, 27U, 256U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(13U, 13U, 256U), TensorShape(3U, 3U, 256U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 256U), TensorShape(256U), TensorShape(13U, 13U, 256U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(224U, 224U, 3U), TensorShape(7U, 7U, 3U, 64U), TensorShape(64U), TensorShape(112U, 112U, 64U), PadStrideInfo(2, 2, 3, 3));
        add_config(TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 64U), TensorShape(64U), TensorShape(28U, 28U, 64U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 4
        add_config(TensorShape(227U, 227U, 3U, 4U), TensorShape(11U, 11U, 3U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U, 4U), PadStrideInfo(4, 4, 0, 0));
        add_config(TensorShape(27U, 27U, 96U, 4U), TensorShape(5U, 5U, 96U, 256U), TensorShape(256U), TensorShape(27U, 27U, 256U, 4U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(13U, 13U, 256U, 4U), TensorShape(3U, 3U, 256U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U, 4U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 384U, 4U), TensorShape(3U, 3U, 384U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U, 4U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 384U, 4U), TensorShape(3U, 3U, 384U, 256U), TensorShape(256U), TensorShape(13U, 13U, 256U, 4U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(224U, 224U, 3U, 4U), TensorShape(7U, 7U, 3U, 64U), TensorShape(64U), TensorShape(112U, 112U, 64U, 4U), PadStrideInfo(2, 2, 3, 3));
        add_config(TensorShape(28U, 28U, 256U, 4U), TensorShape(1U, 1U, 256U, 64U), TensorShape(64U), TensorShape(28U, 28U, 64U, 4U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 8
        add_config(TensorShape(227U, 227U, 3U, 8U), TensorShape(11U, 11U, 3U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U, 8U), PadStrideInfo(4, 4, 0, 0));
        add_config(TensorShape(27U, 27U, 96U, 8U), TensorShape(5U, 5U, 96U, 256U), TensorShape(256U), TensorShape(27U, 27U, 256U, 8U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(13U, 13U, 256U, 8U), TensorShape(3U, 3U, 256U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U, 8U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 384U, 8U), TensorShape(3U, 3U, 384U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U, 8U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 384U, 8U), TensorShape(3U, 3U, 384U, 256U), TensorShape(256U), TensorShape(13U, 13U, 256U, 8U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(224U, 224U, 3U, 8U), TensorShape(7U, 7U, 3U, 64U), TensorShape(64U), TensorShape(112U, 112U, 64U, 8U), PadStrideInfo(2, 2, 3, 3));
        add_config(TensorShape(28U, 28U, 256U, 8U), TensorShape(1U, 1U, 256U, 64U), TensorShape(64U), TensorShape(28U, 28U, 64U, 8U), PadStrideInfo(1, 1, 0, 0));
        // Arbitrary batch size
        add_config(TensorShape(227U, 227U, 3U, 5U), TensorShape(11U, 11U, 3U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U, 5U), PadStrideInfo(4, 4, 0, 0));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LARGE_CONVOLUTION_LAYER_DATASET */
