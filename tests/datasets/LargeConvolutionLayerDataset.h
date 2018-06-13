/*
 * Copyright (c) 2017-2018 ARM Limited.
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
class LargeWinogradConvolutionLayer3x3Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer3x3Dataset()
    {
        // Kernel size 3
        // Batch size 1
        add_config(TensorShape(224U, 222U, 64U), TensorShape(3U, 3U, 64U, 64U), TensorShape(64U), TensorShape(224U, 222U, 64U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(112U, 113U, 64U), TensorShape(3U, 3U, 64U, 128U), TensorShape(128U), TensorShape(112U, 113U, 128U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(112U, 112U, 128U), TensorShape(3U, 3U, 128U, 129U), TensorShape(129U), TensorShape(112U, 110U, 129U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(53U, 56U, 125U), TensorShape(3U, 3U, 125U, 256U), TensorShape(256U), TensorShape(51U, 56U, 256U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(56U, 56U, 256U), TensorShape(3U, 3U, 256U, 256U), TensorShape(256U), TensorShape(56U, 54U, 256U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(28U, 28U, 257U), TensorShape(3U, 3U, 257U, 512U), TensorShape(512U), TensorShape(26U, 28U, 512U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(28U, 28U, 512U), TensorShape(3U, 3U, 512U, 512U), TensorShape(512U), TensorShape(28U, 28U, 512U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(14U, 14U, 512U), TensorShape(3U, 3U, 512U, 512U), TensorShape(512U), TensorShape(12U, 12U, 512U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 3, 2 and 4
        add_config(TensorShape(224U, 222U, 64U, 3U), TensorShape(3U, 3U, 64U, 64U), TensorShape(64U), TensorShape(224U, 222U, 64U, 3U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(112U, 113U, 64U, 2U), TensorShape(3U, 3U, 64U, 128U), TensorShape(128U), TensorShape(110U, 113U, 128U, 2U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(111U, 112U, 127U, 4U), TensorShape(3U, 3U, 127U, 128U), TensorShape(128U), TensorShape(111U, 112U, 128U, 4U), PadStrideInfo(1, 1, 1, 1));
    }
};

class LargeWinogradConvolutionLayer3x1Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer3x1Dataset()
    {
        // Kernel size 3
        // Batch size 1
        add_config(TensorShape(224U, 222U, 64U), TensorShape(3U, 1U, 64U, 64U), TensorShape(64U), TensorShape(224U, 222U, 64U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(112U, 113U, 64U), TensorShape(3U, 1U, 64U, 128U), TensorShape(128U), TensorShape(112U, 113U, 128U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(112U, 112U, 128U), TensorShape(3U, 1U, 128U, 129U), TensorShape(129U), TensorShape(112U, 112U, 129U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(53U, 56U, 125U), TensorShape(3U, 1U, 125U, 256U), TensorShape(256U), TensorShape(51U, 56U, 256U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(56U, 56U, 256U), TensorShape(3U, 1U, 256U, 256U), TensorShape(256U), TensorShape(56U, 56U, 256U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(28U, 28U, 257U), TensorShape(3U, 1U, 257U, 512U), TensorShape(512U), TensorShape(26U, 28U, 512U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(28U, 28U, 512U), TensorShape(3U, 1U, 512U, 512U), TensorShape(512U), TensorShape(28U, 28U, 512U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(14U, 14U, 512U), TensorShape(3U, 1U, 512U, 512U), TensorShape(512U), TensorShape(12U, 14U, 512U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 3, 2 and 4
        add_config(TensorShape(224U, 222U, 64U, 3U), TensorShape(3U, 1U, 64U, 64U), TensorShape(64U), TensorShape(224U, 222U, 64U, 3U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(112U, 113U, 64U, 2U), TensorShape(3U, 1U, 64U, 128U), TensorShape(128U), TensorShape(110U, 113U, 128U, 2U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(111U, 112U, 127U, 4U), TensorShape(3U, 1U, 127U, 128U), TensorShape(128U), TensorShape(111U, 112U, 128U, 4U), PadStrideInfo(1, 1, 1, 0));
    }
};

class LargeWinogradConvolutionLayer1x3Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer1x3Dataset()
    {
        // Kernel size 3
        // Batch size 1
        add_config(TensorShape(224U, 222U, 64U), TensorShape(1U, 3U, 64U, 64U), TensorShape(64U), TensorShape(224U, 222U, 64U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(112U, 113U, 64U), TensorShape(1U, 3U, 64U, 128U), TensorShape(128U), TensorShape(112U, 113U, 128U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(112U, 112U, 128U), TensorShape(1U, 3U, 128U, 129U), TensorShape(129U), TensorShape(112U, 110U, 129U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(53U, 56U, 125U), TensorShape(1U, 3U, 125U, 256U), TensorShape(256U), TensorShape(53U, 56U, 256U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(56U, 56U, 256U), TensorShape(1U, 3U, 256U, 256U), TensorShape(256U), TensorShape(56U, 54U, 256U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(28U, 28U, 257U), TensorShape(1U, 3U, 257U, 512U), TensorShape(512U), TensorShape(28U, 28U, 512U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(28U, 28U, 512U), TensorShape(1U, 3U, 512U, 512U), TensorShape(512U), TensorShape(28U, 28U, 512U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 3U, 512U, 512U), TensorShape(512U), TensorShape(14U, 12U, 512U), PadStrideInfo(1, 1, 0, 0));
        // Batch size 3, 2 and 4
        add_config(TensorShape(224U, 222U, 64U, 3U), TensorShape(1U, 3U, 64U, 64U), TensorShape(64U), TensorShape(224U, 222U, 64U, 3U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(112U, 113U, 64U, 2U), TensorShape(1U, 3U, 64U, 128U), TensorShape(128U), TensorShape(112U, 113U, 128U, 2U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(111U, 112U, 127U, 4U), TensorShape(1U, 3U, 127U, 128U), TensorShape(128U), TensorShape(111U, 112U, 128U, 4U), PadStrideInfo(1, 1, 0, 1));
    }
};

class LargeWinogradConvolutionLayer5x5Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer5x5Dataset()
    {
        // Kernel size 5
        // Batch size 1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(5U, 5U, 3U, 64U), TensorShape(64U), TensorShape(222U, 222U, 64U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(123U, 134U, 16U), TensorShape(5U, 5U, 16U, 7U), TensorShape(7U), TensorShape(121U, 130U, 7U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(181U, 152U, 42U), TensorShape(5U, 5U, 42U, 100U), TensorShape(100U), TensorShape(177U, 148U, 100U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(200U, 201U, 24U), TensorShape(5U, 5U, 24U, 61), TensorShape(61U), TensorShape(200U, 201U, 61), PadStrideInfo(1, 1, 2, 2));

        // Batch size 2, 3 and 4
        add_config(TensorShape(224U, 224U, 3U, 2U), TensorShape(5U, 5U, 3U, 64U), TensorShape(64U), TensorShape(222U, 222U, 64U, 2U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(5U, 5U, 16U, 7U), TensorShape(7U), TensorShape(121U, 130U, 7U, 3U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(181U, 152U, 42U, 4U), TensorShape(5U, 5U, 42U, 100U), TensorShape(100U), TensorShape(177U, 148U, 100U, 4U), PadStrideInfo(1, 1, 0, 0));
    }
};

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
