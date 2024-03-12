/*
 * Copyright (c) 2017-2020, 2023 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_LARGECONVOLUTIONLAYERDATASET_H
#define ACL_TESTS_DATASETS_LARGECONVOLUTIONLAYERDATASET_H

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
        add_config(TensorShape(224U, 222U, 32U), TensorShape(3U, 3U, 32U, 32U), TensorShape(32U), TensorShape(224U, 222U, 32U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(112U, 113U, 32U), TensorShape(3U, 3U, 32U, 64U), TensorShape(64U), TensorShape(112U, 113U, 64U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(112U, 112U, 64U), TensorShape(3U, 3U, 64U, 129U), TensorShape(129U), TensorShape(112U, 112U, 129U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(53U, 56U, 125U), TensorShape(3U, 3U, 125U, 128U), TensorShape(128U), TensorShape(51U, 54U, 128U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(56U, 56U, 128U), TensorShape(3U, 3U, 128U, 128U), TensorShape(128U), TensorShape(54U, 54U, 128U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(28U, 28U, 257U), TensorShape(3U, 3U, 257U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 1, 1));

        // Batch > 1
        add_config(TensorShape(111U, 112U, 127U, 4U), TensorShape(3U, 3U, 127U, 64U), TensorShape(64U), TensorShape(111U, 112U, 64U, 4U), PadStrideInfo(1, 1, 1, 1));
    }
};

class LargeWinogradConvolutionLayer3x3DatasetFp16Subset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer3x3DatasetFp16Subset()
    {
        // Kernel size 3
        // Batch size 1
        add_config(TensorShape(224U, 222U, 32U), TensorShape(3U, 3U, 32U, 32U), TensorShape(32U), TensorShape(224U, 222U, 32U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(112U, 112U, 64U), TensorShape(3U, 3U, 64U, 129U), TensorShape(129U), TensorShape(112U, 112U, 129U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(56U, 56U, 128U), TensorShape(3U, 3U, 128U, 128U), TensorShape(128U), TensorShape(54U, 54U, 128U), PadStrideInfo(1, 1, 0, 0));

        // Batch > 1
        add_config(TensorShape(111U, 112U, 127U, 4U), TensorShape(3U, 3U, 127U, 64U), TensorShape(64U), TensorShape(111U, 112U, 64U, 4U), PadStrideInfo(1, 1, 1, 1));
    }
};

class LargeWinogradConvolutionLayer3x1Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer3x1Dataset()
    {
        // Kernel size 3
        // Batch size 1
        add_config(TensorShape(224U, 222U, 32U), TensorShape(3U, 1U, 32U, 32U), TensorShape(32U), TensorShape(224U, 222U, 32U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(112U, 113U, 32U), TensorShape(3U, 1U, 32U, 64U), TensorShape(64U), TensorShape(112U, 113U, 64U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(112U, 112U, 64U), TensorShape(3U, 1U, 64U, 129U), TensorShape(129U), TensorShape(112U, 112U, 129U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(53U, 56U, 125U), TensorShape(3U, 1U, 125U, 128U), TensorShape(128U), TensorShape(51U, 56U, 128U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(56U, 56U, 128U), TensorShape(3U, 1U, 128U, 128U), TensorShape(128U), TensorShape(56U, 56U, 128U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(28U, 28U, 257U), TensorShape(3U, 1U, 257U, 128U), TensorShape(128U), TensorShape(26U, 28U, 128U), PadStrideInfo(1, 1, 0, 0));

        // Batch > 1
        add_config(TensorShape(111U, 112U, 127U, 4U), TensorShape(3U, 1U, 127U, 64U), TensorShape(64U), TensorShape(111U, 112U, 64U, 4U), PadStrideInfo(1, 1, 1, 0));
    }
};

class LargeWinogradConvolutionLayer3x1DatasetFp16Subset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer3x1DatasetFp16Subset()
    {
        // Kernel size 3
        // Batch size 1
        add_config(TensorShape(112U, 113U, 32U), TensorShape(3U, 1U, 32U, 64U), TensorShape(64U), TensorShape(112U, 113U, 64U), PadStrideInfo(1, 1, 1, 0));
        add_config(TensorShape(53U, 56U, 125U), TensorShape(3U, 1U, 125U, 128U), TensorShape(128U), TensorShape(51U, 56U, 128U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(28U, 28U, 257U), TensorShape(3U, 1U, 257U, 128U), TensorShape(128U), TensorShape(26U, 28U, 128U), PadStrideInfo(1, 1, 0, 0));

        // Batch > 1
        add_config(TensorShape(111U, 112U, 127U, 4U), TensorShape(3U, 1U, 127U, 64U), TensorShape(64U), TensorShape(111U, 112U, 64U, 4U), PadStrideInfo(1, 1, 1, 0));
    }
};

class LargeWinogradConvolutionLayer1x3Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer1x3Dataset()
    {
        // Kernel size 3
        // Batch size 1
        add_config(TensorShape(224U, 222U, 32U), TensorShape(1U, 3U, 32U, 32U), TensorShape(32U), TensorShape(224U, 222U, 32U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(112U, 113U, 32U), TensorShape(1U, 3U, 32U, 64U), TensorShape(64U), TensorShape(112U, 113U, 64U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(112U, 112U, 64U), TensorShape(1U, 3U, 64U, 129U), TensorShape(129U), TensorShape(112U, 110U, 129U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(53U, 56U, 125U), TensorShape(1U, 3U, 125U, 128U), TensorShape(128U), TensorShape(53U, 56U, 128U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(56U, 56U, 128U), TensorShape(1U, 3U, 128U, 128U), TensorShape(128U), TensorShape(56U, 54U, 128U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(28U, 28U, 257U), TensorShape(1U, 3U, 257U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 0, 1));

        // Batch > 1
        add_config(TensorShape(111U, 112U, 127U, 4U), TensorShape(1U, 3U, 127U, 64U), TensorShape(64U), TensorShape(111U, 112U, 64U, 4U), PadStrideInfo(1, 1, 0, 1));
    }
};

class LargeWinogradConvolutionLayer1x3DatasetFp16Subset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer1x3DatasetFp16Subset()
    {
        // Kernel size 3
        // Batch size 1
        add_config(TensorShape(112U, 112U, 64U), TensorShape(1U, 3U, 64U, 129U), TensorShape(129U), TensorShape(112U, 110U, 129U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(53U, 56U, 125U), TensorShape(1U, 3U, 125U, 128U), TensorShape(128U), TensorShape(53U, 56U, 128U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(28U, 28U, 257U), TensorShape(1U, 3U, 257U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 0, 1));

        // Batch > 1
        add_config(TensorShape(111U, 112U, 127U, 4U), TensorShape(1U, 3U, 127U, 64U), TensorShape(64U), TensorShape(111U, 112U, 64U, 4U), PadStrideInfo(1, 1, 0, 1));
    }
};

class LargeWinogradConvolutionLayer5x5Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer5x5Dataset()
    {
        // Kernel size 5
        // Batch size 1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(5U, 5U, 3U, 32U), TensorShape(32U), TensorShape(220U, 220U, 32U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(181U, 152U, 42U), TensorShape(5U, 5U, 42U, 100U), TensorShape(100U), TensorShape(177U, 148U, 100U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(200U, 201U, 24U), TensorShape(5U, 5U, 24U, 61), TensorShape(61U), TensorShape(200U, 201U, 61), PadStrideInfo(1, 1, 2, 2));

        // Batch > 1
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(5U, 5U, 16U, 7U), TensorShape(7U), TensorShape(123U, 134U, 7U, 3U), PadStrideInfo(1, 1, 2, 2));
    }
};

class LargeWinogradConvolutionLayer5x5DatasetFp16Subset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer5x5DatasetFp16Subset()
    {
        // Kernel size 5
        // Batch size 1
        add_config(TensorShape(181U, 152U, 42U), TensorShape(5U, 5U, 42U, 100U), TensorShape(100U), TensorShape(177U, 148U, 100U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(200U, 201U, 24U), TensorShape(5U, 5U, 24U, 61), TensorShape(61U), TensorShape(200U, 201U, 61), PadStrideInfo(1, 1, 2, 2));

        // Batch > 1
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(5U, 5U, 16U, 7U), TensorShape(7U), TensorShape(123U, 134U, 7U, 3U), PadStrideInfo(1, 1, 2, 2));
    }
};

class LargeWinogradConvolutionLayer5x1Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer5x1Dataset()
    {
        // Batch size 1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(5U, 1U, 3U, 32U), TensorShape(32U), TensorShape(224U, 224U, 32U), PadStrideInfo(1, 1, 2, 0));
        add_config(TensorShape(181U, 152U, 42U), TensorShape(5U, 1U, 42U, 100U), TensorShape(100U), TensorShape(177U, 152U, 100U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(200U, 201U, 24U), TensorShape(5U, 1U, 24U, 61), TensorShape(61U), TensorShape(200U, 201U, 61), PadStrideInfo(1, 1, 2, 0));

        // Batch > 1
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(5U, 1U, 16U, 7U), TensorShape(7U), TensorShape(123U, 134U, 7U, 3U), PadStrideInfo(1, 1, 2, 0));
    }
};

class LargeWinogradConvolutionLayer5x1DatasetFp16Subset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer5x1DatasetFp16Subset()
    {
        // Batch size 1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(5U, 1U, 3U, 32U), TensorShape(32U), TensorShape(224U, 224U, 32U), PadStrideInfo(1, 1, 2, 0));
        add_config(TensorShape(200U, 201U, 24U), TensorShape(5U, 1U, 24U, 61), TensorShape(61U), TensorShape(200U, 201U, 61), PadStrideInfo(1, 1, 2, 0));

        // Batch > 1
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(5U, 1U, 16U, 7U), TensorShape(7U), TensorShape(123U, 134U, 7U, 3U), PadStrideInfo(1, 1, 2, 0));
    }
};

class LargeWinogradConvolutionLayer7x1Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer7x1Dataset()
    {
        // Batch size 1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(7U, 1U, 3U, 32U), TensorShape(32U), TensorShape(218U, 224U, 32U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(181U, 152U, 42U), TensorShape(7U, 1U, 42U, 100U), TensorShape(100U), TensorShape(175U, 152U, 100U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(200U, 201U, 24U), TensorShape(7U, 1U, 24U, 61), TensorShape(61U), TensorShape(200U, 201U, 61), PadStrideInfo(1, 1, 3, 0));

        // Batch > 1
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(7U, 1U, 16U, 7U), TensorShape(7U), TensorShape(123U, 134U, 7U, 3U), PadStrideInfo(1, 1, 3, 0));
    }
};

class LargeWinogradConvolutionLayer1x7Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer1x7Dataset()
    {
        // Batch size 1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(1U, 7U, 3U, 32U), TensorShape(32U), TensorShape(224U, 218U, 32U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(181U, 152U, 42U), TensorShape(1U, 7U, 42U, 100U), TensorShape(100U), TensorShape(181U, 146U, 100U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(200U, 201U, 24U), TensorShape(1U, 7U, 24U, 61), TensorShape(61U), TensorShape(200U, 201U, 61), PadStrideInfo(1, 1, 0, 3));

        // Batch > 1
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(1U, 7U, 16U, 7U), TensorShape(7U), TensorShape(123U, 134U, 7U, 3U), PadStrideInfo(1, 1, 0, 3));
    }
};

class LargeWinogradConvolutionLayer1x7DatasetFp16Subset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer1x7DatasetFp16Subset()
    {
        // Batch size 1
        add_config(TensorShape(181U, 152U, 42U), TensorShape(1U, 7U, 42U, 100U), TensorShape(100U), TensorShape(181U, 146U, 100U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(200U, 201U, 24U), TensorShape(1U, 7U, 24U, 61), TensorShape(61U), TensorShape(200U, 201U, 61), PadStrideInfo(1, 1, 0, 3));

        // Batch > 1
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(1U, 7U, 16U, 7U), TensorShape(7U), TensorShape(123U, 134U, 7U, 3U), PadStrideInfo(1, 1, 0, 3));
    }
};

class LargeWinogradConvolutionLayer1x5Dataset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer1x5Dataset()
    {
        // Batch size 1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(1U, 5U, 3U, 32U), TensorShape(32U), TensorShape(224U, 224U, 32U), PadStrideInfo(1, 1, 0, 2));
        add_config(TensorShape(181U, 152U, 42U), TensorShape(1U, 5U, 42U, 100U), TensorShape(100U), TensorShape(181U, 148U, 100U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(200U, 201U, 24U), TensorShape(1U, 5U, 24U, 61), TensorShape(61U), TensorShape(200U, 201U, 61), PadStrideInfo(1, 1, 0, 2));

        // Batch size > 1
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(1U, 5U, 16U, 7U), TensorShape(7U), TensorShape(123U, 130U, 7U, 3U), PadStrideInfo(1, 1, 0, 0));
    }
};

class LargeWinogradConvolutionLayer1x5DatasetFp16Subset final : public ConvolutionLayerDataset
{
public:
    LargeWinogradConvolutionLayer1x5DatasetFp16Subset()
    {
        // Batch size 1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(1U, 5U, 3U, 32U), TensorShape(32U), TensorShape(224U, 224U, 32U), PadStrideInfo(1, 1, 0, 2));
        add_config(TensorShape(181U, 152U, 42U), TensorShape(1U, 5U, 42U, 100U), TensorShape(100U), TensorShape(181U, 148U, 100U), PadStrideInfo(1, 1, 0, 0));

        // Batch size > 1
        add_config(TensorShape(123U, 134U, 16U, 3U), TensorShape(1U, 5U, 16U, 7U), TensorShape(7U), TensorShape(123U, 130U, 7U, 3U), PadStrideInfo(1, 1, 0, 0));
    }
};

class LargeConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    /** Shapes taken from use-cases such as AlexNet, MobileNet, SqueezeNet, etc. */
    LargeConvolutionLayerDataset()
    {
        // Batch size 1
        add_config(TensorShape(227U, 227U, 3U), TensorShape(11U, 11U, 3U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U), PadStrideInfo(4, 4, 0, 0));
        add_config(TensorShape(27U, 27U, 96U), TensorShape(5U, 5U, 96U, 256U), TensorShape(256U), TensorShape(27U, 27U, 256U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(13U, 13U, 256U), TensorShape(1U, 1U, 256U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(224U, 224U, 3U), TensorShape(7U, 7U, 3U, 64U), TensorShape(64U), TensorShape(112U, 112U, 64U), PadStrideInfo(2, 2, 3, 3));
        // Batch size 4
        add_config(TensorShape(227U, 227U, 3U, 4U), TensorShape(11U, 11U, 3U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U, 4U), PadStrideInfo(4, 4, 0, 0));
        add_config(TensorShape(27U, 27U, 96U, 4U), TensorShape(5U, 5U, 96U, 256U), TensorShape(256U), TensorShape(27U, 27U, 256U, 4U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(13U, 13U, 256U, 4U), TensorShape(1U, 1U, 256U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U, 4U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(224U, 224U, 3U, 4U), TensorShape(7U, 7U, 3U, 64U), TensorShape(64U), TensorShape(112U, 112U, 64U, 4U), PadStrideInfo(2, 2, 3, 3));
    }
};

class LargeGroupedConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    LargeGroupedConvolutionLayerDataset()
    {
        // Batch size 1
        add_config(TensorShape(227U, 227U, 4U), TensorShape(11U, 11U, 2U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U), PadStrideInfo(4, 4, 0, 0));
        add_config(TensorShape(27U, 27U, 96U), TensorShape(5U, 5U, 24U, 256U), TensorShape(256U), TensorShape(27U, 27U, 256U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(13U, 13U, 256U), TensorShape(1U, 1U, 128U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 128U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 1, 1));
        // Batch size 4
        add_config(TensorShape(227U, 227U, 4U, 4U), TensorShape(11U, 11U, 2U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U, 4U), PadStrideInfo(4, 4, 0, 0));
        add_config(TensorShape(27U, 27U, 96U, 4U), TensorShape(5U, 5U, 24U, 256U), TensorShape(256U), TensorShape(27U, 27U, 256U, 4U), PadStrideInfo(1, 1, 2, 2));
        add_config(TensorShape(13U, 13U, 256U, 4U), TensorShape(3U, 3U, 128U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U, 4U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 384U, 4U), TensorShape(3U, 3U, 128U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U, 4U), PadStrideInfo(1, 1, 1, 1));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_DATASETS_LARGECONVOLUTIONLAYERDATASET_H
