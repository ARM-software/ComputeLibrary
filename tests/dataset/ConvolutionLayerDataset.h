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
#ifndef __ARM_COMPUTE_TEST_DATASET_CONVOLUTION_LAYER_DATASET_H__
#define __ARM_COMPUTE_TEST_DATASET_CONVOLUTION_LAYER_DATASET_H__

#include "TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "dataset/GenericDataset.h"
#include "dataset/ShapeDatasets.h"

#include <sstream>
#include <type_traits>

#ifdef BOOST
#include "boost_wrapper.h"
#endif

namespace arm_compute
{
namespace test
{
/** Convolution Layer data object */
class ConvolutionLayerDataObject
{
public:
    operator std::string() const
    {
        std::stringstream ss;
        ss << "ConvolutionLayer";
        ss << "_I" << src_shape;
        ss << "_K" << weights_shape;
        ss << "_PS" << info;
        return ss.str();
    }

    friend std::ostream &operator<<(std::ostream &os, const ConvolutionLayerDataObject &obj)
    {
        os << static_cast<std::string>(obj);
        return os;
    }

public:
    TensorShape   src_shape;
    TensorShape   weights_shape;
    TensorShape   bias_shape;
    TensorShape   dst_shape;
    PadStrideInfo info;
};

template <unsigned int Size>
using ConvolutionLayerDataset = GenericDataset<ConvolutionLayerDataObject, Size>;

/** Data set containing small convolution layer shapes */
class SmallConvolutionLayerDataset final : public ConvolutionLayerDataset<6>
{
public:
    SmallConvolutionLayerDataset()
        : GenericDataset
    {
        ConvolutionLayerDataObject{ TensorShape(23U, 27U, 5U), TensorShape(3U, 3U, 5U, 21U), TensorShape(21U), TensorShape(11U, 25U, 21U), PadStrideInfo(2, 1, 0, 0) },
        ConvolutionLayerDataObject{ TensorShape(33U, 27U, 7U), TensorShape(5U, 5U, 7U, 16U), TensorShape(16U), TensorShape(11U, 12U, 16U), PadStrideInfo(3, 2, 1, 0) },
        ConvolutionLayerDataObject{ TensorShape(17U, 31U, 2U, 7U), TensorShape(5U, 5U, 2U, 19U), TensorShape(19U), TensorShape(15U, 15U, 19U, 7U), PadStrideInfo(1, 2, 1, 1) },
        ConvolutionLayerDataObject{ TensorShape(23U, 27U, 5U), TensorShape(3U, 1U, 5U, 21U), TensorShape(21U), TensorShape(11U, 27U, 21U), PadStrideInfo(2, 1, 0, 0) },
        ConvolutionLayerDataObject{ TensorShape(33U, 27U, 7U), TensorShape(5U, 7U, 7U, 16U), TensorShape(16U), TensorShape(11U, 11U, 16U), PadStrideInfo(3, 2, 1, 0) },
        ConvolutionLayerDataObject{ TensorShape(17U, 31U, 2U, 7U), TensorShape(5U, 3U, 2U, 19U), TensorShape(19U), TensorShape(15U, 16U, 19U, 7U), PadStrideInfo(1, 2, 1, 1) }
    }
    {
    }

    ~SmallConvolutionLayerDataset() = default;
};

/** Data set containing direct convolution tensor shapes. */
class DirectConvolutionShapes final : public ShapeDataset<3>
{
public:
    DirectConvolutionShapes()
        : ShapeDataset(TensorShape(3U, 3U, 3U, 2U, 4U, 5U),
                       TensorShape(32U, 37U, 3U),
                       TensorShape(13U, 15U, 8U, 3U))
    {
    }
};

/** AlexNet's convolution layers tensor shapes. */
class AlexNetConvolutionLayerDataset final : public ConvolutionLayerDataset<5>
{
public:
    AlexNetConvolutionLayerDataset()
        : GenericDataset
    {
        ConvolutionLayerDataObject{ TensorShape(227U, 227U, 3U), TensorShape(11U, 11U, 3U, 96U), TensorShape(96U), TensorShape(55U, 55U, 96U), PadStrideInfo(4, 4, 0, 0) },
        ConvolutionLayerDataObject{ TensorShape(27U, 27U, 96U), TensorShape(5U, 5U, 96U, 256U), TensorShape(256U), TensorShape(27U, 27U, 256U), PadStrideInfo(1, 1, 2, 2) },
        ConvolutionLayerDataObject{ TensorShape(13U, 13U, 256U), TensorShape(3U, 3U, 256U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 1, 1) },
        ConvolutionLayerDataObject{ TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 384U), TensorShape(384U), TensorShape(13U, 13U, 384U), PadStrideInfo(1, 1, 1, 1) },
        ConvolutionLayerDataObject{ TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 256U), TensorShape(256U), TensorShape(13U, 13U, 256U), PadStrideInfo(1, 1, 1, 1) }
    }
    {
    }

    ~AlexNetConvolutionLayerDataset() = default;
};

/** LeNet5's convolution layers tensor shapes. */
class LeNet5ConvolutionLayerDataset final : public ConvolutionLayerDataset<2>
{
public:
    LeNet5ConvolutionLayerDataset()
        : GenericDataset
    {
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 1U), TensorShape(5U, 5U, 1U, 20U), TensorShape(20U), TensorShape(24U, 24U, 20U), PadStrideInfo(1, 1, 0, 0) },
        ConvolutionLayerDataObject{ TensorShape(12U, 12U, 20U), TensorShape(5U, 5U, 20U, 50U), TensorShape(50U), TensorShape(8U, 8U, 50U), PadStrideInfo(1, 1, 0, 0) },
    }
    {
    }

    ~LeNet5ConvolutionLayerDataset() = default;
};

/** GoogleLeNet v1 convolution layers tensor shapes (Part 1).
 *
 * @note Dataset is split into two to avoid a register allocation failure produced by clang in Android debug builds.
 */
class GoogLeNetConvolutionLayerDataset1 final : public ConvolutionLayerDataset<32>
{
public:
    GoogLeNetConvolutionLayerDataset1()
        : GenericDataset
    {
        // conv1/7x7_s2
        ConvolutionLayerDataObject{ TensorShape(224U, 224U, 3U), TensorShape(7U, 7U, 3U, 64U), TensorShape(64U), TensorShape(112U, 112U, 64U), PadStrideInfo(2, 2, 3, 3) },
        // conv2/3x3_reduce
        ConvolutionLayerDataObject{ TensorShape(56U, 56U, 64U), TensorShape(1U, 1U, 64U, 64U), TensorShape(64U), TensorShape(56U, 56U, 64U), PadStrideInfo(1, 1, 0, 0) },
        // conv2/3x3
        ConvolutionLayerDataObject{ TensorShape(56U, 56U, 64U), TensorShape(3U, 3U, 64U, 192U), TensorShape(192U), TensorShape(56U, 56U, 192U), PadStrideInfo(1, 1, 1, 1) },
        // inception_3a/1x1
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 64U), TensorShape(64U), TensorShape(28U, 28U, 64U), PadStrideInfo(1, 1, 0, 0) },
        // inception_3a/3x3_reduce
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 96U), TensorShape(96U), TensorShape(28U, 28U, 96U), PadStrideInfo(1, 1, 0, 0) },
        // inception_3a/3x3
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 96U), TensorShape(3U, 3U, 96U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 1, 1) },
        // inception_3a/5x5_reduce
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 16U), TensorShape(16U), TensorShape(28U, 28U, 16U), PadStrideInfo(1, 1, 0, 0) },
        // inception_3a/5x5
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 16U), TensorShape(5U, 5U, 16U, 32U), TensorShape(32U), TensorShape(28U, 28U, 32U), PadStrideInfo(1, 1, 2, 2) },
        // inception_3a/pool_proj
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 32U), TensorShape(32U), TensorShape(28U, 28U, 32U), PadStrideInfo(1, 1, 0, 0) },
        // inception_3b/1x1, inception_3b/3x3_reduce
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 0, 0) },
        // inception_3b/3x3
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 128U), TensorShape(3U, 3U, 128U, 192U), TensorShape(192U), TensorShape(28U, 28U, 192U), PadStrideInfo(1, 1, 1, 1) },
        // inception_3b/5x5_reduce
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 32U), TensorShape(32U), TensorShape(28U, 28U, 32U), PadStrideInfo(1, 1, 0, 0) },
        // inception_3b/5x5
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 32U), TensorShape(5U, 5U, 32U, 96U), TensorShape(96U), TensorShape(28U, 28U, 96U), PadStrideInfo(1, 1, 2, 2) },
        // inception_3b/pool_proj
        ConvolutionLayerDataObject{ TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 64U), TensorShape(64U), TensorShape(28U, 28U, 64U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4a/1x1
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 192U), TensorShape(192U), TensorShape(14U, 14U, 192U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4a/3x3_reduce
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 96U), TensorShape(96U), TensorShape(14U, 14U, 96U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4a/3x3
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 96U), TensorShape(3U, 3U, 96U, 208U), TensorShape(208U), TensorShape(14U, 14U, 208U), PadStrideInfo(1, 1, 1, 1) },
        // inception_4a/5x5_reduce
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 16U), TensorShape(16U), TensorShape(14U, 14U, 16U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4a/5x5
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 16U), TensorShape(5U, 5U, 16U, 48U), TensorShape(48U), TensorShape(14U, 14U, 48U), PadStrideInfo(1, 1, 2, 2) },
        // inception_4a/pool_proj
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4b/1x1
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 160U), TensorShape(160U), TensorShape(14U, 14U, 160U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4b/3x3_reduce, inception_4d/1x1
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 112U), TensorShape(112U), TensorShape(14U, 14U, 112U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4b/3x3
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 112U), TensorShape(3U, 3U, 112U, 224U), TensorShape(224U), TensorShape(14U, 14U, 224U), PadStrideInfo(1, 1, 1, 1) },
        // inception_4b/5x5_reduce, inception_4c/5x5_reduce
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 24U), TensorShape(24U), TensorShape(14U, 14U, 24U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4b/5x5, inception_4c/5x5
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 24U), TensorShape(5U, 5U, 24U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 2, 2) },
        // inception_4b/pool_proj, inception_4c/pool_proj, inception_4d/pool_proj
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4c/1x1, inception_4c/3x3_reduce
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 128U), TensorShape(128U), TensorShape(14U, 14U, 128U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4c/3x3
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 128U), TensorShape(3U, 3U, 128U, 256U), TensorShape(256U), TensorShape(14U, 14U, 256U), PadStrideInfo(1, 1, 1, 1) },
        // inception_4d/3x3_reduce
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 144U), TensorShape(144U), TensorShape(14U, 14U, 144U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4d/3x3
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 144U), TensorShape(3U, 3U, 144U, 288U), TensorShape(288U), TensorShape(14U, 14U, 288U), PadStrideInfo(1, 1, 1, 1) },
        // inception_4d/5x5_reduce
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 32U), TensorShape(32U), TensorShape(14U, 14U, 32U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4d/5x5
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 32U), TensorShape(5U, 5U, 32U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 2, 2) },
    }
    {
    }

    ~GoogLeNetConvolutionLayerDataset1() = default;
};

/** GoogleLeNet v1 convolution layers tensor shapes (Part 2). */
class GoogLeNetConvolutionLayerDataset2 final : public ConvolutionLayerDataset<17>
{
public:
    GoogLeNetConvolutionLayerDataset2()
        : GenericDataset
    {
        // inception_4e/1x1
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 256U), TensorShape(256U), TensorShape(14U, 14U, 256U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4e/3x3_reduce
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 160U), TensorShape(160U), TensorShape(14U, 14U, 160U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4e/3x3
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 160U), TensorShape(3U, 3U, 160U, 320U), TensorShape(320U), TensorShape(14U, 14U, 320U), PadStrideInfo(1, 1, 1, 1) },
        // inception_4e/5x5_reduce
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 32U), TensorShape(32U), TensorShape(14U, 14U, 32U), PadStrideInfo(1, 1, 0, 0) },
        // inception_4e/5x5
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 32U), TensorShape(5U, 5U, 32U, 128U), TensorShape(128U), TensorShape(14U, 14U, 128U), PadStrideInfo(1, 1, 2, 2) },
        // inception_4e/pool_proj
        ConvolutionLayerDataObject{ TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 128U), TensorShape(128U), TensorShape(14U, 14U, 128U), PadStrideInfo(1, 1, 0, 0) },
        // inception_5a/1x1
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 256U), TensorShape(256U), TensorShape(7U, 7U, 256U), PadStrideInfo(1, 1, 0, 0) },
        // inception_5a/3x3_reduce
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 160U), TensorShape(160U), TensorShape(7U, 7U, 160U), PadStrideInfo(1, 1, 0, 0) },
        // inception_5a/3x3
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 160U), TensorShape(3U, 3U, 160U, 320U), TensorShape(320U), TensorShape(7U, 7U, 320U), PadStrideInfo(1, 1, 1, 1) },
        // inception_5a/5x5_reduce
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 32U), TensorShape(32U), TensorShape(7U, 7U, 32U), PadStrideInfo(1, 1, 0, 0) },
        // inception_5a/5x5
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 32U), TensorShape(5U, 5U, 32U, 128U), TensorShape(128U), TensorShape(7U, 7U, 128U), PadStrideInfo(1, 1, 2, 2) },
        // inception_5a/pool_proj, inception_5b/pool_proj
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 128U), TensorShape(128U), TensorShape(7U, 7U, 128U), PadStrideInfo(1, 1, 0, 0) },
        // inception_5b/1x1
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 384U), TensorShape(384U), TensorShape(7U, 7U, 384U), PadStrideInfo(1, 1, 0, 0) },
        // inception_5b/3x3_reduce
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 192U), TensorShape(192U), TensorShape(7U, 7U, 192U), PadStrideInfo(1, 1, 0, 0) },
        // inception_5b/3x3
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 192U), TensorShape(3U, 3U, 192U, 384U), TensorShape(384U), TensorShape(7U, 7U, 384U), PadStrideInfo(1, 1, 1, 1) },
        // inception_5b/5x5_reduce
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 48U), TensorShape(48U), TensorShape(7U, 7U, 48U), PadStrideInfo(1, 1, 0, 0) },
        // inception_5b/5x5
        ConvolutionLayerDataObject{ TensorShape(7U, 7U, 48U), TensorShape(5U, 5U, 48U, 128U), TensorShape(128U), TensorShape(7U, 7U, 128U), PadStrideInfo(1, 1, 2, 2) }
    }
    {
    }

    ~GoogLeNetConvolutionLayerDataset2() = default;
};
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_DATASET_CONVOLUTION_LAYER_DATASET_H__
