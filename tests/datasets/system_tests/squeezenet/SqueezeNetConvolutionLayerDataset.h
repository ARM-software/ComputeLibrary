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
#ifndef ARM_COMPUTE_TEST_SQUEEZENET_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_SQUEEZENET_CONVOLUTION_LAYER_DATASET

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
class SqueezeNetWinogradLayerDataset final : public ConvolutionLayerDataset
{
public:
    SqueezeNetWinogradLayerDataset()
    {
        // fire2/expand3x3, fire3/expand3x3
        add_config(TensorShape(55U, 55U, 16U), TensorShape(3U, 3U, 16U, 64U), TensorShape(64U), TensorShape(55U, 55U, 64U), PadStrideInfo(1, 1, 1, 1));
        // fire4/expand3x3, fire5/expand3x3
        add_config(TensorShape(27U, 27U, 32U), TensorShape(3U, 3U, 32U, 128U), TensorShape(128U), TensorShape(27U, 27U, 128U), PadStrideInfo(1, 1, 1, 1));
        // fire6/expand3x3, fire7/expand3x3
        add_config(TensorShape(13U, 13U, 48U), TensorShape(3U, 3U, 48U, 192U), TensorShape(192U), TensorShape(13U, 13U, 192U), PadStrideInfo(1, 1, 1, 1));
        // fire8/expand3x3, fire9/expand3x3
        add_config(TensorShape(13U, 13U, 64U), TensorShape(3U, 3U, 64U, 256U), TensorShape(256U), TensorShape(13U, 13U, 256U), PadStrideInfo(1, 1, 1, 1));
    }
};

class SqueezeNetConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    SqueezeNetConvolutionLayerDataset()
    {
        // conv1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(3U, 3U, 3U, 64U), TensorShape(64U), TensorShape(111U, 111U, 64U), PadStrideInfo(2, 2, 0, 0));
        // fire2/squeeze1x1
        add_config(TensorShape(55U, 55U, 64U), TensorShape(1U, 1U, 64U, 16U), TensorShape(16U), TensorShape(55U, 55U, 16U), PadStrideInfo(1, 1, 0, 0));
        // fire2/expand1x1, fire3/expand1x1
        add_config(TensorShape(55U, 55U, 16U), TensorShape(1U, 1U, 16U, 64U), TensorShape(64U), TensorShape(55U, 55U, 64U), PadStrideInfo(1, 1, 0, 0));
        // fire2/expand3x3, fire3/expand3x3
        add_config(TensorShape(55U, 55U, 16U), TensorShape(3U, 3U, 16U, 64U), TensorShape(64U), TensorShape(55U, 55U, 64U), PadStrideInfo(1, 1, 1, 1));
        // fire3/squeeze1x1
        add_config(TensorShape(55U, 55U, 128U), TensorShape(1U, 1U, 128U, 16U), TensorShape(16U), TensorShape(55U, 55U, 16U), PadStrideInfo(1, 1, 0, 0));
        // fire4/squeeze1x1
        add_config(TensorShape(27U, 27U, 128U), TensorShape(1U, 1U, 128U, 32U), TensorShape(32U), TensorShape(27U, 27U, 32U), PadStrideInfo(1, 1, 0, 0));
        // fire4/expand1x1, fire5/expand1x1
        add_config(TensorShape(27U, 27U, 32U), TensorShape(1U, 1U, 32U, 128U), TensorShape(128U), TensorShape(27U, 27U, 128U), PadStrideInfo(1, 1, 0, 0));
        // fire4/expand3x3, fire5/expand3x3
        add_config(TensorShape(27U, 27U, 32U), TensorShape(3U, 3U, 32U, 128U), TensorShape(128U), TensorShape(27U, 27U, 128U), PadStrideInfo(1, 1, 1, 1));
        // fire5/squeeze1x1
        add_config(TensorShape(27U, 27U, 256U), TensorShape(1U, 1U, 256U, 32U), TensorShape(32U), TensorShape(27U, 27U, 32U), PadStrideInfo(1, 1, 0, 0));
        // fire6/squeeze1x1
        add_config(TensorShape(13U, 13U, 256U), TensorShape(1U, 1U, 256U, 48U), TensorShape(48U), TensorShape(13U, 13U, 48U), PadStrideInfo(1, 1, 0, 0));
        // fire6/expand1x1, fire7/expand1x1
        add_config(TensorShape(13U, 13U, 48U), TensorShape(1U, 1U, 48U, 192U), TensorShape(192U), TensorShape(13U, 13U, 192U), PadStrideInfo(1, 1, 0, 0));
        // fire6/expand3x3, fire7/expand3x3
        add_config(TensorShape(13U, 13U, 48U), TensorShape(3U, 3U, 48U, 192U), TensorShape(192U), TensorShape(13U, 13U, 192U), PadStrideInfo(1, 1, 1, 1));
        // fire7/squeeze1x1
        add_config(TensorShape(13U, 13U, 384U), TensorShape(1U, 1U, 384U, 48U), TensorShape(48U), TensorShape(13U, 13U, 48U), PadStrideInfo(1, 1, 0, 0));
        // fire8/squeeze1x1
        add_config(TensorShape(13U, 13U, 384U), TensorShape(1U, 1U, 384U, 64U), TensorShape(64U), TensorShape(13U, 13U, 64U), PadStrideInfo(1, 1, 0, 0));
        // fire8/expand1x1, fire9/expand1x1
        add_config(TensorShape(13U, 13U, 64U), TensorShape(1U, 1U, 64U, 256U), TensorShape(256U), TensorShape(13U, 13U, 256U), PadStrideInfo(1, 1, 0, 0));
        // fire8/expand3x3, fire9/expand3x3
        add_config(TensorShape(13U, 13U, 64U), TensorShape(3U, 3U, 64U, 256U), TensorShape(256U), TensorShape(13U, 13U, 256U), PadStrideInfo(1, 1, 1, 1));
        // fire9/squeeze1x1
        add_config(TensorShape(13U, 13U, 512U), TensorShape(1U, 1U, 512U, 64U), TensorShape(64U), TensorShape(13U, 13U, 64U), PadStrideInfo(1, 1, 0, 0));
        // conv10
        add_config(TensorShape(13U, 13U, 512U), TensorShape(1U, 1U, 512U, 1000U), TensorShape(1000U), TensorShape(13U, 13U, 1000U), PadStrideInfo(1, 1, 0, 0));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SQUEEZENET_CONVOLUTION_LAYER_DATASET */
