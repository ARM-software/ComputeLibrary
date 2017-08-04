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
#ifndef ARM_COMPUTE_TEST_VGG16_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_VGG16_CONVOLUTION_LAYER_DATASET

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
class VGG16ConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    VGG16ConvolutionLayerDataset()
    {
        // conv1_1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(3U, 3U, 3U, 64U), TensorShape(64U), TensorShape(224U, 224U, 64U), PadStrideInfo(1, 1, 1, 1));
        // conv1_2
        add_config(TensorShape(224U, 224U, 64U), TensorShape(3U, 3U, 64U, 64U), TensorShape(64U), TensorShape(224U, 224U, 64U), PadStrideInfo(1, 1, 1, 1));
        // conv2_1
        add_config(TensorShape(112U, 112U, 64U), TensorShape(3U, 3U, 64U, 128U), TensorShape(128U), TensorShape(112U, 112U, 128U), PadStrideInfo(1, 1, 1, 1));
        // conv2_2
        add_config(TensorShape(112U, 112U, 128U), TensorShape(3U, 3U, 128U, 128U), TensorShape(128U), TensorShape(112U, 112U, 128U), PadStrideInfo(1, 1, 1, 1));
        // conv3_1
        add_config(TensorShape(56U, 56U, 128U), TensorShape(3U, 3U, 128U, 256U), TensorShape(256U), TensorShape(56U, 56U, 256U), PadStrideInfo(1, 1, 1, 1));
        // conv3_2, conv3_3
        add_config(TensorShape(56U, 56U, 256U), TensorShape(3U, 3U, 256U, 256U), TensorShape(256U), TensorShape(56U, 56U, 256U), PadStrideInfo(1, 1, 1, 1));
        // conv4_1
        add_config(TensorShape(28U, 28U, 256U), TensorShape(3U, 3U, 256U, 512U), TensorShape(512U), TensorShape(28U, 28U, 512U), PadStrideInfo(1, 1, 1, 1));
        // conv4_2, conv4_3
        add_config(TensorShape(28U, 28U, 512U), TensorShape(3U, 3U, 512U, 512U), TensorShape(512U), TensorShape(28U, 28U, 512U), PadStrideInfo(1, 1, 1, 1));
        // conv5_1, conv5_2, conv5_3
        add_config(TensorShape(14U, 14U, 512U), TensorShape(3U, 3U, 512U, 512U), TensorShape(512U), TensorShape(14U, 14U, 512U), PadStrideInfo(1, 1, 1, 1));
    }
};

class VGG16DirectConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    VGG16DirectConvolutionLayerDataset()
    {
        // conv1_1
        add_config(TensorShape(224U, 224U, 3U), TensorShape(3U, 3U, 3U, 64U), TensorShape(64U), TensorShape(224U, 224U, 64U), PadStrideInfo(1, 1, 1, 1));
        // conv1_2
        add_config(TensorShape(224U, 224U, 64U), TensorShape(3U, 3U, 64U, 64U), TensorShape(64U), TensorShape(224U, 224U, 64U), PadStrideInfo(1, 1, 1, 1));
        // conv2_1
        add_config(TensorShape(112U, 112U, 64U), TensorShape(3U, 3U, 64U, 128U), TensorShape(128U), TensorShape(112U, 112U, 128U), PadStrideInfo(1, 1, 1, 1));
        // conv2_2
        add_config(TensorShape(112U, 112U, 128U), TensorShape(3U, 3U, 128U, 128U), TensorShape(128U), TensorShape(112U, 112U, 128U), PadStrideInfo(1, 1, 1, 1));
        // conv3_1
        add_config(TensorShape(56U, 56U, 128U), TensorShape(3U, 3U, 128U, 256U), TensorShape(256U), TensorShape(56U, 56U, 256U), PadStrideInfo(1, 1, 1, 1));
        // conv3_2, conv3_3
        add_config(TensorShape(56U, 56U, 256U), TensorShape(3U, 3U, 256U, 256U), TensorShape(256U), TensorShape(56U, 56U, 256U), PadStrideInfo(1, 1, 1, 1));
        // conv4_1
        add_config(TensorShape(28U, 28U, 256U), TensorShape(3U, 3U, 256U, 512U), TensorShape(512U), TensorShape(28U, 28U, 512U), PadStrideInfo(1, 1, 1, 1));
        // conv4_2, conv4_3
        add_config(TensorShape(28U, 28U, 512U), TensorShape(3U, 3U, 512U, 512U), TensorShape(512U), TensorShape(28U, 28U, 512U), PadStrideInfo(1, 1, 1, 1));
        // conv5_1, conv5_2, conv5_3
        add_config(TensorShape(14U, 14U, 512U), TensorShape(3U, 3U, 512U, 512U), TensorShape(512U), TensorShape(14U, 14U, 512U), PadStrideInfo(1, 1, 1, 1));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_VGG16_CONVOLUTION_LAYER_DATASET */
