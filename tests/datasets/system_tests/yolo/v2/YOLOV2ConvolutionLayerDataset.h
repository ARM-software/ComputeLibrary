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
#ifndef ARM_COMPUTE_TEST_YOLOV2_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_YOLOV2_CONVOLUTION_LAYER_DATASET

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
class YOLOV2ConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    YOLOV2ConvolutionLayerDataset()
    {
        // conv1
        add_config(TensorShape(416U, 416U, 3U), TensorShape(3U, 3U, 3U, 32U), TensorShape(32U), TensorShape(416U, 416U, 32U), PadStrideInfo(1, 1, 1, 1));
        // conv2
        add_config(TensorShape(208U, 208U, 32U), TensorShape(3U, 3U, 32U, 64U), TensorShape(64U), TensorShape(208U, 208U, 64U), PadStrideInfo(1, 1, 1, 1));
        // conv3, conv5
        add_config(TensorShape(104U, 104U, 64U), TensorShape(3U, 3U, 64U, 128U), TensorShape(128U), TensorShape(104U, 104U, 128U), PadStrideInfo(1, 1, 1, 1));
        // conv4
        add_config(TensorShape(104U, 104U, 128U), TensorShape(1U, 1U, 128U, 64U), TensorShape(64U), TensorShape(104U, 104U, 64U), PadStrideInfo(1, 1, 0, 0));
        // conv6, conv8
        add_config(TensorShape(52U, 52U, 128U), TensorShape(3U, 3U, 128U, 256U), TensorShape(256U), TensorShape(52U, 52U, 256U), PadStrideInfo(1, 1, 1, 1));
        // conv7
        add_config(TensorShape(52U, 52U, 256U), TensorShape(1U, 1U, 256U, 128U), TensorShape(128U), TensorShape(52U, 52U, 128U), PadStrideInfo(1, 1, 0, 0));
        // conv9, conv11, conv13
        add_config(TensorShape(26U, 26U, 256U), TensorShape(3U, 3U, 256U, 512U), TensorShape(512U), TensorShape(26U, 26U, 512U), PadStrideInfo(1, 1, 1, 1));
        // conv10, conv12
        add_config(TensorShape(26U, 26U, 512U), TensorShape(1U, 1U, 512U, 256U), TensorShape(256U), TensorShape(26U, 26U, 256U), PadStrideInfo(1, 1, 0, 0));
        // conv14, conv16, conv18
        add_config(TensorShape(13U, 13U, 512U), TensorShape(3U, 3U, 512U, 1024U), TensorShape(1024U), TensorShape(13U, 13U, 1024U), PadStrideInfo(1, 1, 1, 1));
        // conv15, conv17
        add_config(TensorShape(13U, 13U, 1024U), TensorShape(1U, 1U, 1024U, 512U), TensorShape(512U), TensorShape(13U, 13U, 512U), PadStrideInfo(1, 1, 0, 0));
        // conv19, conv20
        add_config(TensorShape(13U, 13U, 1024U), TensorShape(3U, 3U, 1024U, 1024U), TensorShape(1024U), TensorShape(13U, 13U, 1024U), PadStrideInfo(1, 1, 1, 1));
        // conv21
        add_config(TensorShape(13U, 13U, 3072U), TensorShape(3U, 3U, 3072U, 1024U), TensorShape(1024U), TensorShape(13U, 13U, 1024U), PadStrideInfo(1, 1, 1, 1));
        // conv22
        add_config(TensorShape(13U, 13U, 1024U), TensorShape(1U, 1U, 1024U, 425U), TensorShape(425U), TensorShape(13U, 13U, 425U), PadStrideInfo(1, 1, 0, 0));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_YOLOV2_CONVOLUTION_LAYER_DATASET */
