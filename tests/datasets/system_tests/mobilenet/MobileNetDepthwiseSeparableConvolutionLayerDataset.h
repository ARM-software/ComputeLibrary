/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_MOBILENET_DEPTHWISE_SEPARABLE_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_MOBILENET_DEPTHWISE_SEPARABLE_CONVOLUTION_LAYER_DATASET

#include "tests/datasets/DepthwiseSeparableConvolutionLayerDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class MobileNetDepthwiseSeparableConvolutionLayerDataset final : public DepthwiseSeparableConvolutionLayerDataset
{
public:
    MobileNetDepthwiseSeparableConvolutionLayerDataset()
    {
        add_config(TensorShape(14U, 14U, 512U), TensorShape(3U, 3U, 512U), TensorShape(512U), TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 512U), TensorShape(512U), TensorShape(14U, 14U, 512U),
                   PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::FLOOR));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MOBILENET_DEPTHWISE_SEPARABLE_CONVOLUTION_LAYER_DATASET */
