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
#ifndef ARM_COMPUTE_TEST_MOBILENET_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_MOBILENET_CONVOLUTION_LAYER_DATASET

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
class MobileNetConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    MobileNetConvolutionLayerDataset()
    {
        add_config(TensorShape(10U, 10U, 384U), TensorShape(1U, 1U, 384U, 384U), TensorShape(384U), TensorShape(10U, 10U, 384U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(5U, 5U, 768U), TensorShape(1U, 1U, 768U, 768U), TensorShape(768U), TensorShape(5U, 5U, 768U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(40U, 40U, 96U), TensorShape(1U, 1U, 96U, 96U), TensorShape(96U), TensorShape(40U, 40U, 96U), PadStrideInfo(1, 1, 0, 0));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MOBILENET_CONVOLUTION_LAYER_DATASET */
