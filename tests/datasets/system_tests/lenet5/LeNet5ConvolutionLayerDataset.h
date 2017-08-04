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
#ifndef ARM_COMPUTE_TEST_LENET5_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_LENET5_CONVOLUTION_LAYER_DATASET

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
class LeNet5ConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    LeNet5ConvolutionLayerDataset()
    {
        add_config(TensorShape(28U, 28U, 1U), TensorShape(5U, 5U, 1U, 20U), TensorShape(20U), TensorShape(24U, 24U, 20U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(12U, 12U, 20U), TensorShape(5U, 5U, 20U, 50U), TensorShape(50U), TensorShape(8U, 8U, 50U), PadStrideInfo(1, 1, 0, 0));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LENET5_CONVOLUTION_LAYER_DATASET */
