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
#ifndef ARM_COMPUTE_TEST_SMALL_DEPTHWISE_CONVOLUTION_DATASET
#define ARM_COMPUTE_TEST_SMALL_DEPTHWISE_CONVOLUTION_DATASET

#include "tests/datasets_new/DepthwiseConvolutionDataset.h"

#include "tests/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class SmallDepthwiseConvolutionDataset final : public DepthwiseConvolutionDataset
{
public:
    SmallDepthwiseConvolutionDataset()
    {
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 3U, 5U), TensorShape(11U, 25U, 5U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(3U, 3U, 7U), TensorShape(11U, 13U, 7U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(3U, 3U, 2U), TensorShape(17U, 16U, 2U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 3U, 5U), TensorShape(21U, 13U, 5U), PadStrideInfo(1, 2, 0, 0));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(3U, 3U, 7U), TensorShape(16U, 9U, 7U), PadStrideInfo(2, 3, 0, 1));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(3U, 3U, 2U), TensorShape(9U, 31U, 2U), PadStrideInfo(2, 1, 1, 1));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SMALL_DEPTHWISE_CONVOLUTION_DATASET */
