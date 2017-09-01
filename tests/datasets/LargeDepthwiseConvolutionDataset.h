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
#ifndef ARM_COMPUTE_TEST_LARGE_DEPTHWISE_CONVOLUTION_DATASET
#define ARM_COMPUTE_TEST_LARGE_DEPTHWISE_CONVOLUTION_DATASET

#include "tests/datasets/DepthwiseConvolutionDataset.h"

#include "tests/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class LargeDepthwiseConvolutionDataset final : public DepthwiseConvolutionDataset
{
public:
    LargeDepthwiseConvolutionDataset()
    {
        add_config(TensorShape(233U, 277U, 55U), TensorShape(3U, 3U, 55U), TensorShape(116U, 275U, 55U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(333U, 277U, 77U), TensorShape(3U, 3U, 77U), TensorShape(111U, 138U, 77U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(177U, 311U, 22U), TensorShape(3U, 3U, 22U), TensorShape(177U, 156U, 22U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(233U, 277U, 55U), TensorShape(3U, 3U, 55U), TensorShape(231U, 138U, 55U), PadStrideInfo(1, 2, 0, 0));
        add_config(TensorShape(333U, 277U, 77U), TensorShape(3U, 3U, 77U), TensorShape(166U, 93U, 77U), PadStrideInfo(2, 3, 0, 1));
        add_config(TensorShape(177U, 311U, 22U), TensorShape(3U, 3U, 22U), TensorShape(89U, 311U, 22U), PadStrideInfo(2, 1, 1, 1));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LARGE_DEPTHWISE_CONVOLUTION_DATASET */
