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
#ifndef ARM_COMPUTE_TEST_DIRECT_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_DIRECT_CONVOLUTION_LAYER_DATASET

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
/** Stripped down version of AlexNet as not all kernel sizes and strides are supported. */
class DirectConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    DirectConvolutionLayerDataset()
    {
        add_config(TensorShape(13U, 13U, 256U), TensorShape(3U, 3U, 256U, 3U), TensorShape(3U), TensorShape(13U, 13U, 3U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 4U), TensorShape(4U), TensorShape(13U, 13U, 4U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 5U), TensorShape(5U), TensorShape(13U, 13U, 5U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(13U, 13U, 256U), TensorShape(3U, 3U, 256U, 3U), TensorShape(3U), TensorShape(7U, 7U, 3U), PadStrideInfo(2, 2, 1, 1));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 4U), TensorShape(4U), TensorShape(7U, 7U, 4U), PadStrideInfo(2, 2, 1, 1));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 5U), TensorShape(5U), TensorShape(7U, 7U, 5U), PadStrideInfo(2, 2, 1, 1));
        add_config(TensorShape(13U, 13U, 256U), TensorShape(3U, 3U, 256U, 3U), TensorShape(3U), TensorShape(12U, 12U, 3U), PadStrideInfo(1, 1, 0, 1, 0, 1, DimensionRoundingType::FLOOR));
        add_config(TensorShape(13U, 13U, 384U), TensorShape(3U, 3U, 384U, 5U), TensorShape(5U), TensorShape(12U, 12U, 5U), PadStrideInfo(1, 1, 1, 0, 1, 0, DimensionRoundingType::FLOOR));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DIRECT_CONVOLUTION_LAYER_DATASET */
