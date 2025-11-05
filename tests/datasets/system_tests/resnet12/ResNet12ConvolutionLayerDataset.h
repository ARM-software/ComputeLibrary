/*
 * Copyright (c) 2019, 2025 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_SYSTEM_TESTS_RESNET12_RESNET12CONVOLUTIONLAYERDATASET_H
#define ACL_TESTS_DATASETS_SYSTEM_TESTS_RESNET12_RESNET12CONVOLUTIONLAYERDATASET_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include "tests/datasets/ConvolutionLayerDataset.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ResNet12FFTConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    ResNet12FFTConvolutionLayerDataset()
    {
        add_config(TensorShape(192U, 128U, 64U), TensorShape(9U, 9U, 64U, 3U), TensorShape(3U),
                   TensorShape(192U, 128U, 3U), PadStrideInfo(1, 1, 4, 4));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_DATASETS_SYSTEM_TESTS_RESNET12_RESNET12CONVOLUTIONLAYERDATASET_H
