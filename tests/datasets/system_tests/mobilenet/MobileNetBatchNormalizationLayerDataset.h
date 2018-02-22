/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_MOBILENET_BATCHNORMALIZATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_MOBILENET_BATCHNORMALIZATION_LAYER_DATASET

#include "tests/datasets/BatchNormalizationLayerDataset.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class MobileNetBatchNormalizationLayerDataset final : public BatchNormalizationLayerDataset
{
public:
    MobileNetBatchNormalizationLayerDataset()
    {
        // conv1_bn, dwc0_bn
        add_config(TensorShape(112U, 112U, 32U), TensorShape(32U), 0.001f);
        // pwc0_bn
        add_config(TensorShape(112U, 112U, 64U), TensorShape(64U), 0.001f);
        // dwc1_bn
        add_config(TensorShape(56U, 56U, 64U), TensorShape(64U), 0.001f);
        // dwc2_bn, pwc1_bn, pwc2_bn
        add_config(TensorShape(56U, 56U, 128U), TensorShape(128U), 0.001f);
        // dwc3_bn
        add_config(TensorShape(28U, 28U, 128U), TensorShape(128U), 0.001f);
        // dwc4_bn, pwc3_bn, pwc4_bn
        add_config(TensorShape(28U, 28U, 256U), TensorShape(256U), 0.001f);
        // dwc5_bn
        add_config(TensorShape(14U, 14U, 256U), TensorShape(256U), 0.001f);
        // dwc6_bn, dwc7_bn, dwc8_bn, dwc9_bn, dwc10_bn, pwc5_bn, pwc6_bn, pwc7_bn, pwc8_bn, pwc9_bn, pwc10_bn
        add_config(TensorShape(14U, 14U, 512U), TensorShape(512U), 0.001f);
        // dwc11_bn
        add_config(TensorShape(7U, 7U, 512U), TensorShape(512U), 0.001f);
        // dwc12_bn, pwc11_bn, pwc12_bn
        add_config(TensorShape(7U, 7U, 1024U), TensorShape(1024U), 0.001f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MOBILENET_BATCHNORMALIZATION_LAYER_DATASET */
