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
#ifndef ARM_COMPUTE_TEST_YOLOV2_BATCHNORMALIZATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_YOLOV2_BATCHNORMALIZATION_LAYER_DATASET

#include "tests/datasets/BatchNormalizationLayerDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class YOLOV2BatchNormalizationLayerDataset final : public BatchNormalizationLayerDataset
{
public:
    YOLOV2BatchNormalizationLayerDataset()
    {
        // conv1_bn
        add_config(TensorShape(416U, 416U, 32U), TensorShape(32U), 0.00001f);
        // conv2_bn
        add_config(TensorShape(208U, 208U, 64U), TensorShape(64U), 0.00001f);
        // conv3_bn, conv5_bn
        add_config(TensorShape(104U, 104U, 128U), TensorShape(128U), 0.00001f);
        // conv4_bn
        add_config(TensorShape(104U, 104U, 64U), TensorShape(64U), 0.00001f);
        // conv6_bn, conv8_bn
        add_config(TensorShape(52U, 52U, 256U), TensorShape(256U), 0.00001f);
        // conv7_bn
        add_config(TensorShape(52U, 52U, 128U), TensorShape(128U), 0.00001f);
        // conv9_bn, conv11_bn, conv13_bn
        add_config(TensorShape(26U, 26U, 512U), TensorShape(512U), 0.00001f);
        // conv10_bn, conv12_bn
        add_config(TensorShape(26U, 26U, 256U), TensorShape(256U), 0.00001f);
        // conv14_bn, conv16_bn, conv18_bn, conv19_bn, conv20_bn, conv21_bn
        add_config(TensorShape(13U, 13U, 1024U), TensorShape(1024U), 0.00001f);
        // conv15_bn, conv17_bn
        add_config(TensorShape(13U, 13U, 512U), TensorShape(512U), 0.00001f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_YOLOV2_BATCHNORMALIZATION_LAYER_DATASET */
