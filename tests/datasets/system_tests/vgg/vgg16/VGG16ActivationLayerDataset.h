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
#ifndef ARM_COMPUTE_TEST_VGG16_ACTIVATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_VGG16_ACTIVATION_LAYER_DATASET

#include "tests/framework/datasets/Datasets.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class VGG16ActivationLayerDataset final : public
    framework::dataset::CartesianProductDataset<framework::dataset::InitializerListDataset<TensorShape>, framework::dataset::SingletonDataset<ActivationLayerInfo>>
{
public:
    VGG16ActivationLayerDataset()
        : CartesianProductDataset
    {
        framework::dataset::make("Shape", { // relu1_1, relu1_2
            TensorShape(224U, 224U, 64U),
            // relu2_1, relu2_2
            TensorShape(112U, 112U, 128U),
            // relu3_1, relu3_2, relu3_3
            TensorShape(56U, 56U, 256U),
            // relu4_1, relu4_2, relu4_3
            TensorShape(28U, 28U, 512U),
            // relu5_1, relu5_2, relu5_3
            TensorShape(14U, 14U, 512U),
            // relu6, relu7
            TensorShape(4096U) }),
        framework::dataset::make("Info", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
    }
    {
    }
    VGG16ActivationLayerDataset(VGG16ActivationLayerDataset &&) = default;
    ~VGG16ActivationLayerDataset()                              = default;
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_VGG16_ACTIVATION_LAYER_DATASET */
