/*
 * Copyright (c) 2017-2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_MOBILENET_ACTIVATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_MOBILENET_ACTIVATION_LAYER_DATASET

#include "tests/framework/datasets/Datasets.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class MobileNetActivationLayerDataset final : public
    framework::dataset::CartesianProductDataset<framework::dataset::InitializerListDataset<TensorShape>, framework::dataset::SingletonDataset<ActivationLayerInfo>>
{
public:
    MobileNetActivationLayerDataset()
        : CartesianProductDataset
    {
        framework::dataset::make("Shape", {
            TensorShape(112U, 112U, 32U), TensorShape(112U, 112U, 64U), TensorShape(56U, 56U, 64U), TensorShape(56U, 56U, 128U),
            TensorShape(28U, 28U, 128U), TensorShape(28U, 28U, 256U), TensorShape(14U, 14U, 256U), TensorShape(14U, 14U, 512U),
            TensorShape(7U, 7U, 512U), TensorShape(7U, 7U, 1024U) }),
        framework::dataset::make("Info", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f))
    }
    {
    }
    MobileNetActivationLayerDataset(MobileNetActivationLayerDataset &&) = default;
    ~MobileNetActivationLayerDataset()                                  = default;
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MOBILENET_ACTIVATION_LAYER_DATASET */
