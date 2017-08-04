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
#ifndef ARM_COMPUTE_TEST_SQUEEZENET_ACTIVATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_SQUEEZENET_ACTIVATION_LAYER_DATASET

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
class SqueezeNetActivationLayerDataset final : public
    framework::dataset::CartesianProductDataset<framework::dataset::InitializerListDataset<TensorShape>, framework::dataset::SingletonDataset<ActivationLayerInfo>>
{
public:
    SqueezeNetActivationLayerDataset()
        : CartesianProductDataset
    {
        framework::dataset::make("Shape", { // relu_conv1
            TensorShape(111U, 111U, 64U),
            // fire2/relu_squeeze1x1, fire3/relu_squeeze1x1
            TensorShape(55U, 55U, 16U),
            // fire2/relu_expand1x1, fire2/relu_expand3x3, fire3/relu_expand1x1, fire3/relu_expand3x3
            TensorShape(55U, 55U, 64U),
            // fire4/relu_squeeze1x1, fire5/relu_squeeze1x1
            TensorShape(27U, 27U, 32U),
            // fire4/relu_expand1x1, fire4/relu_expand3x3, fire5/relu_expand1x1, fire5/relu_expand3x3
            TensorShape(27U, 27U, 128U),
            // fire6/relu_squeeze1x1, fire7/relu_squeeze1x1
            TensorShape(13U, 13U, 48U),
            // fire6/relu_expand1x1, fire6/relu_expand3x3, fire7/relu_expand1x1, fire7/relu_expand3x3
            TensorShape(13U, 13U, 192U),
            // fire8/relu_squeeze1x1, fire9/relu_squeeze1x1
            TensorShape(13U, 13U, 64U),
            // fire8/relu_expand1x1, fire8/relu_expand3x3, fire9/relu_expand1x1, fire9/relu_expand3x3
            TensorShape(13U, 13U, 256U),
            // relu_conv10
            TensorShape(13U, 13U, 1000U) }),
        framework::dataset::make("Info", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
    }
    {
    }
    SqueezeNetActivationLayerDataset(SqueezeNetActivationLayerDataset &&) = default;
    ~SqueezeNetActivationLayerDataset()                                   = default;
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SQUEEZENET_ACTIVATION_LAYER_DATASET */
