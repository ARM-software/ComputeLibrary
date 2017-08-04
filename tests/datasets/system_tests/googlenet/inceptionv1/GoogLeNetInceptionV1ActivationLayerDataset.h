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
#ifndef ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV1_ACTIVATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV1_ACTIVATION_LAYER_DATASET

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
class GoogLeNetInceptionV1ActivationLayerDataset final : public
    framework::dataset::CartesianProductDataset<framework::dataset::InitializerListDataset<TensorShape>, framework::dataset::SingletonDataset<ActivationLayerInfo>>
{
public:
    GoogLeNetInceptionV1ActivationLayerDataset()
        : CartesianProductDataset
    {
        framework::dataset::make("Shape", { // conv1/relu_7x7
            TensorShape(112U, 112U, 64U),
            // conv2/relu_3x3_reduce
            TensorShape(56U, 56U, 64U),
            // conv2/relu_3x3
            TensorShape(56U, 56U, 192U),
            // inception_3a/relu_1x1, inception_3b/relu_pool_proj
            TensorShape(28U, 28U, 64U),
            // inception_3a/relu_3x3_reduce, inception_3b/relu_5x5
            TensorShape(28U, 28U, 96U),
            // inception_3a/relu_3x3, inception_3b/relu_1x1, inception_3b/relu_3x3_reduce
            TensorShape(28U, 28U, 128U),
            // inception_3a/relu_5x5_reduce
            TensorShape(28U, 28U, 16U),
            // inception_3a/relu_5x5, inception_3a/relu_pool_proj, inception_3b/relu_5x5_reduce
            TensorShape(28U, 28U, 32U),
            // inception_3b/relu_3x3
            TensorShape(28U, 28U, 192U),
            // inception_4a/relu_1x1
            TensorShape(14U, 14U, 192U),
            // inception_4a/relu_3x3_reduce
            TensorShape(14U, 14U, 96U),
            // inception_4a/relu_3x3
            TensorShape(14U, 14U, 208U),
            // inception_4a/relu_5x5_reduce
            TensorShape(14U, 14U, 16U),
            // inception_4a/relu_5x5
            TensorShape(14U, 14U, 48U),
            // inception_4a/relu_pool_proj, inception_4b/relu_5x5, inception_4b/relu_pool_proj, inception_4c/relu_5x5, inception_4c/relu_pool_proj, inception_4d/relu_5x5, inception_4d/relu_pool_proj
            TensorShape(14U, 14U, 64U),
            // inception_4b/relu_1x1, inception_4e/relu_3x3_reduce
            TensorShape(14U, 14U, 160U),
            // inception_4b/relu_3x3_reduce, inception_4d/relu_1x1
            TensorShape(14U, 14U, 112U),
            // inception_4b/relu_3x3
            TensorShape(14U, 14U, 224U),
            // inception_4b/relu_5x5_reduce, inception_4c/relu_5x5_reduce
            TensorShape(14U, 14U, 24U),
            // inception_4c/relu_1x1, inception_4c/relu_3x3_reduce, inception_4e/relu_5x5, inception_4e/relu_pool_proj
            TensorShape(14U, 14U, 128U),
            // inception_4c/relu_3x3, inception_4e/relu_1x1
            TensorShape(14U, 14U, 256U),
            // inception_4d/relu_3x3_reduce
            TensorShape(14U, 14U, 144U),
            // inception_4d/relu_3x3
            TensorShape(14U, 14U, 288U),
            // inception_4d/relu_5x5_reduce, inception_4e/relu_5x5_reduce
            TensorShape(14U, 14U, 32U),
            // inception_4e/relu_3x3
            TensorShape(14U, 14U, 320U),
            // inception_5a/relu_1x1
            TensorShape(7U, 7U, 256U),
            // inception_5a/relu_3x3_reduce
            TensorShape(7U, 7U, 160U),
            // inception_5a/relu_3x3
            TensorShape(7U, 7U, 320U),
            // inception_5a/relu_5x5_reduce
            TensorShape(7U, 7U, 32U),
            // inception_5a/relu_5x5, inception_5a/relu_pool_proj, inception_5b/relu_5x5, inception_5b/relu_pool_proj
            TensorShape(7U, 7U, 128U),
            // inception_5b/relu_1x1, inception_5b/relu_3x3
            TensorShape(7U, 7U, 384U),
            // inception_5b/relu_3x3_reduce
            TensorShape(7U, 7U, 192U),
            // inception_5b/relu_5x5_reduce
            TensorShape(7U, 7U, 48U) }),
        framework::dataset::make("Info", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
    }
    {
    }
    GoogLeNetInceptionV1ActivationLayerDataset(GoogLeNetInceptionV1ActivationLayerDataset &&) = default;
    ~GoogLeNetInceptionV1ActivationLayerDataset()                                             = default;
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV1_ACTIVATION_LAYER_DATASET */
