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
#ifndef ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_ACTIVATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_ACTIVATION_LAYER_DATASET

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
class GoogLeNetInceptionV4ActivationLayerDataset final : public
    framework::dataset::CartesianProductDataset<framework::dataset::InitializerListDataset<TensorShape>, framework::dataset::SingletonDataset<ActivationLayerInfo>>
{
public:
    GoogLeNetInceptionV4ActivationLayerDataset()
        : CartesianProductDataset
    {
        framework::dataset::make("Shape", { // conv1_3x3_s2_relu
            TensorShape(149U, 149U, 32U),
            // conv2_3x3_s1_relu
            TensorShape(147U, 147U, 32U),
            // conv3_3x3_s1_relu
            TensorShape(147U, 147U, 64U),
            // inception_stem1_3x3_s2_relu
            TensorShape(73U, 73U, 96U),
            // inception_stem2_3x3_reduce_relu, inception_stem2_1x7_reduce_relu, inception_stem2_1x7_relu, inception_stem2_7x1_relu
            TensorShape(73U, 73U, 64U),
            // inception_stem2_3x3_relu, inception_stem2_3x3_2_relu
            TensorShape(71U, 71U, 96U),
            // inception_stem3_3x3_s2_relu, reduction_a_3x3_2_reduce_relu
            TensorShape(35U, 35U, 192U),
            // inception_a1_1x1_2_relu, inception_a1_3x3_relu, inception_a1_3x3_2_relu, inception_a1_3x3_3_relu, inception_a1_1x1_relu, inception_a2_1x1_2_relu, inception_a2_3x3_relu, inception_a2_3x3_2_relu, inception_a2_3x3_3_relu, inception_a2_1x1_relu, inception_a3_1x1_2_relu, inception_a3_3x3_relu, inception_a3_3x3_2_relu, inception_a3_3x3_3_relu, inception_a3_1x1_relu, inception_a4_1x1_2_relu, inception_a4_3x3_relu, inception_a4_3x3_2_relu, inception_a4_3x3_3_relu, inception_a4_1x1_relu
            TensorShape(35U, 35U, 96U),
            // inception_a1_3x3_reduce_relu, inception_a1_3x3_2_reduce_relu, inception_a2_3x3_reduce_relu, inception_a2_3x3_2_reduce_relu, inception_a3_3x3_reduce_relu, inception_a3_3x3_2_reduce_relu, inception_a4_3x3_reduce_relu, inception_a4_3x3_2_reduce_relu
            TensorShape(35U, 35U, 64U),
            // reduction_a_3x3_relu, inception_b1_1x1_2_relu, inception_b2_1x1_2_relu, inception_b3_1x1_2_relu, inception_b4_1x1_2_relu, inception_b5_1x1_2_relu, inception_b6_1x1_2_relu, inception_b7_1x1_2_relu
            TensorShape(17U, 17U, 384U),
            // reduction_a_3x3_2_relu
            TensorShape(35U, 35U, 224U),
            // reduction_a_3x3_3_relu, inception_b1_7x1_relu, inception_b1_1x7_3_relu, inception_b2_7x1_relu, inception_b2_1x7_3_relu, inception_b3_7x1_relu, inception_b3_1x7_3_relu, inception_b4_7x1_relu, inception_b4_1x7_3_relu, inception_b5_7x1_relu, inception_b5_1x7_3_relu, inception_b6_7x1_relu, inception_b6_1x7_3_relu, inception_b7_7x1_relu, inception_b7_1x7_3_relu, reduction_b_1x7_reduce_relu, reduction_b_1x7_relu
            TensorShape(17U, 17U, 256U),
            // inception_b1_1x7_reduce_relu, inception_b1_7x1_2_reduce_relu, inception_b1_7x1_2_relu, inception_b2_1x7_reduce_relu, inception_b2_7x1_2_reduce_relu, inception_b2_7x1_2_relu, inception_b3_1x7_reduce_relu, inception_b3_7x1_2_reduce_relu, inception_b3_7x1_2_relu, inception_b4_1x7_reduce_relu, inception_b4_7x1_2_reduce_relu, inception_b4_7x1_2_relu, inception_b5_1x7_reduce_relu, inception_b5_7x1_2_reduce_relu, inception_b5_7x1_2_relu, inception_b6_1x7_reduce_relu, inception_b6_7x1_2_reduce_relu, inception_b6_7x1_2_relu, inception_b7_1x7_reduce_relu, inception_b7_7x1_2_reduce_relu, inception_b7_7x1_2_relu, reduction_b_3x3_reduce_relu
            TensorShape(17U, 17U, 192U),
            // inception_b1_1x7_relu, inception_b1_1x7_2_relu, inception_b1_7x1_3_relu, inception_b2_1x7_relu, inception_b2_1x7_2_relu, inception_b2_7x1_3_relu, inception_b3_1x7_relu, inception_b3_1x7_2_relu, inception_b3_7x1_3_relu, inception_b4_1x7_relu, inception_b4_1x7_2_relu, inception_b4_7x1_3_relu, inception_b5_1x7_relu, inception_b5_1x7_2_relu, inception_b5_7x1_3_relu, inception_b6_1x7_relu, inception_b6_1x7_2_relu, inception_b6_7x1_3_relu, inception_b7_1x7_relu, inception_b7_1x7_2_relu, inception_b7_7x1_3_relu
            TensorShape(17U, 17U, 224U),
            // inception_b1_1x1_relu, inception_b2_1x1_relu, inception_b3_1x1_relu, inception_b4_1x1_relu, inception_b5_1x1_relu, inception_b6_1x1_relu, inception_b7_1x1_relu
            TensorShape(17U, 17U, 128U),
            // reduction_b_3x3_relu
            TensorShape(8U, 8U, 192U),
            // reduction_b_7x1_relu
            TensorShape(17U, 17U, 320U),
            // reduction_b_3x3_2_relu
            TensorShape(8U, 8U, 320U),
            // inception_c1_1x1_2_relu, inception_c1_1x3_relu, inception_c1_3x1_relu, inception_c1_1x3_3_relu, inception_c1_3x1_3_relu, inception_c1_1x1_relu, inception_c2_1x1_2_relu, inception_c2_1x3_relu, inception_c2_3x1_relu, inception_c2_1x3_3_relu, inception_c2_3x1_3_relu, inception_c2_1x1_relu, inception_c3_1x1_2_relu, inception_c3_1x3_relu, inception_c3_3x1_relu, inception_c3_1x3_3_relu, inception_c3_3x1_3_relu, inception_c3_1x1_relu
            TensorShape(8U, 8U, 256U),
            // inception_c1_1x1_3_relu, inception_c1_1x1_4_relu, inception_c2_1x1_3_relu, inception_c2_1x1_4_relu, inception_c3_1x1_3_relu, inception_c3_1x1_4_relu
            TensorShape(8U, 8U, 384U),
            // inception_c1_3x1_2_relu, inception_c2_3x1_2_relu, inception_c3_3x1_2_relu
            TensorShape(8U, 8U, 448U),
            // inception_c1_1x3_2_relu, inception_c2_1x3_2_relu, inception_c3_1x3_2_relu
            TensorShape(8U, 8U, 512U) }),
        framework::dataset::make("Info", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
    }
    {
    }
    GoogLeNetInceptionV4ActivationLayerDataset(GoogLeNetInceptionV4ActivationLayerDataset &&) = default;
    ~GoogLeNetInceptionV4ActivationLayerDataset()                                             = default;
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_ACTIVATION_LAYER_DATASET */
