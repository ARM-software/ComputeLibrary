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
#ifndef ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_BATCHNORMALIZATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_BATCHNORMALIZATION_LAYER_DATASET

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
class GoogLeNetInceptionV4BatchNormalizationLayerDataset final : public BatchNormalizationLayerDataset
{
public:
    GoogLeNetInceptionV4BatchNormalizationLayerDataset()
    {
        // conv1_3x3_s2_bn
        add_config(TensorShape(149U, 149U, 32U), TensorShape(32U), 0.000010f);
        // conv2_3x3_s1_bn
        add_config(TensorShape(147U, 147U, 32U), TensorShape(32U), 0.000010f);
        // conv3_3x3_s1_bn
        add_config(TensorShape(147U, 147U, 64U), TensorShape(64U), 0.000010f);
        // inception_stem1_3x3_s2_bn
        add_config(TensorShape(73U, 73U, 96U), TensorShape(96U), 0.000010f);
        // inception_stem2_3x3_reduce_bn, inception_stem2_1x7_reduce_bn, inception_stem2_1x7_bn, inception_stem2_7x1_bn
        add_config(TensorShape(73U, 73U, 64U), TensorShape(64U), 0.000010f);
        // inception_stem2_3x3_bn, inception_stem2_3x3_2_bn
        add_config(TensorShape(71U, 71U, 96U), TensorShape(96U), 0.000010f);
        // inception_stem3_3x3_s2_bn, reduction_a_3x3_2_reduce_bn
        add_config(TensorShape(35U, 35U, 192U), TensorShape(192U), 0.000010f);
        // inception_a1_1x1_2_bn, inception_a1_3x3_bn, inception_a1_3x3_2_bn, inception_a1_3x3_3_bn, inception_a1_1x1_bn, inception_a2_1x1_2_bn, inception_a2_3x3_bn, inception_a2_3x3_2_bn, inception_a2_3x3_3_bn, inception_a2_1x1_bn, inception_a3_1x1_2_bn, inception_a3_3x3_bn, inception_a3_3x3_2_bn, inception_a3_3x3_3_bn, inception_a3_1x1_bn, inception_a4_1x1_2_bn, inception_a4_3x3_bn, inception_a4_3x3_2_bn, inception_a4_3x3_3_bn, inception_a4_1x1_bn
        add_config(TensorShape(35U, 35U, 96U), TensorShape(96U), 0.000010f);
        // inception_a1_3x3_reduce_bn, inception_a1_3x3_2_reduce_bn, inception_a2_3x3_reduce_bn, inception_a2_3x3_2_reduce_bn, inception_a3_3x3_reduce_bn, inception_a3_3x3_2_reduce_bn, inception_a4_3x3_reduce_bn, inception_a4_3x3_2_reduce_bn
        add_config(TensorShape(35U, 35U, 64U), TensorShape(64U), 0.000010f);
        // reduction_a_3x3_bn, inception_b1_1x1_2_bn, inception_b2_1x1_2_bn, inception_b3_1x1_2_bn, inception_b4_1x1_2_bn, inception_b5_1x1_2_bn, inception_b6_1x1_2_bn, inception_b7_1x1_2_bn
        add_config(TensorShape(17U, 17U, 384U), TensorShape(384U), 0.000010f);
        // reduction_a_3x3_2_bn
        add_config(TensorShape(35U, 35U, 224U), TensorShape(224U), 0.000010f);
        // reduction_a_3x3_3_bn, inception_b1_7x1_bn, inception_b1_1x7_3_bn, inception_b2_7x1_bn, inception_b2_1x7_3_bn, inception_b3_7x1_bn, inception_b3_1x7_3_bn, inception_b4_7x1_bn, inception_b4_1x7_3_bn, inception_b5_7x1_bn, inception_b5_1x7_3_bn, inception_b6_7x1_bn, inception_b6_1x7_3_bn, inception_b7_7x1_bn, inception_b7_1x7_3_bn, reduction_b_1x7_reduce_bn, reduction_b_1x7_bn
        add_config(TensorShape(17U, 17U, 256U), TensorShape(256U), 0.000010f);
        // inception_b1_1x7_reduce_bn, inception_b1_7x1_2_reduce_bn, inception_b1_7x1_2_bn, inception_b2_1x7_reduce_bn, inception_b2_7x1_2_reduce_bn, inception_b2_7x1_2_bn, inception_b3_1x7_reduce_bn, inception_b3_7x1_2_reduce_bn, inception_b3_7x1_2_bn, inception_b4_1x7_reduce_bn, inception_b4_7x1_2_reduce_bn, inception_b4_7x1_2_bn, inception_b5_1x7_reduce_bn, inception_b5_7x1_2_reduce_bn, inception_b5_7x1_2_bn, inception_b6_1x7_reduce_bn, inception_b6_7x1_2_reduce_bn, inception_b6_7x1_2_bn, inception_b7_1x7_reduce_bn, inception_b7_7x1_2_reduce_bn, inception_b7_7x1_2_bn, reduction_b_3x3_reduce_bn
        add_config(TensorShape(17U, 17U, 192U), TensorShape(192U), 0.000010f);
        // inception_b1_1x7_bn, inception_b1_1x7_2_bn, inception_b1_7x1_3_bn, inception_b2_1x7_bn, inception_b2_1x7_2_bn, inception_b2_7x1_3_bn, inception_b3_1x7_bn, inception_b3_1x7_2_bn, inception_b3_7x1_3_bn, inception_b4_1x7_bn, inception_b4_1x7_2_bn, inception_b4_7x1_3_bn, inception_b5_1x7_bn, inception_b5_1x7_2_bn, inception_b5_7x1_3_bn, inception_b6_1x7_bn, inception_b6_1x7_2_bn, inception_b6_7x1_3_bn, inception_b7_1x7_bn, inception_b7_1x7_2_bn, inception_b7_7x1_3_bn
        add_config(TensorShape(17U, 17U, 224U), TensorShape(224U), 0.000010f);
        // inception_b1_1x1_bn, inception_b2_1x1_bn, inception_b3_1x1_bn, inception_b4_1x1_bn, inception_b5_1x1_bn, inception_b6_1x1_bn, inception_b7_1x1_bn
        add_config(TensorShape(17U, 17U, 128U), TensorShape(128U), 0.000010f);
        // reduction_b_3x3_bn
        add_config(TensorShape(8U, 8U, 192U), TensorShape(192U), 0.000010f);
        // reduction_b_7x1_bn
        add_config(TensorShape(17U, 17U, 320U), TensorShape(320U), 0.000010f);
        // reduction_b_3x3_2_bn
        add_config(TensorShape(8U, 8U, 320U), TensorShape(320U), 0.000010f);
        // inception_c1_1x1_2_bn, inception_c1_1x3_bn, inception_c1_3x1_bn, inception_c1_1x3_3_bn, inception_c1_3x1_3_bn, inception_c1_1x1_bn, inception_c2_1x1_2_bn, inception_c2_1x3_bn, inception_c2_3x1_bn, inception_c2_1x3_3_bn, inception_c2_3x1_3_bn, inception_c2_1x1_bn, inception_c3_1x1_2_bn, inception_c3_1x3_bn, inception_c3_3x1_bn, inception_c3_1x3_3_bn, inception_c3_3x1_3_bn, inception_c3_1x1_bn
        add_config(TensorShape(8U, 8U, 256U), TensorShape(256U), 0.000010f);
        // inception_c1_1x1_3_bn, inception_c1_1x1_4_bn, inception_c2_1x1_3_bn, inception_c2_1x1_4_bn, inception_c3_1x1_3_bn, inception_c3_1x1_4_bn
        add_config(TensorShape(8U, 8U, 384U), TensorShape(384U), 0.000010f);
        // inception_c1_3x1_2_bn, inception_c2_3x1_2_bn, inception_c3_3x1_2_bn
        add_config(TensorShape(8U, 8U, 448U), TensorShape(448U), 0.000010f);
        // inception_c1_1x3_2_bn, inception_c2_1x3_2_bn, inception_c3_1x3_2_bn
        add_config(TensorShape(8U, 8U, 512U), TensorShape(512U), 0.000010f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_BATCHNORMALIZATION_LAYER_DATASET */
