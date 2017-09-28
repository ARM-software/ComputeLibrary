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
#ifndef ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_POOLING_LAYER_DATASET
#define ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_POOLING_LAYER_DATASET

#include "tests/datasets/PoolingLayerDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class GoogLeNetInceptionV4PoolingLayerDataset final : public PoolingLayerDataset
{
public:
    GoogLeNetInceptionV4PoolingLayerDataset()
    {
        // inception_stem1_pool
        add_config(TensorShape(147U, 147U, 64U), TensorShape(73U, 73U, 64U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        // inception_stem3_pool
        add_config(TensorShape(71U, 71U, 192U), TensorShape(35U, 35U, 192U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        // inception_a1_pool_ave, inception_a2_pool_ave, inception_a3_pool_ave, inception_a4_pool_ave
        add_config(TensorShape(35U, 35U, 384U), TensorShape(35U, 35U, 384U), PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)));
        // reduction_a_pool
        add_config(TensorShape(35U, 35U, 384U), TensorShape(17U, 17U, 384U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        // inception_b1_pool_ave, inception_b2_pool_ave, inception_b3_pool_ave, inception_b4_pool_ave, inception_b5_pool_ave, inception_b6_pool_ave, inception_b7_pool_ave
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(17U, 17U, 1024U), PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)));
        // reduction_b_pool
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(8U, 8U, 1024U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        // inception_c1_pool_ave, inception_c2_pool_ave, inception_c3_pool_ave
        add_config(TensorShape(8U, 8U, 1536U), TensorShape(8U, 8U, 1536U), PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_POOLING_LAYER_DATASET */
