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
#ifndef ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV1_POOLING_LAYER_DATASET
#define ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV1_POOLING_LAYER_DATASET

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
class GoogLeNetInceptionV1PoolingLayerDataset final : public PoolingLayerDataset
{
public:
    GoogLeNetInceptionV1PoolingLayerDataset()
    {
        // pool1/3x3_s2
        add_config(TensorShape(112U, 112U, 64U), TensorShape(56U, 56U, 64U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        // pool2/3x3_s2
        add_config(TensorShape(56U, 56U, 192U), TensorShape(28U, 28U, 192U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        // inception_3a/pool
        add_config(TensorShape(28U, 28U, 192U), TensorShape(28U, 28U, 192U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)));
        // inception_3b/pool
        add_config(TensorShape(28U, 28U, 256U), TensorShape(28U, 28U, 256U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)));
        // pool3/3x3_s2
        add_config(TensorShape(28U, 28U, 480U), TensorShape(14U, 14U, 480U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        // inception_4a/pool
        add_config(TensorShape(14U, 14U, 480U), TensorShape(14U, 14U, 480U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)));
        // inception_4b/pool, inception_4c/pool, inception_4d/pool
        add_config(TensorShape(14U, 14U, 512U), TensorShape(14U, 14U, 512U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)));
        // inception_4e/pool
        add_config(TensorShape(14U, 14U, 528U), TensorShape(14U, 14U, 528U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)));
        // pool4/3x3_s2
        add_config(TensorShape(14U, 14U, 832U), TensorShape(7U, 7U, 832U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        // inception_5a/pool, inception_5b/pool
        add_config(TensorShape(7U, 7U, 832U), TensorShape(7U, 7U, 832U), PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV1_POOLING_LAYER_DATASET */
