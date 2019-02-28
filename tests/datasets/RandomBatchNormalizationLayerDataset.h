/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_RANDOM_BATCH_NORMALIZATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_RANDOM_BATCH_NORMALIZATION_LAYER_DATASET

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
class SmallRandomBatchNormalizationLayerDataset final : public BatchNormalizationLayerDataset
{
public:
    SmallRandomBatchNormalizationLayerDataset()
    {
        add_config(TensorShape(15U, 16U, 2U, 12U), TensorShape(2U), 0.1f);
        add_config(TensorShape(21U, 11U, 12U, 7U), TensorShape(12U), 0.1f);
        add_config(TensorShape(7U, 3U, 6U, 11U), TensorShape(6U), 0.1f);
    }
};
class LargeRandomBatchNormalizationLayerDataset final : public BatchNormalizationLayerDataset
{
public:
    LargeRandomBatchNormalizationLayerDataset()
    {
        add_config(TensorShape(111U, 47U, 21U, 11U), TensorShape(21U), 0.1f);
        add_config(TensorShape(236U, 3U, 169U, 7U), TensorShape(169U), 0.1f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_RANDOM_BATCH_NORMALIZATION_LAYER_DATASET */
