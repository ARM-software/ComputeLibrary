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
#ifndef ARM_COMPUTE_TEST_RANDOM_NORMALIZE_PLANAR_YUV_LAYER_DATASET
#define ARM_COMPUTE_TEST_RANDOM_NORMALIZE_PLANAR_YUV_LAYER_DATASET

#include "tests/datasets/NormalizePlanarYUVLayerDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class RandomNormalizePlanarYUVLayerDataset final : public NormalizePlanarYUVLayerDataset
{
public:
    RandomNormalizePlanarYUVLayerDataset()
    {
        add_config(TensorShape(15U, 4U, 4U, 1U), TensorShape(4U));
        add_config(TensorShape(21U, 11U, 12U, 1U), TensorShape(12U));
        add_config(TensorShape(7U, 3U, 6U, 1U), TensorShape(6U));
        add_config(TensorShape(7U, 2U, 3U, 1U), TensorShape(3U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_RANDOM_NORMALIZE_PLANAR_YUV_LAYER_DATASET */
