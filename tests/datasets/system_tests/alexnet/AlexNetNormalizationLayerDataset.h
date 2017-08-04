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
#ifndef ARM_COMPUTE_TEST_ALEXNET_NORMALIZATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_ALEXNET_NORMALIZATION_LAYER_DATASET

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
class AlexNetNormalizationLayerDataset final : public
    framework::dataset::CartesianProductDataset<framework::dataset::InitializerListDataset<TensorShape>, framework::dataset::SingletonDataset<NormalizationLayerInfo>>
{
public:
    AlexNetNormalizationLayerDataset()
        : CartesianProductDataset
    {
        framework::dataset::make("Shape", { TensorShape(55U, 55U, 96U), TensorShape(27U, 27U, 256U) }),
        framework::dataset::make("Info", NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f))
    }
    {
    }
    AlexNetNormalizationLayerDataset(AlexNetNormalizationLayerDataset &&) = default;
    ~AlexNetNormalizationLayerDataset()                                   = default;
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ALEXNET_NORMALIZATION_LAYER_DATASET */
