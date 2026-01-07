/*
 * Copyright (c) 2026 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_TOPKVDATASET_H
#define ACL_TESTS_DATASETS_TOPKVDATASET_H

#include "arm_compute/core/TensorShape.h"

#include "tests/framework/datasets/Datasets.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace datasets
{
/** Parent type for all for shape datasets. */
using ShapeDataset = framework::dataset::ContainerDataset<std::vector<TensorShape>>;

/** Data set containing small edge cases for TopKV operator. */
class Small1DTopKV final : public ShapeDataset
{
public:
    Small1DTopKV() : ShapeDataset("Shape", {TensorShape{8U, 1U}, TensorShape{1U, 8U}})
    {
    }
};

/** Data set containing small 2D tensor shapes for TopKV operator. */
class SmallTopKV final : public ShapeDataset
{
public:
    SmallTopKV()
        : ShapeDataset("Shape",
                       {TensorShape{8U, 1U}, TensorShape{8U, 7U}, TensorShape{15U, 13U}, TensorShape{32U, 64U}})
    {
    }
};

/** Data set containing small 2D tensor shapes for TopKV operator. */
class LargeTopKV final : public ShapeDataset
{
public:
    LargeTopKV()
        : ShapeDataset("Shape", {TensorShape{1000U, 64U}, TensorShape{1500U, 128U}, TensorShape{1000U, 32000U}})
    {
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_DATASETS_TOPKVDATASET_H
