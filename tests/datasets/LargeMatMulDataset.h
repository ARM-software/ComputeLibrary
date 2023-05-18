/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_LARGEMATMULDATASET
#define ACL_TESTS_DATASETS_LARGEMATMULDATASET

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/datasets/MatMulDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class LargeMatMulDataset final : public MatMulDataset
{
public:
    LargeMatMulDataset()
    {
        add_config(TensorShape(21U, 13U, 3U, 2U), TensorShape(33U, 21U, 3U, 2U), TensorShape(33U, 13U, 3U, 2U));
        add_config(TensorShape(38U, 12U, 1U, 5U), TensorShape(21U, 38U, 1U, 5U), TensorShape(21U, 12U, 1U, 5U));
        add_config(TensorShape(45U, 38U, 3U, 2U), TensorShape(21U, 45U, 3U, 2U), TensorShape(21U, 38U, 3U, 2U));
    }
};

class HighDimensionalMatMulDataset final : public MatMulDataset
{
public:
    HighDimensionalMatMulDataset()
    {
        add_config(TensorShape(5U, 5U, 2U, 2U, 2U, 2U), TensorShape(5U, 5U, 2U, 2U, 2U, 2U), TensorShape(5U, 5U, 2U, 2U, 2U, 2U)); // 6D tensor
    }
};

class LargeMatMulDatasetRhsExportToCLImageRhsNT final : public MatMulDataset
{
public:
    // For shape choices, please refer to the explanations given in SmallMatMulDatasetRhsExportToCLImageRhsNT
    LargeMatMulDatasetRhsExportToCLImageRhsNT()
    {
        add_config(TensorShape(21U, 13U, 3U, 2U), TensorShape(32U, 21U, 3U, 2U), TensorShape(32U, 13U, 3U, 2U));
        add_config(TensorShape(38U, 12U, 1U, 5U, 2U), TensorShape(20U, 38U, 1U, 5U, 2U), TensorShape(20U, 12U, 1U, 5U, 2U));
        add_config(TensorShape(45U, 38U, 3U, 2U, 3U), TensorShape(20U, 45U, 3U, 2U, 3U), TensorShape(20U, 38U, 3U, 2U, 3U));
    }
};
class LargeMatMulDatasetRhsExportToCLImageRhsT final : public MatMulDataset
{
public:
    // For shape choices, please refer to the explanations given in SmallMatMulDatasetRhsExportToCLImageRhsT
    LargeMatMulDatasetRhsExportToCLImageRhsT()
    {
        add_config(TensorShape(28U, 13U, 3U, 2U), TensorShape(32U, 28U, 3U, 2U), TensorShape(32U, 13U, 3U, 2U));
        add_config(TensorShape(40U, 12U, 1U, 5U, 2U), TensorShape(20U, 40U, 1U, 5U, 2U), TensorShape(20U, 12U, 1U, 5U, 2U));
        add_config(TensorShape(44U, 38U, 3U, 2U, 3U), TensorShape(20U, 44U, 3U, 2U, 3U), TensorShape(20U, 38U, 3U, 2U, 3U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ACL_TESTS_DATASETS_LARGEMATMULDATASET */
