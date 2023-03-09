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
#ifndef ACL_TESTS_DATASETS_SMALLBATCHMATMULDATASET
#define ACL_TESTS_DATASETS_SMALLBATCHMATMULDATASET

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/datasets/BatchMatMulDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class SmallBatchMatMulDataset final : public BatchMatMulDataset
{
public:
    SmallBatchMatMulDataset()
    {
        add_config(TensorShape(3U, 4U, 2U, 2U), TensorShape(2U, 3U, 2U, 2U), TensorShape(2U, 4U, 2U, 2U));
        add_config(TensorShape(9U, 6U), TensorShape(5U, 9U), TensorShape(5U, 6U));
        add_config(TensorShape(31U, 1U), TensorShape(23U, 31U), TensorShape(23U, 1U));
        add_config(TensorShape(8U, 4U, 2U), TensorShape(16U, 8U, 2U), TensorShape(16U, 4U, 2U));
        add_config(TensorShape(32U, 2U), TensorShape(17U, 32U), TensorShape(17U, 2U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ACL_TESTS_DATASETS_SMALLBATCHMATMULDATASET */
