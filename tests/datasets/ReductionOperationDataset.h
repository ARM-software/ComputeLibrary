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
#ifndef __ARM_COMPUTE_TEST_REDUCTION_OPERATION_DATASET_H__
#define __ARM_COMPUTE_TEST_REDUCTION_OPERATION_DATASET_H__

#include "arm_compute/core/Types.h"
#include "tests/framework/datasets/ContainerDataset.h"
#include "utils/TypePrinter.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ReductionOperations final : public framework::dataset::ContainerDataset<std::vector<ReductionOperation>>
{
public:
    ReductionOperations()
        : ContainerDataset("ReductionOperation",
    {
        ReductionOperation::SUM_SQUARE
    })
    {
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_REDUCTION_OPERATION_DATASET_H__ */
