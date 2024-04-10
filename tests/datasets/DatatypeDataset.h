/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_DATATYPE_DATASET_H
#define ARM_COMPUTE_TEST_DATATYPE_DATASET_H

#include "arm_compute/core/Types.h"
#include "tests/framework/datasets/ContainerDataset.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace datasets
{
class QuantizedTypes final : public framework::dataset::ContainerDataset<std::vector<DataType>>
{
public:
    QuantizedTypes()
        : ContainerDataset("QuantizedTypes",
    {
        DataType::QSYMM8,
                 DataType::QASYMM8,
                 DataType::QASYMM8_SIGNED,
                 DataType::QSYMM16,
    })
    {
    }
};
class QuantizedPerChannelTypes final : public framework::dataset::ContainerDataset<std::vector<DataType>>
{
public:
    QuantizedPerChannelTypes()
        : ContainerDataset("QuantizedPerChannelTypes",
    {
        DataType::QSYMM8_PER_CHANNEL
    })
    {
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATATYPE_DATASET_H */
