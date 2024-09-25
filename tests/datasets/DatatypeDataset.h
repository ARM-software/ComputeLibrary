/*
 * Copyright (c) 2019-2020, 2024 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_DATATYPEDATASET_H
#define ACL_TESTS_DATASETS_DATATYPEDATASET_H

#include "arm_compute/core/CoreTypes.h"
#include "tests/framework/datasets/ContainerDataset.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace datasets
{
class AllDataTypes final : public framework::dataset::ContainerDataset<std::vector<DataType>>
{
public:
    AllDataTypes(const std::string &name)
        : ContainerDataset(name,
    {
        DataType::QSYMM8,
        DataType::QASYMM8,
        DataType::QASYMM8_SIGNED,
        DataType::QSYMM16,
        DataType::U8,                 /**< unsigned 8-bit number */
        DataType::S8,                 /**< signed 8-bit number */
        DataType::QSYMM8_PER_CHANNEL, /**< quantized, symmetric per channel fixed-point 8-bit number */
        DataType::U16,                /**< unsigned 16-bit number */
        DataType::S16,                /**< signed 16-bit number */
        DataType::QSYMM16,            /**< quantized, symmetric fixed-point 16-bit number */
        DataType::QASYMM16,           /**< quantized, asymmetric fixed-point 16-bit number */
        DataType::U32,                /**< unsigned 32-bit number */
        DataType::S32,                /**< signed 32-bit number */
        DataType::U64,                /**< unsigned 64-bit number */
        DataType::S64,                /**< signed 64-bit number */
        DataType::BFLOAT16,           /**< 16-bit brain floating-point number */
        DataType::F16,                /**< 16-bit floating-point number */
        DataType::F32,                /**< 32-bit floating-point number */
        DataType::F64,                /**< 64-bit floating-point number */
        DataType::SIZET               /**< size_t */
    })
    {
    }
};

class CommonDataTypes final : public framework::dataset::ContainerDataset<std::vector<DataType>>
{
public:
    CommonDataTypes(const std::string &name)
        : ContainerDataset(name,
    {
        DataType::QASYMM8,
        DataType::QASYMM8_SIGNED,
        DataType::QSYMM8_PER_CHANNEL, /**< quantized, symmetric per channel fixed-point 8-bit number */
        DataType::S32,                /**< signed 32-bit number */
        DataType::BFLOAT16,           /**< 16-bit brain floating-point number */
        DataType::F16,                /**< 16-bit floating-point number */
        DataType::F32,                /**< 32-bit floating-point number */
    })
    {
    }
};
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
#endif // ACL_TESTS_DATASETS_DATATYPEDATASET_H
