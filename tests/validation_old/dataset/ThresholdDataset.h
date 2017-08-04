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
#ifndef __ARM_COMPUTE_TEST_DATASET_THRESHOLD_DATASET_H__
#define __ARM_COMPUTE_TEST_DATASET_THRESHOLD_DATASET_H__

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/validation_old/dataset/GenericDataset.h"

#include <ostream>
#include <sstream>

#include <tuple>
#include <type_traits>

#ifdef BOOST
#include "tests/validation_old/boost_wrapper.h"
#endif /* BOOST */

namespace arm_compute
{
namespace test
{
class ThresholdDataObject
{
public:
    uint8_t       threshold;
    uint8_t       false_value;
    uint8_t       true_value;
    ThresholdType type;
    uint8_t       upper;

    operator std::string() const
    {
        std::stringstream ss;
        ss << "Threshold";
        ss << "_threshold_value" << threshold;
        ss << "_false_value" << std::boolalpha << false_value;
        ss << "_true_value" << std::boolalpha << true_value;
        ss << "_type";
        ss << ((type == ThresholdType::BINARY) ? "binary" : "range");
        ss << "_upper" << upper;
        return ss.str();
    }

    friend std::ostream &operator<<(std::ostream &os, const ThresholdDataObject &obj)
    {
        os << static_cast<std::string>(obj);
        return os;
    }
};

class ThresholdDataset : public GenericDataset<ThresholdDataObject, 4>
{
public:
    ThresholdDataset()
        : GenericDataset
    {
        ThresholdDataObject{ 10U, 25U, 3U, ThresholdType::BINARY, 0U },
        ThresholdDataObject{ 20U, 1U, 0U, ThresholdType::BINARY, 0U },
        ThresholdDataObject{ 30U, 1U, 0U, ThresholdType::RANGE, 100U },
        ThresholdDataObject{ 100U, 1U, 0U, ThresholdType::RANGE, 200U },
    }
    {
    }

    ~ThresholdDataset() = default;
};

} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_DATASET_THRESHOLD_DATASET_H__ */
