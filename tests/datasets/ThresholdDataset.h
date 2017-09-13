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
#ifndef ARM_COMPUTE_TEST_THRESHOLD_DATASET
#define ARM_COMPUTE_TEST_THRESHOLD_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ThresholdDataset
{
public:
    using type = std::tuple<uint8_t, uint8_t, uint8_t, ThresholdType, uint8_t>;

    struct iterator
    {
        iterator(std::vector<uint8_t>::const_iterator       threshold_it,
                 std::vector<uint8_t>::const_iterator       false_value_it,
                 std::vector<uint8_t>::const_iterator       true_value_it,
                 std::vector<ThresholdType>::const_iterator type_it,
                 std::vector<uint8_t>::const_iterator       upper_it)
            : _threshold_it{ std::move(threshold_it) },
              _false_value_it{ std::move(false_value_it) },
              _true_value_it{ std::move(true_value_it) },
              _type_it{ std::move(type_it) },
              _upper_it{ std::move(upper_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Threshold=" << static_cast<unsigned>(*_threshold_it) << ":";
            description << "FalseValue_=" << std::boolalpha << static_cast<unsigned>(*_false_value_it) << ":";
            description << "TrueValue=" << std::boolalpha << static_cast<unsigned>(*_true_value_it) << ":";
            description << "Type=" << ((*_type_it == ThresholdType::BINARY) ? "binary" : "range") << ":";
            description << "Upper=" << static_cast<unsigned>(*_upper_it);

            return description.str();
        }

        ThresholdDataset::type operator*() const
        {
            return std::make_tuple(*_threshold_it, *_false_value_it, *_true_value_it, *_type_it, *_upper_it);
        }

        iterator &operator++()
        {
            ++_threshold_it;
            ++_false_value_it;
            ++_true_value_it;
            ++_type_it;
            ++_upper_it;

            return *this;
        }

    private:
        std::vector<uint8_t>::const_iterator       _threshold_it;
        std::vector<uint8_t>::const_iterator       _false_value_it;
        std::vector<uint8_t>::const_iterator       _true_value_it;
        std::vector<ThresholdType>::const_iterator _type_it;
        std::vector<uint8_t>::const_iterator       _upper_it;
    };

    iterator begin() const
    {
        return iterator(_thresholds.begin(), _false_values.begin(), _true_values.begin(), _types.begin(), _uppers.begin());
    }

    int size() const
    {
        return std::min(_thresholds.size(), std::min(_false_values.size(), std::min(_true_values.size(), std::min(_types.size(), _uppers.size()))));
    }

    void add_config(uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType threshold_type, uint8_t upper)
    {
        _thresholds.emplace_back(std::move(threshold));
        _false_values.emplace_back(std::move(false_value));
        _true_values.emplace_back(std::move(true_value));
        _types.emplace_back(std::move(threshold_type));
        _uppers.emplace_back(std::move(upper));
    }

protected:
    ThresholdDataset()                    = default;
    ThresholdDataset(ThresholdDataset &&) = default;

private:
    std::vector<uint8_t>       _thresholds{};
    std::vector<uint8_t>       _false_values{};
    std::vector<uint8_t>       _true_values{};
    std::vector<ThresholdType> _types{};
    std::vector<uint8_t>       _uppers{};
};

class MixedThresholdDataset final : public ThresholdDataset
{
public:
    MixedThresholdDataset()
    {
        add_config(10U, 25U, 3U, ThresholdType::BINARY, 0U);
        add_config(20U, 1U, 0U, ThresholdType::BINARY, 0U);
        add_config(30U, 1U, 0U, ThresholdType::RANGE, 100U);
        add_config(100U, 1U, 0U, ThresholdType::RANGE, 200U);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_THRESHOLD_DATASET */
