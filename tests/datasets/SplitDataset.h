/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_SPLIT_DATASET
#define ARM_COMPUTE_TEST_SPLIT_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class SplitDataset
{
public:
    using type = std::tuple<TensorShape, unsigned int, unsigned int>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator  tensor_shapes_it,
                 std::vector<unsigned int>::const_iterator axis_values_it,
                 std::vector<unsigned int>::const_iterator splits_values_it)
            : _tensor_shapes_it{ std::move(tensor_shapes_it) },
              _axis_values_it{ std::move(axis_values_it) },
              _splits_values_it{ std::move(splits_values_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Shape=" << *_tensor_shapes_it << ":";
            description << "Axis=" << *_axis_values_it << ":";
            description << "Splits=" << *_splits_values_it << ":";
            return description.str();
        }

        SplitDataset::type operator*() const
        {
            return std::make_tuple(*_tensor_shapes_it, *_axis_values_it, *_splits_values_it);
        }

        iterator &operator++()
        {
            ++_tensor_shapes_it;
            ++_axis_values_it;
            ++_splits_values_it;
            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator  _tensor_shapes_it;
        std::vector<unsigned int>::const_iterator _axis_values_it;
        std::vector<unsigned int>::const_iterator _splits_values_it;
    };

    iterator begin() const
    {
        return iterator(_tensor_shapes.begin(), _axis_values.begin(), _splits_values.begin());
    }

    int size() const
    {
        return std::min(_tensor_shapes.size(), std::min(_axis_values.size(), _splits_values.size()));
    }

    void add_config(TensorShape shape, unsigned int axis, unsigned int splits)
    {
        _tensor_shapes.emplace_back(std::move(shape));
        _axis_values.emplace_back(axis);
        _splits_values.emplace_back(splits);
    }

protected:
    SplitDataset()                = default;
    SplitDataset(SplitDataset &&) = default;

private:
    std::vector<TensorShape>  _tensor_shapes{};
    std::vector<unsigned int> _axis_values{};
    std::vector<unsigned int> _splits_values{};
};

class SmallSplitDataset final : public SplitDataset
{
public:
    SmallSplitDataset()
    {
        add_config(TensorShape(128U), 0U, 4U);
        add_config(TensorShape(6U, 3U, 4U), 2U, 2U);
        add_config(TensorShape(27U, 14U, 2U), 1U, 2U);
        add_config(TensorShape(64U, 32U, 4U, 6U), 3U, 3U);
    }
};

class LargeSplitDataset final : public SplitDataset
{
public:
    LargeSplitDataset()
    {
        add_config(TensorShape(512U), 0U, 8U);
        add_config(TensorShape(128U, 64U, 8U), 2U, 2U);
        add_config(TensorShape(128U, 64U, 8U, 2U), 1U, 2U);
        add_config(TensorShape(128U, 64U, 32U, 4U), 3U, 4U);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SPLIT_DATASET */
