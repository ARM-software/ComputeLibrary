/*
 * Copyright (c) 2018-2019 Arm Limited.
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

#ifndef ARM_COMPUTE_TEST_GATHER_DATASET
#define ARM_COMPUTE_TEST_GATHER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class GatherDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, int>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator input_shapes_it,
                 std::vector<TensorShape>::const_iterator starts_values_it,
                 std::vector<int>::const_iterator         axis_it)
            : _input_shapes_it{ std::move(input_shapes_it) },
              _indices_shapes_it{ std::move(starts_values_it) },
              _axis_it{ std::move(axis_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "InputShape=" << *_input_shapes_it << ":";
            description << "IndicesShape=" << *_indices_shapes_it << ":";
            description << "Axis=" << *_axis_it << ":";
            return description.str();
        }

        GatherDataset::type operator*() const
        {
            return std::make_tuple(*_input_shapes_it, *_indices_shapes_it, *_axis_it);
        }

        iterator &operator++()
        {
            ++_input_shapes_it;
            ++_indices_shapes_it;
            ++_axis_it;
            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _input_shapes_it;
        std::vector<TensorShape>::const_iterator _indices_shapes_it;
        std::vector<int>::const_iterator         _axis_it;
    };

    iterator begin() const
    {
        return iterator(_input_shapes.begin(), _indices_shapes.begin(), _axis.begin());
    }

    int size() const
    {
        return std::min(_input_shapes.size(), std::min(_indices_shapes.size(), _axis.size()));
    }

    void add_config(TensorShape input_shape, TensorShape indices_shape, int axis)
    {
        _input_shapes.emplace_back(std::move(input_shape));
        _indices_shapes.emplace_back(std::move(indices_shape));
        _axis.emplace_back(std::move(axis));
    }

protected:
    GatherDataset()                 = default;
    GatherDataset(GatherDataset &&) = default;

private:
    std::vector<TensorShape> _input_shapes{};
    std::vector<TensorShape> _indices_shapes{};
    std::vector<int>         _axis{};
};

class SmallGatherDataset final : public GatherDataset
{
public:
    SmallGatherDataset()
    {
        // 2D input
        add_config(TensorShape(15U, 15U), TensorShape(5U), 0);
        add_config(TensorShape(15U, 15U), TensorShape(5U), 1);
        add_config(TensorShape(5U, 5U), TensorShape(80U), -1);

        // 3D input
        add_config(TensorShape(5U, 5U, 5U), TensorShape(19U), 0);
        add_config(TensorShape(5U, 4U, 6U), TensorShape(30U), 1);
        add_config(TensorShape(3U, 5U, 7U), TensorShape(20U), 2);
        add_config(TensorShape(5U, 4U, 6U), TensorShape(30U), -1);
        add_config(TensorShape(3U, 5U, 7U), TensorShape(20U), -2);

        // 4D input
        add_config(TensorShape(4U, 3U, 4U, 5U), TensorShape(4U), 0);
        add_config(TensorShape(4U, 3U, 5U, 5U), TensorShape(5U), 1);
        add_config(TensorShape(4U, 3U, 2U, 5U), TensorShape(6U), 2);
        add_config(TensorShape(3U, 4U, 4U, 6U), TensorShape(7U), 3);
        add_config(TensorShape(4U, 3U, 5U, 5U), TensorShape(5U), -1);
        add_config(TensorShape(4U, 3U, 2U, 5U), TensorShape(6U), -2);
        add_config(TensorShape(3U, 4U, 4U, 6U), TensorShape(7U), -3);
    }
};

class LargeGatherDataset final : public GatherDataset
{
public:
    LargeGatherDataset()
    {
        // 2D input
        add_config(TensorShape(150U, 150U), TensorShape(50U), 0);
        add_config(TensorShape(150U, 150U), TensorShape(50U), 1);
        add_config(TensorShape(150U, 150U), TensorShape(50U), -1);

        // 3D input
        add_config(TensorShape(50U, 40U, 60U), TensorShape(33U), 0);
        add_config(TensorShape(40U, 50U, 60U), TensorShape(24U), 1);
        add_config(TensorShape(70U, 80U, 100U), TensorShape(50U), 2);
        add_config(TensorShape(40U, 50U, 60U), TensorShape(24U), -1);
        add_config(TensorShape(70U, 80U, 100U), TensorShape(50U), -2);

        // 4D input
        add_config(TensorShape(30U, 40U, 20U, 20U), TensorShape(33U), 0);
        add_config(TensorShape(23U, 10U, 60U, 20U), TensorShape(24U), 1);
        add_config(TensorShape(14U, 20U, 10U, 31U), TensorShape(30U), 2);
        add_config(TensorShape(34U, 10U, 40U, 20U), TensorShape(50U), 3);
        add_config(TensorShape(23U, 10U, 60U, 20U), TensorShape(24U), -1);
        add_config(TensorShape(14U, 20U, 10U, 31U), TensorShape(30U), -2);
        add_config(TensorShape(34U, 10U, 40U, 20U), TensorShape(50U), -3);
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute

#endif /* ARM_COMPUTE_TEST_GATHER_DATASET */
