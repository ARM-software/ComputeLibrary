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
#ifndef ARM_COMPUTE_TEST_RESHAPE_LAYER_DATASET
#define ARM_COMPUTE_TEST_RESHAPE_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ReshapeLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator in_it, std::vector<TensorShape>::const_iterator out_it)
            : _in_it{ std::move(in_it) }, _out_it{ std::move(out_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_in_it << ":";
            description << "Out=" << *_out_it;
            return description.str();
        }

        ReshapeLayerDataset::type operator*() const
        {
            return std::make_tuple(*_in_it, *_out_it);
        }

        iterator &operator++()
        {
            ++_in_it;
            ++_out_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _in_it;
        std::vector<TensorShape>::const_iterator _out_it;
    };

    iterator begin() const
    {
        return iterator(_in_shapes.begin(), _out_shapes.begin());
    }

    int size() const
    {
        return std::min(_in_shapes.size(), _out_shapes.size());
    }

    void add_config(TensorShape in, TensorShape out)
    {
        _in_shapes.emplace_back(std::move(in));
        _out_shapes.emplace_back(std::move(out));
    }

protected:
    ReshapeLayerDataset()                       = default;
    ReshapeLayerDataset(ReshapeLayerDataset &&) = default;

private:
    std::vector<TensorShape> _in_shapes{};
    std::vector<TensorShape> _out_shapes{};
};

class SmallReshapeLayerDataset final : public ReshapeLayerDataset
{
public:
    SmallReshapeLayerDataset()
    {
        add_config(TensorShape(16U), TensorShape(4U, 2U, 2U));
        add_config(TensorShape(2U, 2U, 8U), TensorShape(4U, 8U));
        add_config(TensorShape(3U, 3U, 16U), TensorShape(144U));
        add_config(TensorShape(17U, 3U, 12U), TensorShape(1U, 1U, 612U));
        add_config(TensorShape(26U, 26U, 32U), TensorShape(13U, 13U, 128U));
        add_config(TensorShape(31U, 23U, 4U, 7U), TensorShape(2U, 14U, 713U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_RESHAPE_LAYER_DATASET */
