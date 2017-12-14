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
#ifndef ARM_COMPUTE_TEST_FULLYCONNECTED_LAYER_DATASET
#define ARM_COMPUTE_TEST_FULLYCONNECTED_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class FullyConnectedLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, TensorShape>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator src_it,
                 std::vector<TensorShape>::const_iterator weights_it,
                 std::vector<TensorShape>::const_iterator biases_it,
                 std::vector<TensorShape>::const_iterator dst_it)
            : _src_it{ std::move(src_it) },
              _weights_it{ std::move(weights_it) },
              _biases_it{ std::move(biases_it) },
              _dst_it{ std::move(dst_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "Weights=" << *_weights_it << ":";
            description << "Biases=" << *_biases_it << ":";
            description << "Out=" << *_dst_it;
            return description.str();
        }

        FullyConnectedLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_weights_it, *_biases_it, *_dst_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_weights_it;
            ++_biases_it;
            ++_dst_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _src_it;
        std::vector<TensorShape>::const_iterator _weights_it;
        std::vector<TensorShape>::const_iterator _biases_it;
        std::vector<TensorShape>::const_iterator _dst_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _weight_shapes.begin(), _bias_shapes.begin(), _dst_shapes.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_weight_shapes.size(), std::min(_bias_shapes.size(), _dst_shapes.size())));
    }

    void add_config(TensorShape src, TensorShape weights, TensorShape biases, TensorShape dst)
    {
        _src_shapes.emplace_back(std::move(src));
        _weight_shapes.emplace_back(std::move(weights));
        _bias_shapes.emplace_back(std::move(biases));
        _dst_shapes.emplace_back(std::move(dst));
    }

protected:
    FullyConnectedLayerDataset()                              = default;
    FullyConnectedLayerDataset(FullyConnectedLayerDataset &&) = default;

private:
    std::vector<TensorShape> _src_shapes{};
    std::vector<TensorShape> _weight_shapes{};
    std::vector<TensorShape> _bias_shapes{};
    std::vector<TensorShape> _dst_shapes{};
};

class SmallFullyConnectedLayerDataset final : public FullyConnectedLayerDataset
{
public:
    SmallFullyConnectedLayerDataset()
    {
        // Conv -> FC
        add_config(TensorShape(1U, 1U, 1U, 3U), TensorShape(1U, 10U), TensorShape(10U), TensorShape(10U, 3U));
        // Conv -> FC
        add_config(TensorShape(9U, 5U, 7U), TensorShape(315U, 271U), TensorShape(271U), TensorShape(271U));
        // Conv -> FC (batched)
        add_config(TensorShape(9U, 5U, 7U, 3U), TensorShape(315U, 271U), TensorShape(271U), TensorShape(271U, 3U));
        // FC -> FC
        add_config(TensorShape(1U), TensorShape(1U, 10U), TensorShape(10U), TensorShape(10U));
        // FC -> FC (batched)
        add_config(TensorShape(1U, 3U), TensorShape(1U, 10U), TensorShape(10U), TensorShape(10U, 3U));
        // FC -> FC
        add_config(TensorShape(201U), TensorShape(201U, 529U), TensorShape(529U), TensorShape(529U));
        // FC -> FC (batched)
        add_config(TensorShape(201U, 3U), TensorShape(201U, 529U), TensorShape(529U), TensorShape(529U, 3U));

        add_config(TensorShape(9U, 5U, 7U, 3U, 2U), TensorShape(315U, 271U), TensorShape(271U), TensorShape(271U, 3U, 2U));
    }
};

class LargeFullyConnectedLayerDataset final : public FullyConnectedLayerDataset
{
public:
    LargeFullyConnectedLayerDataset()
    {
        add_config(TensorShape(9U, 5U, 257U), TensorShape(11565U, 2123U), TensorShape(2123U), TensorShape(2123U));
        add_config(TensorShape(9U, 5U, 257U, 2U), TensorShape(11565U, 2123U), TensorShape(2123U), TensorShape(2123U, 2U));
        add_config(TensorShape(3127U), TensorShape(3127U, 989U), TensorShape(989U), TensorShape(989U));
        add_config(TensorShape(3127U, 2U), TensorShape(3127U, 989U), TensorShape(989U), TensorShape(989U, 2U));

        add_config(TensorShape(9U, 5U, 257U, 2U, 3U), TensorShape(11565U, 2123U), TensorShape(2123U), TensorShape(2123U, 2U, 3U));
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FULLYCONNECTED_LAYER_DATASET */
