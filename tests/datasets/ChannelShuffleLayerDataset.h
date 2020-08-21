/*
 * Copyright (c) 2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_CHANNEL_SHUFFLE_LAYER_DATASET
#define ARM_COMPUTE_TEST_CHANNEL_SHUFFLE_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ChannelShuffleLayerDataset
{
public:
    using type = std::tuple<TensorShape, int>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator tensor_it,
                 std::vector<int>::const_iterator         num_groups_it)
            : _tensor_it{ std::move(tensor_it) },
              _num_groups_it{ std::move(num_groups_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_tensor_it << ":";
            description << "NumGroups=" << *_num_groups_it;
            return description.str();
        }

        ChannelShuffleLayerDataset::type operator*() const
        {
            return std::make_tuple(*_tensor_it, *_num_groups_it);
        }

        iterator &operator++()
        {
            ++_tensor_it;
            ++_num_groups_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _tensor_it;
        std::vector<int>::const_iterator         _num_groups_it;
    };

    iterator begin() const
    {
        return iterator(_tensor_shapes.begin(), _num_groups.begin());
    }

    int size() const
    {
        return std::min(_tensor_shapes.size(), _num_groups.size());
    }

    void add_config(TensorShape tensor, int num_groups)
    {
        _tensor_shapes.emplace_back(std::move(tensor));
        _num_groups.emplace_back(std::move(num_groups));
    }

protected:
    ChannelShuffleLayerDataset()                              = default;
    ChannelShuffleLayerDataset(ChannelShuffleLayerDataset &&) = default;

private:
    std::vector<TensorShape> _tensor_shapes{};
    std::vector<int>         _num_groups{};
};

class SmallRandomChannelShuffleLayerDataset final : public ChannelShuffleLayerDataset
{
public:
    SmallRandomChannelShuffleLayerDataset()
    {
        add_config(TensorShape(15U, 16U, 4U, 12U), 2);
        add_config(TensorShape(21U, 11U, 12U, 7U), 4);
        add_config(TensorShape(21U, 11U, 12U, 7U), 6);
        add_config(TensorShape(7U, 3U, 6U, 11U), 3);
    }
};

class LargeRandomChannelShuffleLayerDataset final : public ChannelShuffleLayerDataset
{
public:
    LargeRandomChannelShuffleLayerDataset()
    {
        add_config(TensorShape(210U, 43U, 20U, 3U), 5);
        add_config(TensorShape(283U, 213U, 15U, 3U), 3);
        add_config(TensorShape(500U, 115U, 16U, 2U), 4);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CHANNEL_SHUFFLE_LAYER_DATASET */
