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
#ifndef ARM_COMPUTE_TEST_SPACE_TO_BATCH_LAYER_DATASET
#define ARM_COMPUTE_TEST_SPACE_TO_BATCH_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class SpaceToBatchLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, TensorShape>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator src_it,
                 std::vector<TensorShape>::const_iterator block_shape_it,
                 std::vector<TensorShape>::const_iterator paddings_shape_it,
                 std::vector<TensorShape>::const_iterator dst_it)
            : _src_it{ std::move(src_it) },
              _block_shape_it{ std::move(block_shape_it) },
              _paddings_shape_it{ std::move(paddings_shape_it) },
              _dst_it{ std::move(dst_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "BlockShape=" << *_block_shape_it << ":";
            description << "PaddingsShape=" << *_paddings_shape_it << ":";
            description << "Out=" << *_dst_it;
            return description.str();
        }

        SpaceToBatchLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_block_shape_it, *_paddings_shape_it, *_dst_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_block_shape_it;
            ++_paddings_shape_it;
            ++_dst_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _src_it;
        std::vector<TensorShape>::const_iterator _block_shape_it;
        std::vector<TensorShape>::const_iterator _paddings_shape_it;
        std::vector<TensorShape>::const_iterator _dst_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _block_shape_shapes.begin(), _padding_shapes.begin(), _dst_shapes.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_block_shape_shapes.size(), std::min(_padding_shapes.size(), _dst_shapes.size())));
    }

    void add_config(TensorShape src, TensorShape block_shape, TensorShape padding_shapes, TensorShape dst)
    {
        _src_shapes.emplace_back(std::move(src));
        _block_shape_shapes.emplace_back(std::move(block_shape));
        _padding_shapes.emplace_back(std::move(padding_shapes));
        _dst_shapes.emplace_back(std::move(dst));
    }

protected:
    SpaceToBatchLayerDataset()                            = default;
    SpaceToBatchLayerDataset(SpaceToBatchLayerDataset &&) = default;

private:
    std::vector<TensorShape> _src_shapes{};
    std::vector<TensorShape> _block_shape_shapes{};
    std::vector<TensorShape> _padding_shapes{};
    std::vector<TensorShape> _dst_shapes{};
};

class SmallSpaceToBatchLayerDataset final : public SpaceToBatchLayerDataset
{
public:
    SmallSpaceToBatchLayerDataset()
    {
        add_config(TensorShape(2U, 2U, 1U, 1U), TensorShape(2U), TensorShape(2U, 2U), TensorShape(1U, 1U, 1U, 4U));
        add_config(TensorShape(6U, 2U, 1U, 1U), TensorShape(2U), TensorShape(2U, 2U), TensorShape(3U, 1U, 1U, 4U));
        add_config(TensorShape(2U, 4U, 2U, 1U), TensorShape(2U), TensorShape(2U, 2U), TensorShape(1U, 2U, 2U, 4U));
        add_config(TensorShape(2U, 6U, 1U, 2U), TensorShape(2U), TensorShape(2U, 2U), TensorShape(1U, 3U, 1U, 8U));
        add_config(TensorShape(6U, 8U, 1U, 1U), TensorShape(2U), TensorShape(2U, 2U), TensorShape(3U, 4U, 1U, 4U));
        add_config(TensorShape(6U, 8U, 15U, 5U), TensorShape(2U), TensorShape(2U, 2U), TensorShape(3U, 4U, 15U, 20U));
    }
};
class LargeSpaceToBatchLayerDataset final : public SpaceToBatchLayerDataset
{
public:
    LargeSpaceToBatchLayerDataset()
    {
        add_config(TensorShape(128U, 64U, 2U, 1U), TensorShape(2U), TensorShape(2U, 2U), TensorShape(64U, 32U, 2U, 4U));
        add_config(TensorShape(512U, 64U, 2U, 1U), TensorShape(2U), TensorShape(2U, 2U), TensorShape(128U, 16U, 2U, 16U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SPACE_TO_BATCH_LAYER_DATASET */
