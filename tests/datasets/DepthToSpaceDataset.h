/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_DEPTH_TO_SPACE_LAYER_DATASET
#define ARM_COMPUTE_TEST_DEPTH_TO_SPACE_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class DepthToSpaceLayerDataset
{
public:
    using type = std::tuple<TensorShape, int32_t, TensorShape>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator src_it,
                 std::vector<int32_t>::const_iterator     block_shape_it,
                 std::vector<TensorShape>::const_iterator dst_it)
            : _src_it{ std::move(src_it) },
              _block_shape_it{ std::move(block_shape_it) },
              _dst_it{ std::move(dst_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "BlockShape=" << *_block_shape_it << ":";
            description << "Out=" << *_dst_it;
            return description.str();
        }

        DepthToSpaceLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_block_shape_it, *_dst_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_block_shape_it;
            ++_dst_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _src_it;
        std::vector<int32_t>::const_iterator     _block_shape_it;
        std::vector<TensorShape>::const_iterator _dst_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _block_shape.begin(), _dst_shapes.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_block_shape.size(), _dst_shapes.size()));
    }

    void add_config(TensorShape src, int32_t block_shape, TensorShape dst)
    {
        _src_shapes.emplace_back(std::move(src));
        _block_shape.emplace_back(std::move(block_shape));
        _dst_shapes.emplace_back(std::move(dst));
    }

protected:
    DepthToSpaceLayerDataset()                            = default;
    DepthToSpaceLayerDataset(DepthToSpaceLayerDataset &&) = default;

private:
    std::vector<TensorShape> _src_shapes{};
    std::vector<int32_t>     _block_shape{};
    std::vector<TensorShape> _dst_shapes{};
};

class SmallDepthToSpaceLayerDataset final : public DepthToSpaceLayerDataset
{
public:
    SmallDepthToSpaceLayerDataset()
    {
        add_config(TensorShape(1U, 1U, 4U, 4U), 2U, TensorShape(2U, 2U, 1U, 4U));
        add_config(TensorShape(3U, 1U, 4U, 4U), 2U, TensorShape(6U, 2U, 1U, 4U));
        add_config(TensorShape(1U, 2U, 4U, 4U), 2U, TensorShape(2U, 4U, 1U, 4U));
        add_config(TensorShape(1U, 3U, 4U, 8U), 2U, TensorShape(2U, 6U, 1U, 8U));
        add_config(TensorShape(3U, 4U, 4U, 4U), 2U, TensorShape(6U, 8U, 1U, 4U));
        add_config(TensorShape(1U, 1U, 16U, 8U), 4U, TensorShape(4U, 4U, 1U, 8U));
        add_config(TensorShape(1U, 1U, 8U), 2U, TensorShape(2U, 2U, 2U));
    }
};

class LargeDepthToSpaceLayerDataset final : public DepthToSpaceLayerDataset
{
public:
    LargeDepthToSpaceLayerDataset()
    {
        add_config(TensorShape(64U, 32U, 4U), 2U, TensorShape(128U, 64U, 1U));
        add_config(TensorShape(128U, 16U, 16U), 4U, TensorShape(512U, 64U, 1U));
        add_config(TensorShape(16U, 8U, 8U), 2U, TensorShape(32U, 16U, 2U));
        add_config(TensorShape(8U, 16U, 16U), 2U, TensorShape(16U, 32U, 4U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTH_TO_SPACE_LAYER_DATASET */
