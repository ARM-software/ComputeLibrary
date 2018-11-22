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
#ifndef ARM_COMPUTE_TEST_REORGLAYER_DATASET
#define ARM_COMPUTE_TEST_REORGLAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ReorgLayerDataset
{
public:
    using type = std::tuple<TensorShape, unsigned int>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator  src_it,
                 std::vector<unsigned int>::const_iterator stride_it)
            : _src_it{ std::move(src_it) },
              _stride_it{ std::move(stride_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "Stride=" << *_stride_it;
            return description.str();
        }

        ReorgLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_stride_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_stride_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator  _src_it;
        std::vector<unsigned int>::const_iterator _stride_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _stride.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), _stride.size());
    }

    void add_config(TensorShape src, unsigned int stride)
    {
        _src_shapes.emplace_back(std::move(src));
        _stride.emplace_back(std::move(stride));
    }

protected:
    ReorgLayerDataset()                     = default;
    ReorgLayerDataset(ReorgLayerDataset &&) = default;

private:
    std::vector<TensorShape>  _src_shapes{};
    std::vector<unsigned int> _stride{};
};

/** Dataset containing small reorg layer shapes. */
class SmallReorgLayerDataset final : public ReorgLayerDataset
{
public:
    SmallReorgLayerDataset()
    {
        add_config(TensorShape(26U, 26U, 64U, 1U), 2U);
        add_config(TensorShape(28U, 28U, 13U, 1U), 4U);
        add_config(TensorShape(12U, 14U, 4U, 1U), 2U);
        add_config(TensorShape(9U, 12U, 2U, 4U), 3U);
        add_config(TensorShape(25U, 15U, 4U, 2U), 5U);
    }
};

/** Dataset containing large reorg layer shapes. */
class LargeReorgLayerDataset final : public ReorgLayerDataset
{
public:
    LargeReorgLayerDataset()
    {
        add_config(TensorShape(49U, 28U, 64U, 1U), 7U);
        add_config(TensorShape(63U, 21U, 13U, 1U), 3U);
        add_config(TensorShape(48U, 54U, 4U, 1U), 2U);
        add_config(TensorShape(114U, 117U, 2U, 4U), 3U);
        add_config(TensorShape(100U, 95U, 4U, 2U), 5U);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_REORGLAYER_DATASET */
