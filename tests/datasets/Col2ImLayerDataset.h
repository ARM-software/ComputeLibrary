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
#ifndef ARM_COMPUTE_TEST_COL2IM_DATASET
#define ARM_COMPUTE_TEST_COL2IM_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class Col2ImLayerDataset
{
public:
    using type = std::tuple<TensorShape, unsigned int, unsigned int, unsigned int>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator  src_it,
                 std::vector<unsigned int>::const_iterator convolved_width_it,
                 std::vector<unsigned int>::const_iterator convolved_height_it,
                 std::vector<unsigned int>::const_iterator num_groups_it)
            : _src_it{ std::move(src_it) },
              _convolved_width_it{ std::move(convolved_width_it) },
              _convolved_height_it{ std::move(convolved_height_it) },
              _num_groups_it{ std::move(num_groups_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "ConvolvedWidth=" << *_convolved_width_it << ":";
            description << "ConvolvedHeight=" << *_convolved_height_it << ":";
            description << "NumGroups=" << *_num_groups_it;
            return description.str();
        }

        Col2ImLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_convolved_width_it, *_convolved_height_it, *_num_groups_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_convolved_width_it;
            ++_convolved_height_it;
            ++_num_groups_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator  _src_it;
        std::vector<unsigned int>::const_iterator _convolved_width_it;
        std::vector<unsigned int>::const_iterator _convolved_height_it;
        std::vector<unsigned int>::const_iterator _num_groups_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _convolved_widths.begin(), _convolved_heights.begin(), _num_groups.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_convolved_widths.size(), std::min(_convolved_heights.size(), _num_groups.size())));
    }

    void add_config(TensorShape src, unsigned int convolved_width, unsigned int convolved_height, unsigned int info)
    {
        _src_shapes.emplace_back(std::move(src));
        _convolved_widths.emplace_back(std::move(convolved_width));
        _convolved_heights.emplace_back(std::move(convolved_height));
        _num_groups.emplace_back(std::move(info));
    }

protected:
    Col2ImLayerDataset()                      = default;
    Col2ImLayerDataset(Col2ImLayerDataset &&) = default;

private:
    std::vector<TensorShape>  _src_shapes{};
    std::vector<unsigned int> _convolved_widths{};
    std::vector<unsigned int> _convolved_heights{};
    std::vector<unsigned int> _num_groups{};
};

/** Dataset containing small grouped col2im shapes. */
class SmallGroupedCol2ImLayerDataset final : public Col2ImLayerDataset
{
public:
    SmallGroupedCol2ImLayerDataset()
    {
        add_config(TensorShape(10U, 12U, 1U, 1U), 3U, 4U, 1U);
        add_config(TensorShape(12U, 30U, 1U, 2U), 5U, 6U, 1U);
        add_config(TensorShape(12U, 30U, 4U, 1U), 5U, 6U, 1U);
        add_config(TensorShape(10U, 12U, 2U, 4U), 3U, 4U, 2U);
        add_config(TensorShape(10U, 12U, 2U, 4U), 3U, 4U, 2U);
        add_config(TensorShape(8U, 16U, 3U, 1U), 4U, 4U, 3U);
        add_config(TensorShape(8U, 16U, 3U, 3U), 4U, 4U, 3U);
        add_config(TensorShape(12U, 20U, 4U, 1U), 5U, 4U, 4U);
        add_config(TensorShape(12U, 20U, 4U, 3U), 5U, 4U, 4U);
    }
};

/** Dataset containing large grouped col2im shapes. */
class LargeGroupedCol2ImLayerDataset final : public Col2ImLayerDataset
{
public:
    LargeGroupedCol2ImLayerDataset()
    {
        add_config(TensorShape(233U, 280U, 1U, 55U), 14U, 20U, 1U);
        add_config(TensorShape(333U, 280U, 1U, 77U), 14U, 20U, 1U);
        add_config(TensorShape(333U, 280U, 77U, 1U), 14U, 20U, 1U);
        add_config(TensorShape(120U, 300U, 8U, 3U), 20U, 15U, 8U);
        add_config(TensorShape(233U, 300U, 8U, 3U), 20U, 15U, 8U);
        add_config(TensorShape(333U, 280U, 12U, 5U), 20U, 14U, 12U);
        add_config(TensorShape(177U, 300U, 12U, 5U), 15U, 20U, 12U);
        add_config(TensorShape(450U, 400U, 16U, 5U), 20U, 20U, 16U);
        add_config(TensorShape(220U, 400U, 16U, 5U), 20U, 20U, 16U);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_COL2IM_DATASET */
