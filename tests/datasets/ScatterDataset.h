/*
 * Copyright (c) 2024 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_SCATTERDATASET_H
#define ACL_TESTS_DATASETS_SCATTERDATASET_H

#include "arm_compute/core/TensorShape.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{

class ScatterDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, TensorShape>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator src_it,
                 std::vector<TensorShape>::const_iterator updates_it,
                 std::vector<TensorShape>::const_iterator indices_it,
                 std::vector<TensorShape>::const_iterator dst_it)
            : _src_it{ std::move(src_it) },
              _updates_it{ std::move(updates_it) },
              _indices_it{std::move(indices_it)},
              _dst_it{ std::move(dst_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "A=" << *_src_it << ":";
            description << "B=" << *_updates_it << ":";
            description << "C=" << *_indices_it << ":";
            description << "Out=" << *_dst_it << ":";
            return description.str();
        }

        ScatterDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_updates_it, *_indices_it, *_dst_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_updates_it;
            ++_indices_it;
            ++_dst_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _src_it;
        std::vector<TensorShape>::const_iterator _updates_it;
        std::vector<TensorShape>::const_iterator _indices_it;
        std::vector<TensorShape>::const_iterator _dst_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _update_shapes.begin(), _indices_shapes.begin(), _dst_shapes.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_indices_shapes.size(), std::min(_update_shapes.size(), _dst_shapes.size())));
    }

    void add_config(TensorShape a, TensorShape b, TensorShape c, TensorShape dst)
    {
        _src_shapes.emplace_back(std::move(a));
        _update_shapes.emplace_back(std::move(b));
        _indices_shapes.emplace_back(std::move(c));
        _dst_shapes.emplace_back(std::move(dst));
    }

protected:
    ScatterDataset()                 = default;
    ScatterDataset(ScatterDataset &&) = default;

private:
    std::vector<TensorShape> _src_shapes{};
    std::vector<TensorShape> _update_shapes{};
    std::vector<TensorShape> _indices_shapes{};
    std::vector<TensorShape> _dst_shapes{};
};

class Small1DScatterDataset final : public ScatterDataset
{
public:
    Small1DScatterDataset()
    {
        add_config(TensorShape(6U), TensorShape(6U), TensorShape(1U, 6U), TensorShape(6U));
        add_config(TensorShape(10U), TensorShape(2U), TensorShape(1U, 2U), TensorShape(10U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_DATASETS_SCATTERDATASET_H
