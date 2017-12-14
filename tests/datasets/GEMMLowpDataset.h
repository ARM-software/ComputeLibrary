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
#ifndef ARM_COMPUTE_TEST_GEMMLOWP_DATASET
#define ARM_COMPUTE_TEST_GEMMLOWP_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class GEMMLowpDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, int32_t, int32_t>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator a_it,
                 std::vector<TensorShape>::const_iterator b_it,
                 std::vector<TensorShape>::const_iterator c_it,
                 std::vector<int32_t>::const_iterator     a_offset_it,
                 std::vector<int32_t>::const_iterator     b_offset_it)
            : _a_it{ std::move(a_it) },
              _b_it{ std::move(b_it) },
              _c_it{ std::move(c_it) },
              _a_offset_it{ std::move(a_offset_it) },
              _b_offset_it{ std::move(b_offset_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "A=" << *_a_it << ":";
            description << "B=" << *_b_it << ":";
            description << "C=" << *_c_it << ":";
            description << "a_offset=" << *_a_offset_it << ":";
            description << "b_offset=" << *_b_offset_it << ":";
            return description.str();
        }

        GEMMLowpDataset::type operator*() const
        {
            return std::make_tuple(*_a_it, *_b_it, *_c_it, *_a_offset_it, *_b_offset_it);
        }

        iterator &operator++()
        {
            ++_a_it;
            ++_b_it;
            ++_c_it;
            ++_a_offset_it;
            ++_b_offset_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _a_it;
        std::vector<TensorShape>::const_iterator _b_it;
        std::vector<TensorShape>::const_iterator _c_it;
        std::vector<int32_t>::const_iterator     _a_offset_it;
        std::vector<int32_t>::const_iterator     _b_offset_it;
    };

    iterator begin() const
    {
        return iterator(_a_shapes.begin(), _b_shapes.begin(), _c_shapes.begin(), _a_offset.begin(), _b_offset.begin());
    }

    int size() const
    {
        return std::min(_a_shapes.size(), std::min(_b_shapes.size(), std::min(_c_shapes.size(), std::min(_a_offset.size(), _b_offset.size()))));
    }

    void add_config(TensorShape a, TensorShape b, TensorShape c, int32_t a_offset, int32_t b_offset)
    {
        _a_shapes.emplace_back(std::move(a));
        _b_shapes.emplace_back(std::move(b));
        _c_shapes.emplace_back(std::move(c));
        _a_offset.emplace_back(std::move(a_offset));
        _b_offset.emplace_back(std::move(b_offset));
    }

protected:
    GEMMLowpDataset()                   = default;
    GEMMLowpDataset(GEMMLowpDataset &&) = default;

private:
    std::vector<TensorShape> _a_shapes{};
    std::vector<TensorShape> _b_shapes{};
    std::vector<TensorShape> _c_shapes{};
    std::vector<int32_t>     _a_offset{};
    std::vector<int32_t>     _b_offset{};
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMMLOWP_DATASET */
