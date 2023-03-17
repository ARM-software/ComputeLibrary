/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_MATMULDATASET
#define ACL_TESTS_DATASETS_MATMULDATASET

#include "arm_compute/core/TensorShape.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class MatMulDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator a_it,
                 std::vector<TensorShape>::const_iterator b_it,
                 std::vector<TensorShape>::const_iterator dst_it)
            : _a_it{ std::move(a_it) },
              _b_it{ std::move(b_it) },
              _dst_it{ std::move(dst_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "A=" << *_a_it << ":";
            description << "B=" << *_b_it << ":";
            description << "Out=" << *_dst_it << ":";
            return description.str();
        }

        MatMulDataset::type operator*() const
        {
            return std::make_tuple(*_a_it, *_b_it, *_dst_it);
        }

        iterator &operator++()
        {
            ++_a_it;
            ++_b_it;
            ++_dst_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _a_it;
        std::vector<TensorShape>::const_iterator _b_it;
        std::vector<TensorShape>::const_iterator _dst_it;
    };

    iterator begin() const
    {
        return iterator(_a_shapes.begin(), _b_shapes.begin(), _dst_shapes.begin());
    }

    int size() const
    {
        return std::min(_a_shapes.size(), std::min(_b_shapes.size(), _dst_shapes.size()));
    }

    void add_config(TensorShape a, TensorShape b, TensorShape dst)
    {
        _a_shapes.emplace_back(std::move(a));
        _b_shapes.emplace_back(std::move(b));
        _dst_shapes.emplace_back(std::move(dst));
    }

protected:
    MatMulDataset()                 = default;
    MatMulDataset(MatMulDataset &&) = default;

private:
    std::vector<TensorShape> _a_shapes{};
    std::vector<TensorShape> _b_shapes{};
    std::vector<TensorShape> _dst_shapes{};
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ACL_TESTS_DATASETS_MATMULDATASET */
