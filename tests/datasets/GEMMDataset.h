/*
 * Copyright (c) 2017, 2025 Arm Limited.
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

#ifndef ACL_TESTS_DATASETS_GEMMDATASET_H
#define ACL_TESTS_DATASETS_GEMMDATASET_H

#include "arm_compute/core/TensorShape.h"

#include "utils/TypePrinter.h"

#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace datasets
{
class GEMMDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, TensorShape, float, float>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator a_it,
                 std::vector<TensorShape>::const_iterator b_it,
                 std::vector<TensorShape>::const_iterator c_it,
                 std::vector<TensorShape>::const_iterator dst_it,
                 std::vector<float>::const_iterator       alpha_it,
                 std::vector<float>::const_iterator       beta_it)
            : _a_it{a_it}, _b_it{b_it}, _c_it{c_it}, _dst_it{dst_it}, _alpha_it{alpha_it}, _beta_it{beta_it}
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "A=" << *_a_it << ":";
            description << "B=" << *_b_it << ":";
            description << "C=" << *_c_it << ":";
            description << "Out=" << *_dst_it << ":";
            description << "Alpha=" << *_alpha_it << ":";
            description << "Beta=" << *_beta_it;
            return description.str();
        }

        GEMMDataset::type operator*() const
        {
            return std::make_tuple(*_a_it, *_b_it, *_c_it, *_dst_it, *_alpha_it, *_beta_it);
        }

        iterator &operator++()
        {
            ++_a_it;
            ++_b_it;
            ++_c_it;
            ++_dst_it;
            ++_alpha_it;
            ++_beta_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _a_it;
        std::vector<TensorShape>::const_iterator _b_it;
        std::vector<TensorShape>::const_iterator _c_it;
        std::vector<TensorShape>::const_iterator _dst_it;
        std::vector<float>::const_iterator       _alpha_it;
        std::vector<float>::const_iterator       _beta_it;
    };

    iterator begin() const
    {
        return iterator(_a_shapes.begin(), _b_shapes.begin(), _c_shapes.begin(), _dst_shapes.begin(), _alpha.begin(),
                        _beta.begin());
    }

    int size() const
    {
        return std::min({
            _a_shapes.size(),
            _b_shapes.size(),
            _c_shapes.size(),
            _dst_shapes.size(),
            _alpha.size(),
            _beta.size(),
        });
    }

    void add_config(TensorShape a, TensorShape b, TensorShape c, TensorShape dst, float alpha, float beta)
    {
        _a_shapes.emplace_back(std::move(a));
        _b_shapes.emplace_back(std::move(b));
        _c_shapes.emplace_back(std::move(c));
        _dst_shapes.emplace_back(std::move(dst));
        _alpha.push_back(alpha);
        _beta.push_back(beta);
    }

    // Overload for the common case: A = M x K, B = K x N, C = M x N, Dst = M x N
    void add_config(size_t m, size_t n, size_t k, float alpha, float beta)
    {
        _a_shapes.emplace_back(k, m);
        _b_shapes.emplace_back(n, k);
        _c_shapes.emplace_back(n, m);
        _dst_shapes.emplace_back(n, m);
        _alpha.push_back(alpha);
        _beta.push_back(beta);
    }

private:
    std::vector<TensorShape> _a_shapes{};
    std::vector<TensorShape> _b_shapes{};
    std::vector<TensorShape> _c_shapes{};
    std::vector<TensorShape> _dst_shapes{};
    std::vector<float>       _alpha{};
    std::vector<float>       _beta{};
};
} // namespace datasets
} // namespace test
} // namespace arm_compute

#endif // ACL_TESTS_DATASETS_GEMMDATASET_H
