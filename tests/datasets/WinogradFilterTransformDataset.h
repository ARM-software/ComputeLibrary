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
#ifndef ARM_COMPUTE_TEST_WINOGRAD_FILTER_TRANSFORM_DATASET
#define ARM_COMPUTE_TEST_WINOGRAD_FILTER_TRANSFORM_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class WinogradFilterTransformDataset
{
public:
    using type = std::tuple<TensorShape, bool>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator a_it,
                 std::vector<bool>::const_iterator        is_nchw_it)
            : _a_it{ std::move(a_it) },
              _is_nchw_it{ std::move(is_nchw_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Input=" << *_a_it << ":";
            description << "IsNCHW=" << *_is_nchw_it << ":";
            return description.str();
        }

        WinogradFilterTransformDataset::type operator*() const
        {
            return std::make_tuple(*_a_it, *_is_nchw_it);
        }

        iterator &operator++()
        {
            ++_a_it;
            ++_is_nchw_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _a_it;
        std::vector<bool>::const_iterator        _is_nchw_it;
    };

    iterator begin() const
    {
        return iterator(_a_shapes.begin(), _is_nchw.begin());
    }

    int size() const
    {
        return std::min(_a_shapes.size(), _is_nchw.size());
    }

    void add_config(TensorShape a, bool is_nchw)
    {
        _a_shapes.emplace_back(std::move(a));
        _is_nchw.emplace_back(std::move(is_nchw));
    }

protected:
    WinogradFilterTransformDataset()                                  = default;
    WinogradFilterTransformDataset(WinogradFilterTransformDataset &&) = default;

private:
    std::vector<TensorShape> _a_shapes{};
    std::vector<bool>        _is_nchw{};
};

class SmallWinogradFilterTransformDataset final : public WinogradFilterTransformDataset
{
public:
    SmallWinogradFilterTransformDataset()
    {
        add_config(TensorShape(3U, 3U, 7U, 4U), true);
        add_config(TensorShape(3U, 3U, 4U, 13U), true);
        add_config(TensorShape(3U, 3U, 9U, 2U), true);
        add_config(TensorShape(3U, 3U, 3U, 5U), true);
    }
};

class LargeWinogradFilterTransformDataset final : public WinogradFilterTransformDataset
{
public:
    LargeWinogradFilterTransformDataset()
    {
        add_config(TensorShape(3U, 3U, 32U, 64U), true);
        add_config(TensorShape(3U, 3U, 51U, 13U), true);
        add_config(TensorShape(3U, 3U, 53U, 47U), true);
        add_config(TensorShape(3U, 3U, 128U, 384U), true);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WINOGRAD_FILTER_TRANSFORM_DATASET */
