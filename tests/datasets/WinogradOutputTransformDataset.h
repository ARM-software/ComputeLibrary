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
#ifndef ARM_COMPUTE_TEST_WINOGRAD_OUTPUT_TRANSFORM_DATASET
#define ARM_COMPUTE_TEST_WINOGRAD_OUTPUT_TRANSFORM_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class WinogradOutputTransformDataset
{
public:
    using type = std::tuple<TensorShape, Size2D, Size2D, Size2D, DataLayout>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator a_it,
                 std::vector<Size2D>::const_iterator      b_it,
                 std::vector<Size2D>::const_iterator      c_it,
                 std::vector<Size2D>::const_iterator      d_it,
                 std::vector<DataLayout>::const_iterator  data_layout_it)
            : _a_it{ std::move(a_it) },
              _b_it{ std::move(b_it) },
              _c_it{ std::move(c_it) },
              _d_it{ std::move(d_it) },
              _data_layout_it{ std::move(data_layout_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Input=" << *_a_it << ":";
            description << "KernelDims=" << *_b_it << ":";
            description << "OutputDims=" << *_c_it << ":";
            description << "NumTiles=" << *_d_it << ":";
            description << "DataLayout=" << *_data_layout_it;
            return description.str();
        }

        WinogradOutputTransformDataset::type operator*() const
        {
            return std::make_tuple(*_a_it, *_b_it, *_c_it, *_d_it, *_data_layout_it);
        }

        iterator &operator++()
        {
            ++_a_it;
            ++_b_it;
            ++_c_it;
            ++_d_it;
            ++_data_layout_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _a_it;
        std::vector<Size2D>::const_iterator      _b_it;
        std::vector<Size2D>::const_iterator      _c_it;
        std::vector<Size2D>::const_iterator      _d_it;
        std::vector<DataLayout>::const_iterator  _data_layout_it;
    };

    iterator begin() const
    {
        return iterator(_a_shapes.begin(), _b_dims.begin(), _c_dims.begin(), _d_dims.begin(), _data_layout.begin());
    }

    int size() const
    {
        return std::min(_a_shapes.size(), std::min(_b_dims.size(), std::min(_c_dims.size(), std::min(_d_dims.size(), _data_layout.size()))));
    }

    void add_config(TensorShape a, Size2D b, Size2D c, Size2D d, DataLayout data_layout)
    {
        _a_shapes.emplace_back(std::move(a));
        _b_dims.emplace_back(std::move(b));
        _c_dims.emplace_back(std::move(c));
        _d_dims.emplace_back(std::move(d));
        _data_layout.emplace_back(std::move(data_layout));
    }

protected:
    WinogradOutputTransformDataset()                                  = default;
    WinogradOutputTransformDataset(WinogradOutputTransformDataset &&) = default;

private:
    std::vector<TensorShape> _a_shapes{};
    std::vector<Size2D>      _b_dims{};
    std::vector<Size2D>      _c_dims{};
    std::vector<Size2D>      _d_dims{};
    std::vector<DataLayout>  _data_layout{};
};

class SmallWinogradOutputTransformDataset final : public WinogradOutputTransformDataset
{
public:
    SmallWinogradOutputTransformDataset()
    {
        add_config(TensorShape(24U, 49U, 16U), Size2D(3, 3), Size2D(14U, 14U), Size2D(7U, 7U), DataLayout::NCHW);
        add_config(TensorShape(13U, 6U, 16U), Size2D(3, 3), Size2D(5U, 4U), Size2D(3U, 2U), DataLayout::NCHW);
        add_config(TensorShape(7U, 20U, 16U), Size2D(3, 3), Size2D(8U, 9U), Size2D(4U, 5U), DataLayout::NCHW);
        add_config(TensorShape(24U, 49U, 16U, 3U), Size2D(3, 3), Size2D(14U, 14U), Size2D(7U, 7U), DataLayout::NCHW);
        add_config(TensorShape(13U, 6U, 16U, 2U), Size2D(3, 3), Size2D(5U, 4U), Size2D(3U, 2U), DataLayout::NCHW);
        add_config(TensorShape(7U, 20U, 16U, 5U), Size2D(3, 3), Size2D(8U, 9U), Size2D(4U, 5U), DataLayout::NCHW);
    }
};

class LargeWinogradOutputTransformDataset final : public WinogradOutputTransformDataset
{
public:
    LargeWinogradOutputTransformDataset()
    {
        add_config(TensorShape(128U, 3136U, 16U), Size2D(3, 3), Size2D(112U, 112U), Size2D(56U, 56U), DataLayout::NCHW);
        add_config(TensorShape(256U, 784U, 16U), Size2D(3, 3), Size2D(55U, 55U), Size2D(28U, 28U), DataLayout::NCHW);
        add_config(TensorShape(512U, 169U, 16U), Size2D(3, 3), Size2D(26U, 26U), Size2D(13U, 13U), DataLayout::NCHW);
        add_config(TensorShape(128U, 3136U, 16U, 3U), Size2D(3, 3), Size2D(112U, 112U), Size2D(56U, 56U), DataLayout::NCHW);
        add_config(TensorShape(256U, 784U, 16U, 2U), Size2D(3, 3), Size2D(55U, 55U), Size2D(28U, 28U), DataLayout::NCHW);
        add_config(TensorShape(512U, 169U, 16U, 5U), Size2D(3, 3), Size2D(26U, 26U), Size2D(13U, 13U), DataLayout::NCHW);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WINOGRAD_OUTPUT_TRANSFORM_DATASET */
