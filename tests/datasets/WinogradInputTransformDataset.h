/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_WINOGRAD_INPUT_TRANSFORM_DATASET
#define ARM_COMPUTE_TEST_WINOGRAD_INPUT_TRANSFORM_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class WinogradInputTransformDataset
{
public:
    using type = std::tuple<TensorShape, WinogradInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator in_it, std::vector<WinogradInfo>::const_iterator info_it)
            : _in_it{ std::move(in_it) }, _info_it{ std::move(info_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_in_it << ":";
            description << "WinogradInfo=" << *_info_it;
            return description.str();
        }

        WinogradInputTransformDataset::type operator*() const
        {
            return std::make_tuple(*_in_it, *_info_it);
        }

        iterator &operator++()
        {
            ++_in_it;
            ++_info_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator  _in_it;
        std::vector<WinogradInfo>::const_iterator _info_it;
    };

    iterator begin() const
    {
        return iterator(_in_shapes.begin(), _infos.begin());
    }

    int size() const
    {
        return std::min(_in_shapes.size(), _infos.size());
    }

    void add_config(TensorShape in, WinogradInfo info)
    {
        _in_shapes.emplace_back(std::move(in));
        _infos.emplace_back(std::move(info));
    }

protected:
    WinogradInputTransformDataset()                                 = default;
    WinogradInputTransformDataset(WinogradInputTransformDataset &&) = default;

private:
    std::vector<TensorShape>  _in_shapes{};
    std::vector<WinogradInfo> _infos{};
};

class SmallWinogradInputTransformDataset2x2_3x3 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset2x2_3x3()
    {
        add_config(TensorShape(9U, 9U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(9U, 9U, 3U, 4U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset2x1_3x1 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset2x1_3x1()
    {
        add_config(TensorShape(9U, 9U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(9U, 9U, 3U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset1x2_1x3 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset1x2_1x3()
    {
        add_config(TensorShape(9U, 9U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(9U, 9U, 3U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset4x4_3x3 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset4x4_3x3()
    {
        add_config(TensorShape(9U, 9U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(9U, 9U, 3U, 4U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset4x1_3x1 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset4x1_3x1()
    {
        add_config(TensorShape(9U, 9U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(9U, 9U, 3U, 4U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset1x4_1x3 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset1x4_1x3()
    {
        add_config(TensorShape(9U, 9U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(9U, 9U, 3U, 4U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset4x4_5x5 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset4x4_5x5()
    {
        add_config(TensorShape(9U, 9U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(9U, 9U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(9U, 9U, 3U, 4U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset4x1_5x1 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset4x1_5x1()
    {
        add_config(TensorShape(9U, 9U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(9U, 9U, 3U, 4U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset1x4_1x5 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset1x4_1x5()
    {
        add_config(TensorShape(9U, 9U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(9U, 9U, 3U, 4U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset2x2_7x7 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset2x2_7x7()
    {
        add_config(TensorShape(27U, 13U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(9U, 9U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 3U, 4U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset2x1_7x1 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset2x1_7x1()
    {
        add_config(TensorShape(23U, 31U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 31U, 3U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));
    }
};

class SmallWinogradInputTransformDataset1x2_1x7 final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset1x2_1x7()
    {
        add_config(TensorShape(23U, 31U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 31U, 3U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset2x2_3x3 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset2x2_3x3()
    {
        add_config(TensorShape(9U, 9U, 3U, 5U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(128U, 64U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(42U, 37U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(57U, 60U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset2x1_3x1 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset2x1_3x1()
    {
        add_config(TensorShape(9U, 9U, 3U, 5U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(128U, 64U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(42U, 37U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(57U, 60U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset1x2_1x3 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset1x2_1x3()
    {
        add_config(TensorShape(9U, 9U, 3U, 5U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(128U, 64U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(42U, 37U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(57U, 60U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset4x4_3x3 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset4x4_3x3()
    {
        add_config(TensorShape(9U, 9U, 3U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(128U, 64U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(42U, 37U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(57U, 60U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset4x1_3x1 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset4x1_3x1()
    {
        add_config(TensorShape(9U, 9U, 3U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(128U, 64U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(42U, 37U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(57U, 60U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset1x4_1x3 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset1x4_1x3()
    {
        add_config(TensorShape(9U, 9U, 3U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(128U, 64U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(42U, 37U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(57U, 60U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset4x4_5x5 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset4x4_5x5()
    {
        add_config(TensorShape(9U, 9U, 3U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(128U, 64U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(42U, 37U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(57U, 60U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset4x1_5x1 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset4x1_5x1()
    {
        add_config(TensorShape(9U, 9U, 3U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(27U, 13U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(128U, 64U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(42U, 37U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(57U, 60U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(83U, 72U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset1x4_1x5 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset1x4_1x5()
    {
        add_config(TensorShape(9U, 9U, 3U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(42U, 37U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(57U, 60U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset1x2_1x7 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset1x2_1x7()
    {
        add_config(TensorShape(23U, 31U, 3U, 5U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(42U, 37U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(57U, 60U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset2x1_7x1 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset2x1_7x1()
    {
        add_config(TensorShape(23U, 31U, 3U, 5U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(42U, 37U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(57U, 60U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
    }
};

class LargeWinogradInputTransformDataset2x2_7x7 final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset2x2_7x7()
    {
        add_config(TensorShape(27U, 13U, 3U, 5U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 1U, 3U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(27U, 13U, 2U, 4U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(27U, 13U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(14U, 14U, 512U, 2U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(42U, 37U, 8U, 15U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(42U, 37U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(57U, 60U, 13U, 8U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(57U, 60U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(128U, 64U, 21U, 13U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(128U, 64U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(83U, 72U, 14U, 5U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(83U, 72U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WINOGRAD_INPUT_TRANSFORM_DATASET */
