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
    using type = std::tuple<TensorShape, WinogradInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator  a_it,
                 std::vector<WinogradInfo>::const_iterator info_it)
            : _a_it{ std::move(a_it) },
              _info_it{ std::move(info_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Input=" << *_a_it << ":";
            description << "WinogradInfo=" << *_info_it << ":";
            return description.str();
        }

        WinogradOutputTransformDataset::type operator*() const
        {
            return std::make_tuple(*_a_it, *_info_it);
        }

        iterator &operator++()
        {
            ++_a_it;
            ++_info_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator  _a_it;
        std::vector<WinogradInfo>::const_iterator _info_it;
    };

    iterator begin() const
    {
        return iterator(_a_shapes.begin(), _info.begin());
    }

    int size() const
    {
        return std::min(_a_shapes.size(), _info.size());
    }

    void add_config(TensorShape a, WinogradInfo b)
    {
        _a_shapes.emplace_back(std::move(a));
        _info.emplace_back(std::move(b));
    }

protected:
    WinogradOutputTransformDataset()                                  = default;
    WinogradOutputTransformDataset(WinogradOutputTransformDataset &&) = default;

private:
    std::vector<TensorShape>  _a_shapes{};
    std::vector<WinogradInfo> _info{};
};

class SmallWinogradOutputTransformDataset final : public WinogradOutputTransformDataset
{
public:
    SmallWinogradOutputTransformDataset()
    {
        // NCHW
        // (2x2, 3x3)
        add_config(TensorShape(13U, 6U, 16U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 20U, 16U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(1U, 442U, 16U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 12U, 16U, 3U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 49U, 16U, 2U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));

        // (4x4, 3x3)
        add_config(TensorShape(13U, 4U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(10U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 6U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 117U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 4U, 36U, 3U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 16U, 36U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 12U, 16U, 5U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));

        // (4x4, 5x5)
        add_config(TensorShape(13U, 1U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 4U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(5U, 104U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 2U, 64U, 3U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 9U, 64U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 2U, 64U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));

        // NHWC
        // (4x4, 3x3)
        add_config(TensorShape(13U, 4U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(10U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 6U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 117U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(7U, 4U, 36U, 3U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(24U, 16U, 36U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NHWC));

        // (4x4, 5x5)
        add_config(TensorShape(13U, 1U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 4U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(5U, 104U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(7U, 2U, 64U, 3U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(24U, 9U, 64U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NHWC));
        add_config(TensorShape(7U, 2U, 64U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
    }
};

class LargeWinogradOutputTransformDataset final : public WinogradOutputTransformDataset
{
public:
    LargeWinogradOutputTransformDataset()
    {
        // NCHW
        // (2x2, 3x3)
        add_config(TensorShape(64U, 12544U, 16U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(224U, 224U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(32U, 3080U, 16U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 756U, 16U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(64U, 12544U, 16U, 3U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(224U, 224U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(32U, 3080U, 16U, 2U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 756U, 16U, 5U), WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));

        // (4x4, 3x3)
        add_config(TensorShape(64U, 3136U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(224U, 224U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(32U, 784U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 196U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(64U, 3136U, 36U, 3U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(224U, 224U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(32U, 784U, 36U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 196U, 36U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));

        // (4x4, 5x5)
        add_config(TensorShape(32U, 756U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 182U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(32U, 756U, 64U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 182U, 64U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));

        // NHWC
        // (4x4, 3x3)
        add_config(TensorShape(64U, 3136U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(224U, 224U), PadStrideInfo(1, 1, 1, 1), DataLayout::NHWC));
        add_config(TensorShape(32U, 784U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 196U, 36U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(64U, 3136U, 36U, 3U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(224U, 224U), PadStrideInfo(1, 1, 1, 1), DataLayout::NHWC));
        add_config(TensorShape(32U, 784U, 36U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 196U, 36U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));

        // (4x4, 5x5)
        add_config(TensorShape(32U, 756U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 182U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(32U, 756U, 64U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 182U, 64U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WINOGRAD_OUTPUT_TRANSFORM_DATASET */
