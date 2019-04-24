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

class SmallWinogradOutputTransformDatasetNCHW final : public WinogradOutputTransformDataset
{
public:
    SmallWinogradOutputTransformDatasetNCHW()
    {
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

        // (2x1, 3x1)
        add_config(TensorShape(13U, 18U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 44U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(1U, 891U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(53U, 33U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 30U, 4U, 3U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 98U, 4U, 2U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));

        // (1x2, 1x3)
        add_config(TensorShape(13U, 14U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 50U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(1U, 901U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 32U, 4U, 3U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 98U, 4U, 2U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));

        // (4x1, 3x1)
        add_config(TensorShape(13U, 12U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 22U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(1U, 462U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(53U, 33U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 20U, 6U, 3U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 56U, 6U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));

        // (1x4, 1x3)
        add_config(TensorShape(13U, 7U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 30U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(1U, 477U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 16U, 6U, 3U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 56U, 6U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));

        // (4x4, 5x5)
        add_config(TensorShape(13U, 1U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 4U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(5U, 104U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 2U, 64U, 3U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 9U, 64U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 2U, 64U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));

        // (4x1, 5x1)
        add_config(TensorShape(13U, 6U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 22U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(5U, 462U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(53U, 33U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 10U, 8U, 3U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 42U, 8U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 20U, 8U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(8U, 10U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));

        // (1x4, 1x5)
        add_config(TensorShape(13U, 7U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(7U, 20U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(5U, 477U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(7U, 16U, 8U, 3U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(24U, 42U, 8U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(7U, 24U, 8U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
    }
};

class SmallWinogradOutputTransformDatasetNHWC_F16 : public WinogradOutputTransformDataset
{
public:
    SmallWinogradOutputTransformDatasetNHWC_F16()
    {
        // (4x1, 3x1)
        add_config(TensorShape(13U, 12U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 22U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(1U, 462U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(53U, 33U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 20U, 6U, 3U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(24U, 56U, 6U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));

        // (1x4, 1x3)
        add_config(TensorShape(13U, 7U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 30U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(1U, 477U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(7U, 16U, 6U, 3U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(24U, 56U, 6U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));

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

        // (4x1, 5x1)
        add_config(TensorShape(13U, 6U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 22U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(5U, 462U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(53U, 33U), PadStrideInfo(1, 1, 2, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 10U, 8U, 3U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(24U, 42U, 8U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 20U, 8U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(8U, 10U), PadStrideInfo(1, 1, 2, 0), DataLayout::NHWC));

        // (1x4, 1x5)
        add_config(TensorShape(13U, 7U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(7U, 6U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 20U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(5U, 477U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 2), DataLayout::NHWC));
        add_config(TensorShape(7U, 16U, 8U, 3U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(24U, 42U, 8U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(7U, 24U, 8U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 2), DataLayout::NHWC));
    }
};

class SmallWinogradOutputTransformDatasetNHWC_F32 : public SmallWinogradOutputTransformDatasetNHWC_F16
{
public:
    SmallWinogradOutputTransformDatasetNHWC_F32()
        : SmallWinogradOutputTransformDatasetNHWC_F16()
    {
        // (2x2, 7x7)
        add_config(TensorShape(13U, 4U, 64U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 6U, 64U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(5U, 360U, 64U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(7U, 2U, 64U, 3U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(24U, 25U, 64U, 2U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NHWC));
        add_config(TensorShape(7U, 2U, 64U, 5U), WinogradInfo(Size2D(2U, 2U), Size2D(7U, 7U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));

        // (2x1, 7x1)
        add_config(TensorShape(13U, 18U, 8U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 22U, 8U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(5U, 858U, 8U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(53U, 33U), PadStrideInfo(1, 1, 2, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 10U, 8U, 3U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(24U, 70U, 8U, 2U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 30U, 8U, 5U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(8U, 10U), PadStrideInfo(1, 1, 2, 0), DataLayout::NHWC));

        // (1x2, 1x7)
        add_config(TensorShape(13U, 18U, 8U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(9U, 9U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(7U, 30U, 8U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(10U, 11U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(5U, 848U, 8U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 2), DataLayout::NHWC));
        add_config(TensorShape(7U, 16U, 8U, 3U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(24U, 70U, 8U, 2U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(14U, 14U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(7U, 32U, 8U, 5U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 2), DataLayout::NHWC));
    }
};

class LargeWinogradOutputTransformDatasetNCHW : public WinogradOutputTransformDataset
{
public:
    LargeWinogradOutputTransformDatasetNCHW()
    {
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

        // (2x1, 3x1)
        add_config(TensorShape(64U, 24976U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(224U, 223U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(32U, 6160U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(112U, 110U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 1568U, 4U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(64U, 24753U, 4U, 3U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(224U, 223U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(32U, 6050U, 4U, 2U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(112U, 110U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 1512U, 4U, 5U), WinogradInfo(Size2D(2U, 1U), Size2D(3U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));

        // (1x2, 1x3)
        add_config(TensorShape(64U, 25088U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(224U, 223U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(32U, 6160U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(112U, 110U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(13U, 1568U, 4U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(64U, 24864U, 4U, 3U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(224U, 223U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(32U, 6048U, 4U, 2U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(112U, 110U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 1512U, 4U, 5U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));

        // (4x1, 3x1)
        add_config(TensorShape(64U, 12488U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(224U, 223U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(32U, 3080U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(112U, 110U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 784U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(64U, 12488U, 6U, 3U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(224U, 223U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(32U, 3080U, 6U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(112U, 110U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 784U, 6U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));

        // (1x4, 1x3)
        add_config(TensorShape(64U, 12544U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(224U, 223U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(32U, 3136U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(112U, 110U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(13U, 784U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(64U, 12544U, 6U, 3U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(224U, 223U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(32U, 3024U, 6U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(112U, 110U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 784U, 6U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));

        // (4x4, 5x5)
        add_config(TensorShape(32U, 756U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 182U, 64U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(32U, 756U, 64U, 2U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 182U, 64U, 5U), WinogradInfo(Size2D(4U, 4U), Size2D(5U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));

        // (4x1, 5x1)
        add_config(TensorShape(32U, 3136U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(112U, 112U), PadStrideInfo(1, 1, 2, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 784U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));
        add_config(TensorShape(32U, 3024U, 8U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(112U, 112U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 784U, 8U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW));

        // (1x4, 1x5)
        add_config(TensorShape(32U, 3136U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 0, 2), DataLayout::NCHW));
        add_config(TensorShape(13U, 784U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
        add_config(TensorShape(32U, 3024U, 8U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW));
        add_config(TensorShape(13U, 784U, 8U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW));
    }
};

class LargeWinogradOutputTransformDatasetNHWC_F16 : public WinogradOutputTransformDataset
{
public:
    LargeWinogradOutputTransformDatasetNHWC_F16()
    {
        // (4x1, 3x1)
        add_config(TensorShape(64U, 12488U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(224U, 223U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(32U, 3080U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(112U, 110U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 784U, 6U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(64U, 12488U, 6U, 3U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(224U, 223U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(32U, 3080U, 6U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(112U, 110U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 784U, 6U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(3U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));

        // (1x4, 1x3)
        add_config(TensorShape(64U, 12544U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(224U, 223U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(32U, 3136U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(112U, 110U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(13U, 784U, 6U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(64U, 12544U, 6U, 3U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(224U, 223U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(32U, 3024U, 6U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(112U, 110U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 784U, 6U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 3U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));

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

        // (4x1, 5x1)
        add_config(TensorShape(32U, 3136U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(112U, 112U), PadStrideInfo(1, 1, 2, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 784U, 8U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(32U, 3024U, 8U, 2U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(112U, 112U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 784U, 8U, 5U), WinogradInfo(Size2D(4U, 1U), Size2D(5U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));

        // (1x4, 1x5)
        add_config(TensorShape(32U, 3136U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 0, 2), DataLayout::NHWC));
        add_config(TensorShape(13U, 784U, 8U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(32U, 3024U, 8U, 2U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(112U, 112U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 784U, 8U, 5U), WinogradInfo(Size2D(1U, 4U), Size2D(1U, 5U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
    }
};

class LargeWinogradOutputTransformDatasetNHWC_F32 : public LargeWinogradOutputTransformDatasetNHWC_F16
{
public:
    LargeWinogradOutputTransformDatasetNHWC_F32()
    {
        // (2x1, 7x1)
        add_config(TensorShape(32U, 6160U, 8U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(112U, 112U), PadStrideInfo(1, 1, 2, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 1456U, 8U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));
        add_config(TensorShape(32U, 5936U, 8U, 2U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(112U, 112U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 1456U, 8U, 5U), WinogradInfo(Size2D(2U, 1U), Size2D(7U, 1U), Size2D(56U, 56U), PadStrideInfo(1, 1, 1, 0), DataLayout::NHWC));

        // (1x2, 1x7)
        add_config(TensorShape(32U, 6160U, 8U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(112U, 112U), PadStrideInfo(1, 1, 0, 2), DataLayout::NHWC));
        add_config(TensorShape(13U, 1456U, 8U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
        add_config(TensorShape(32U, 5936U, 8U, 2U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(112U, 112U), PadStrideInfo(1, 1, 0, 0), DataLayout::NHWC));
        add_config(TensorShape(13U, 1456U, 8U, 5U), WinogradInfo(Size2D(1U, 2U), Size2D(1U, 7U), Size2D(56U, 56U), PadStrideInfo(1, 1, 0, 1), DataLayout::NHWC));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WINOGRAD_OUTPUT_TRANSFORM_DATASET */
