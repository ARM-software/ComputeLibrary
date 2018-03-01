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
    using type = std::tuple<TensorShape, PadStrideInfo, Size2D, bool>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator in_it, std::vector<PadStrideInfo>::const_iterator info_it, std::vector<Size2D>::const_iterator kernel_dims_it,
                 std::vector<bool>::const_iterator format_it)
            : _in_it{ std::move(in_it) }, _info_it{ std::move(info_it) }, _kernel_dims_it{ std::move(kernel_dims_it) }, _format_it{ std::move(format_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_in_it << ":";
            description << "Info=" << *_info_it;
            description << "KernelDims=" << *_kernel_dims_it;
            description << "IsNCHW=" << *_format_it;
            return description.str();
        }

        WinogradInputTransformDataset::type operator*() const
        {
            return std::make_tuple(*_in_it, *_info_it, *_kernel_dims_it, *_format_it);
        }

        iterator &operator++()
        {
            ++_in_it;
            ++_info_it;
            ++_kernel_dims_it;
            ++_format_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator   _in_it;
        std::vector<PadStrideInfo>::const_iterator _info_it;
        std::vector<Size2D>::const_iterator        _kernel_dims_it;
        std::vector<bool>::const_iterator          _format_it;
    };

    iterator begin() const
    {
        return iterator(_in_shapes.begin(), _infos.begin(), _kernel_dims.begin(), _format.begin());
    }

    int size() const
    {
        return std::min(_in_shapes.size(), std::min(_infos.size(), std::min(_kernel_dims.size(), _format.size())));
    }

    void add_config(TensorShape in, PadStrideInfo info, Size2D kernel_dims, bool format)
    {
        _in_shapes.emplace_back(std::move(in));
        _infos.emplace_back(std::move(info));
        _kernel_dims.emplace_back(std::move(kernel_dims));
        _format.emplace_back(std::move(format));
    }

protected:
    WinogradInputTransformDataset()                                 = default;
    WinogradInputTransformDataset(WinogradInputTransformDataset &&) = default;

private:
    std::vector<TensorShape>   _in_shapes{};
    std::vector<PadStrideInfo> _infos{};
    std::vector<Size2D>        _kernel_dims{};
    std::vector<bool>          _format{};
};

class SmallWinogradInputTransformDataset final : public WinogradInputTransformDataset
{
public:
    SmallWinogradInputTransformDataset()
    {
        add_config(TensorShape(9U, 9U), PadStrideInfo(1, 1, 1, 1), Size2D(3U, 3U), true);
        add_config(TensorShape(27U, 13U, 2U), PadStrideInfo(1, 1, 0, 0), Size2D(3U, 3U), true);
        add_config(TensorShape(128U, 64U, 1U, 3U), PadStrideInfo(1, 1, 1, 1), Size2D(3U, 3U), true);
        add_config(TensorShape(9U, 9U, 3U, 4U), PadStrideInfo(1, 1, 0, 0), Size2D(3U, 3U), true);
        add_config(TensorShape(27U, 13U, 2U, 4U), PadStrideInfo(1, 1, 1, 1), Size2D(3U, 3U), true);
        add_config(TensorShape(9U, 9U, 3U, 5U), PadStrideInfo(1, 1, 0, 0), Size2D(3U, 3U), true);
        add_config(TensorShape(14U, 14U, 512U, 2U), PadStrideInfo(1, 1, 1, 1), Size2D(3U, 3U), true);
    }
};

class LargeWinogradInputTransformDataset final : public WinogradInputTransformDataset
{
public:
    LargeWinogradInputTransformDataset()
    {
        add_config(TensorShape(42U, 37U, 8U, 15U), PadStrideInfo(1, 1, 1, 1), Size2D(3U, 3U), true);
        add_config(TensorShape(57U, 60U, 13U, 8U), PadStrideInfo(1, 1, 1, 1), Size2D(3U, 3U), true);
        add_config(TensorShape(128U, 64U, 21U, 13U), PadStrideInfo(1, 1, 0, 0), Size2D(3U, 3U), true);
        add_config(TensorShape(83U, 72U, 14U, 5U), PadStrideInfo(1, 1, 0, 0), Size2D(3U, 3U), true);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WINOGRAD_INPUT_TRANSFORM_DATASET */
