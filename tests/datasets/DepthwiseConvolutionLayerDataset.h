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
#ifndef ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_DATASET
#define ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class DepthwiseConvolutionLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, PadStrideInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator   src_it,
                 std::vector<TensorShape>::const_iterator   weights_it,
                 std::vector<TensorShape>::const_iterator   dst_it,
                 std::vector<PadStrideInfo>::const_iterator infos_it)
            : _src_it{ std::move(src_it) },
              _weights_it{ std::move(weights_it) },
              _dst_it{ std::move(dst_it) },
              _infos_it{ std::move(infos_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "Weights=" << *_weights_it << ":";
            description << "Out=" << *_dst_it << ":";
            description << "Info=" << *_infos_it;
            return description.str();
        }

        DepthwiseConvolutionLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_weights_it, *_dst_it, *_infos_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_weights_it;
            ++_dst_it;
            ++_infos_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator   _src_it;
        std::vector<TensorShape>::const_iterator   _weights_it;
        std::vector<TensorShape>::const_iterator   _dst_it;
        std::vector<PadStrideInfo>::const_iterator _infos_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _weight_shapes.begin(), _dst_shapes.begin(), _infos.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_weight_shapes.size(), std::min(_dst_shapes.size(), _infos.size())));
    }

    void add_config(TensorShape src, TensorShape weights, TensorShape dst, PadStrideInfo info)
    {
        _src_shapes.emplace_back(std::move(src));
        _weight_shapes.emplace_back(std::move(weights));
        _dst_shapes.emplace_back(std::move(dst));
        _infos.emplace_back(std::move(info));
    }

protected:
    DepthwiseConvolutionLayerDataset()                                    = default;
    DepthwiseConvolutionLayerDataset(DepthwiseConvolutionLayerDataset &&) = default;

private:
    std::vector<TensorShape>   _src_shapes{};
    std::vector<TensorShape>   _weight_shapes{};
    std::vector<TensorShape>   _dst_shapes{};
    std::vector<PadStrideInfo> _infos{};
};
class SmallDepthwiseConvolutionLayerDataset final : public DepthwiseConvolutionLayerDataset
{
public:
    SmallDepthwiseConvolutionLayerDataset()
    {
        add_config(TensorShape(7U, 7U, 3U), TensorShape(3U, 3U, 3U), TensorShape(5U, 5U, 3U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(23U, 27U, 5U), TensorShape(3U, 5U, 5U), TensorShape(11U, 23U, 5U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(7U, 3U, 7U), TensorShape(10U, 13U, 7U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(33U, 27U, 11U), TensorShape(3U, 3U, 11U), TensorShape(31U, 14U, 11U), PadStrideInfo(1, 2, 0, 1));
        add_config(TensorShape(17U, 31U, 2U), TensorShape(5U, 9U, 2U), TensorShape(15U, 13U, 2U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(23U, 27U, 5U), TensorShape(11U, 3U, 5U), TensorShape(13U, 13U, 5U), PadStrideInfo(1, 2, 0, 0));
        add_config(TensorShape(17U, 31U, 2U, 3U), TensorShape(5U, 9U, 2U), TensorShape(15U, 13U, 2U, 3U), PadStrideInfo(1, 2, 1, 1));
        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 7U, 7U), TensorShape(11U, 12U, 7U), PadStrideInfo(3, 2, 1, 1, 2, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 7U, 7U), TensorShape(11U, 12U, 7U), PadStrideInfo(3, 2, 1, 1, 0, 2, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 7U, 7U), TensorShape(11U, 12U, 7U), PadStrideInfo(3, 2, 2, 1, 2, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 7U, 7U), TensorShape(11U, 12U, 7U), PadStrideInfo(3, 2, 1, 3, 0, 2, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 7U, 7U), TensorShape(10U, 11U, 7U), PadStrideInfo(3, 2, 1, 0, 1, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U), TensorShape(5U, 7U, 7U), TensorShape(10U, 11U, 7U), PadStrideInfo(3, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR));
    }
};

class LargeDepthwiseConvolutionLayerDataset final : public DepthwiseConvolutionLayerDataset
{
public:
    LargeDepthwiseConvolutionLayerDataset()
    {
        add_config(TensorShape(233U, 277U, 55U), TensorShape(3U, 3U, 55U), TensorShape(116U, 275U, 55U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(333U, 277U, 77U), TensorShape(3U, 3U, 77U), TensorShape(111U, 138U, 77U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(177U, 311U, 22U), TensorShape(3U, 3U, 22U), TensorShape(177U, 156U, 22U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(233U, 277U, 55U), TensorShape(3U, 3U, 55U), TensorShape(231U, 138U, 55U), PadStrideInfo(1, 2, 0, 0));
        add_config(TensorShape(333U, 277U, 77U), TensorShape(3U, 3U, 77U), TensorShape(166U, 93U, 77U), PadStrideInfo(2, 3, 0, 1));
        add_config(TensorShape(177U, 311U, 22U), TensorShape(3U, 3U, 22U), TensorShape(89U, 311U, 22U), PadStrideInfo(2, 1, 1, 1));
    }
};

class SmallDepthwiseConvolutionLayerDataset3x3 final : public DepthwiseConvolutionLayerDataset
{
public:
    SmallDepthwiseConvolutionLayerDataset3x3()
    {
        add_config(TensorShape(7U, 7U, 3U, 2U), TensorShape(3U, 3U, 3U), TensorShape(5U, 5U, 3U, 2U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 11U), TensorShape(3U, 3U, 11U), TensorShape(11U, 14U, 11U), PadStrideInfo(3, 2, 1, 1));
        add_config(TensorShape(21U, 31U, 9U, 4U), TensorShape(3U, 3U, 9U), TensorShape(21U, 15U, 9U, 4U), PadStrideInfo(1, 2, 1, 0));
        add_config(TensorShape(33U, 27U, 11U, 3U), TensorShape(3U, 3U, 11U), TensorShape(31U, 14U, 11U, 3U), PadStrideInfo(1, 2, 0, 1));
    }
};

class LargeDepthwiseConvolutionLayerDataset3x3 final : public DepthwiseConvolutionLayerDataset
{
public:
    LargeDepthwiseConvolutionLayerDataset3x3()
    {
        add_config(TensorShape(233U, 277U, 55U, 3U), TensorShape(3U, 3U, 55U), TensorShape(116U, 275U, 55U, 3U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(333U, 277U, 77U), TensorShape(3U, 3U, 77U), TensorShape(111U, 138U, 77U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(177U, 311U, 22U), TensorShape(3U, 3U, 22U), TensorShape(177U, 156U, 22U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(233U, 277U, 55U), TensorShape(3U, 3U, 55U), TensorShape(231U, 138U, 55U), PadStrideInfo(1, 2, 0, 0));
        add_config(TensorShape(333U, 277U, 77U, 5U), TensorShape(3U, 3U, 77U), TensorShape(166U, 93U, 77U, 5U), PadStrideInfo(2, 3, 0, 1));
        add_config(TensorShape(177U, 311U, 22U), TensorShape(3U, 3U, 22U), TensorShape(89U, 311U, 22U), PadStrideInfo(2, 1, 1, 1));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_DATASET */
