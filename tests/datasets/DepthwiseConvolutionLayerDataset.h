/*
 * Copyright (c) 2017-2019 ARM Limited.
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
    using type = std::tuple<TensorShape, Size2D, PadStrideInfo, Size2D>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator   src_it,
                 std::vector<Size2D>::const_iterator        weights_it,
                 std::vector<PadStrideInfo>::const_iterator infos_it,
                 std::vector<Size2D>::const_iterator        dilation_it)
            : _src_it{ std::move(src_it) },
              _weights_it{ std::move(weights_it) },
              _infos_it{ std::move(infos_it) },
              _dilation_it{ std::move(dilation_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "Weights=" << *_weights_it << ":";
            description << "Info=" << *_infos_it << ":";
            description << "Dilation=" << *_dilation_it;
            return description.str();
        }

        DepthwiseConvolutionLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_weights_it, *_infos_it, *_dilation_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_weights_it;
            ++_infos_it;
            ++_dilation_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator   _src_it;
        std::vector<Size2D>::const_iterator        _weights_it;
        std::vector<PadStrideInfo>::const_iterator _infos_it;
        std::vector<Size2D>::const_iterator        _dilation_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _weight_shapes.begin(), _infos.begin(), _dilations.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_weight_shapes.size(), std::min(_infos.size(), _dilations.size())));
    }

    void add_config(TensorShape src, Size2D weights, PadStrideInfo info, Size2D dilation = Size2D(1U, 1U))
    {
        _src_shapes.emplace_back(std::move(src));
        _weight_shapes.emplace_back(std::move(weights));
        _infos.emplace_back(std::move(info));
        _dilations.emplace_back(std::move(dilation));
    }

protected:
    DepthwiseConvolutionLayerDataset()                                    = default;
    DepthwiseConvolutionLayerDataset(DepthwiseConvolutionLayerDataset &&) = default;

private:
    std::vector<TensorShape>   _src_shapes{};
    std::vector<Size2D>        _weight_shapes{};
    std::vector<PadStrideInfo> _infos{};
    std::vector<Size2D>        _dilations{};
};

/** Dataset containing small, generic depthwise convolution shapes. */
class SmallDepthwiseConvolutionLayerDataset final : public DepthwiseConvolutionLayerDataset
{
public:
    SmallDepthwiseConvolutionLayerDataset()
    {
        add_config(TensorShape(7U, 7U, 1U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(23U, 27U, 5U), Size2D(3U, 5U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(33U, 27U, 7U), Size2D(7U, 3U), PadStrideInfo(3, 2, 1, 0));
        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 1, 1, 2, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 1, 1, 0, 2, DimensionRoundingType::FLOOR));
    }
};

/** Dataset containing large, generic depthwise convolution shapes. */
class LargeDepthwiseConvolutionLayerDataset final : public DepthwiseConvolutionLayerDataset
{
public:
    LargeDepthwiseConvolutionLayerDataset()
    {
        add_config(TensorShape(33U, 27U, 11U), Size2D(3U, 3U), PadStrideInfo(1, 2, 0, 1));
        add_config(TensorShape(17U, 31U, 2U), Size2D(5U, 9U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(23U, 27U, 5U), Size2D(11U, 3U), PadStrideInfo(1, 2, 0, 0));
        add_config(TensorShape(17U, 31U, 2U, 3U), Size2D(5U, 9U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(233U, 277U, 55U), Size2D(3U, 3U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(333U, 277U, 77U), Size2D(3U, 3U), PadStrideInfo(3, 2, 1, 0));
        add_config(TensorShape(177U, 311U, 22U), Size2D(3U, 3U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(233U, 277U, 55U), Size2D(3U, 3U), PadStrideInfo(1, 2, 0, 0));
        add_config(TensorShape(333U, 277U, 77U), Size2D(3U, 3U), PadStrideInfo(2, 3, 0, 1));
        add_config(TensorShape(177U, 311U, 22U), Size2D(3U, 3U), PadStrideInfo(2, 1, 1, 1));
        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 2, 1, 2, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 1, 3, 0, 2, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 1, 0, 1, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(33U, 27U, 7U), Size2D(5U, 7U), PadStrideInfo(3, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR));
    }
};

/** Dataset containing small, 3x3 depthwise convolution shapes. */
class SmallDepthwiseConvolutionLayerDataset3x3 final : public DepthwiseConvolutionLayerDataset
{
public:
    SmallDepthwiseConvolutionLayerDataset3x3()
    {
        add_config(TensorShape(3U, 3U, 2U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(7U, 7U, 3U, 2U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 0));
        add_config(TensorShape(21U, 31U, 9U, 4U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 0));
        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 11U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR));
    }
};

class SmallDepthwiseConvolutionLayerDataset3x3NCHW final : public DepthwiseConvolutionLayerDataset
{
public:
    SmallDepthwiseConvolutionLayerDataset3x3NCHW()
    {
        add_config(TensorShape(33U, 27U, 11U), Size2D(3U, 3U), PadStrideInfo(3, 2, 1, 1));
        // Asymmetric padding
        add_config(TensorShape(33U, 27U, 11U), Size2D(3U, 3U), PadStrideInfo(2, 2, 3, 1, 2, 1, DimensionRoundingType::FLOOR));
    }
};

/** Dataset containing large, 3x3 depthwise convolution shapes. */
class LargeDepthwiseConvolutionLayerDataset3x3 final : public DepthwiseConvolutionLayerDataset
{
public:
    LargeDepthwiseConvolutionLayerDataset3x3()
    {
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 1));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 1));
        add_config(TensorShape(21U, 31U, 9U, 4U), Size2D(3U, 3U), PadStrideInfo(1, 2, 1, 0));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(1, 2, 0, 1));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(21U, 31U, 9U, 4U), Size2D(3U, 3U), PadStrideInfo(2, 1, 1, 0));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 1, 0, 1));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 1, 1, 1));
        add_config(TensorShape(21U, 31U, 9U, 4U), Size2D(3U, 3U), PadStrideInfo(2, 2, 1, 0));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 1));
        add_config(TensorShape(33U, 27U, 11U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 2, 1, 1));
        add_config(TensorShape(233U, 277U, 55U, 3U), Size2D(3U, 3U), PadStrideInfo(2, 1, 0, 0));
        add_config(TensorShape(177U, 311U, 22U), Size2D(3U, 3U), PadStrideInfo(1, 2, 1, 1));
        add_config(TensorShape(233U, 277U, 55U), Size2D(3U, 3U), PadStrideInfo(1, 2, 0, 0));
        add_config(TensorShape(333U, 277U, 77U, 5U), Size2D(3U, 3U), PadStrideInfo(2, 3, 0, 1));
        add_config(TensorShape(177U, 311U, 22U), Size2D(3U, 3U), PadStrideInfo(2, 1, 1, 1));
    }
};

/** Dataset containing optimized, 3x3 depthwise convolution shapes. */
class SmallOptimizedDepthwiseConvolutionLayerDataset3x3 final : public DepthwiseConvolutionLayerDataset
{
public:
    SmallOptimizedDepthwiseConvolutionLayerDataset3x3()
    {
        // Stride 1
        add_config(TensorShape(7U, 7U, 16U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL));
        add_config(TensorShape(7U, 7U, 16U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL));
        // Stride 2
        add_config(TensorShape(7U, 7U, 32U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL));
        add_config(TensorShape(7U, 7U, 32U), Size2D(3U, 3U), PadStrideInfo(2, 2, 1, 1, 1, 1, DimensionRoundingType::CEIL));
    }
};
/** Dataset containing optimized, 3x3 depthwise convolution shapes. */
class LargeOptimizedDepthwiseConvolutionLayerDataset3x3 final : public DepthwiseConvolutionLayerDataset
{
public:
    LargeOptimizedDepthwiseConvolutionLayerDataset3x3()
    {
        // Stride 1
        add_config(TensorShape(233U, 277U, 16U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL));
        add_config(TensorShape(233U, 7U, 16U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL));
        add_config(TensorShape(7U, 7U, 21U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL));
        add_config(TensorShape(28U, 28U, 16U), Size2D(3U, 3U), PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL));
        add_config(TensorShape(28U, 28U, 16U), Size2D(3U, 3U), PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL));
        // Stride 2
        add_config(TensorShape(233U, 277U, 32U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL));
        add_config(TensorShape(233U, 277U, 32U), Size2D(3U, 3U), PadStrideInfo(2, 2, 1, 1, 1, 1, DimensionRoundingType::CEIL));
        add_config(TensorShape(8U, 8U, 32U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::FLOOR));
        add_config(TensorShape(8U, 8U, 32U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL));
        add_config(TensorShape(8U, 8U, 33U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL));
        add_config(TensorShape(64U, 64U, 128U), Size2D(3U, 3U), PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_DATASET */
