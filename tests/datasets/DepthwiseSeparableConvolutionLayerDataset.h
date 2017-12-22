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
#ifndef ARM_COMPUTE_TEST_DEPTHWISE_SEPARABLE_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_DEPTHWISE_SEPARABLE_CONVOLUTION_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class DepthwiseSeparableConvolutionLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, TensorShape, TensorShape, TensorShape, TensorShape, PadStrideInfo, PadStrideInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator   src_it,
                 std::vector<TensorShape>::const_iterator   filter_it,
                 std::vector<TensorShape>::const_iterator   filter_biases_it,
                 std::vector<TensorShape>::const_iterator   depthwise_out_it,
                 std::vector<TensorShape>::const_iterator   weights_it,
                 std::vector<TensorShape>::const_iterator   biases_it,
                 std::vector<TensorShape>::const_iterator   dst_it,
                 std::vector<PadStrideInfo>::const_iterator depthwise_infos_it,
                 std::vector<PadStrideInfo>::const_iterator pointwise_infos_it)
            : _src_it{ std::move(src_it) },
              _filter_it{ std::move(filter_it) },
              _filter_biases_it{ std::move(filter_biases_it) },
              _depthwise_out_it{ std::move(depthwise_out_it) },
              _weights_it{ std::move(weights_it) },
              _biases_it{ std::move(biases_it) },
              _dst_it{ std::move(dst_it) },
              _depthwise_infos_it{ std::move(depthwise_infos_it) },
              _pointwise_infos_it{ std::move(pointwise_infos_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "Filter=" << *_filter_it << ":";
            description << "FilterBiases=" << *_filter_biases_it << ":";
            description << "DepthwiseOut=" << *_depthwise_out_it << ":";
            description << "Weights=" << *_weights_it << ":";
            description << "Biases=" << *_biases_it << ":";
            description << "Out=" << *_dst_it << ":";
            description << "DepthwiseInfo=" << *_depthwise_infos_it << ":";
            description << "PointwiseInfo=" << *_pointwise_infos_it;
            return description.str();
        }

        DepthwiseSeparableConvolutionLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_filter_it, *_filter_biases_it, *_depthwise_out_it, *_weights_it, *_biases_it, *_dst_it, *_depthwise_infos_it, *_pointwise_infos_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_filter_it;
            ++_filter_biases_it;
            ++_depthwise_out_it;
            ++_weights_it;
            ++_biases_it;
            ++_dst_it;
            ++_depthwise_infos_it;
            ++_pointwise_infos_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator   _src_it;
        std::vector<TensorShape>::const_iterator   _filter_it;
        std::vector<TensorShape>::const_iterator   _filter_biases_it;
        std::vector<TensorShape>::const_iterator   _depthwise_out_it;
        std::vector<TensorShape>::const_iterator   _weights_it;
        std::vector<TensorShape>::const_iterator   _biases_it;
        std::vector<TensorShape>::const_iterator   _dst_it;
        std::vector<PadStrideInfo>::const_iterator _depthwise_infos_it;
        std::vector<PadStrideInfo>::const_iterator _pointwise_infos_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _filter_shapes.begin(), _filter_biases_shapes.begin(), _depthwise_out_shapes.begin(), _weight_shapes.begin(), _bias_shapes.begin(), _dst_shapes.begin(),
                        _depthwise_infos.begin(),
                        _pointwise_infos.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_filter_shapes.size(), std::min(_filter_biases_shapes.size(), std::min(_depthwise_out_shapes.size(), std::min(_weight_shapes.size(),
                                                                                                                            std::min(_bias_shapes.size(), std::min(_dst_shapes.size(),
                                                                                                                                    std::min(_depthwise_infos.size(), _pointwise_infos.size()))))))));
    }

    void add_config(TensorShape src, TensorShape filter, TensorShape filter_bias, TensorShape depthwise_out, TensorShape weights, TensorShape biases, TensorShape dst, PadStrideInfo depthwise_info,
                    PadStrideInfo pointwise_info)
    {
        _src_shapes.emplace_back(std::move(src));
        _filter_shapes.emplace_back(std::move(filter));
        _filter_biases_shapes.emplace_back(std::move(filter_bias));
        _depthwise_out_shapes.emplace_back(std::move(depthwise_out));
        _weight_shapes.emplace_back(std::move(weights));
        _bias_shapes.emplace_back(std::move(biases));
        _dst_shapes.emplace_back(std::move(dst));
        _depthwise_infos.emplace_back(std::move(depthwise_info));
        _pointwise_infos.emplace_back(std::move(pointwise_info));
    }

protected:
    DepthwiseSeparableConvolutionLayerDataset()                                             = default;
    DepthwiseSeparableConvolutionLayerDataset(DepthwiseSeparableConvolutionLayerDataset &&) = default;

private:
    std::vector<TensorShape>   _src_shapes{};
    std::vector<TensorShape>   _filter_shapes{};
    std::vector<TensorShape>   _filter_biases_shapes{};
    std::vector<TensorShape>   _depthwise_out_shapes{};
    std::vector<TensorShape>   _weight_shapes{};
    std::vector<TensorShape>   _bias_shapes{};
    std::vector<TensorShape>   _dst_shapes{};
    std::vector<PadStrideInfo> _depthwise_infos{};
    std::vector<PadStrideInfo> _pointwise_infos{};
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISE_SEPARABLE_CONVOLUTION_LAYER_DATASET */
