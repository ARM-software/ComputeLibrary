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
#ifndef ARM_COMPUTE_TEST_RNN_LAYER_DATASET
#define ARM_COMPUTE_TEST_RNN_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class RNNLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, TensorShape, TensorShape, ActivationLayerInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator         src_it,
                 std::vector<TensorShape>::const_iterator         weights_it,
                 std::vector<TensorShape>::const_iterator         recurrent_weights_it,
                 std::vector<TensorShape>::const_iterator         biases_it,
                 std::vector<TensorShape>::const_iterator         dst_it,
                 std::vector<ActivationLayerInfo>::const_iterator infos_it)
            : _src_it{ std::move(src_it) },
              _weights_it{ std::move(weights_it) },
              _recurrent_weights_it{ std::move(recurrent_weights_it) },
              _biases_it{ std::move(biases_it) },
              _dst_it{ std::move(dst_it) },
              _infos_it{ std::move(infos_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "Weights=" << *_weights_it << ":";
            description << "Biases=" << *_biases_it << ":";
            description << "Out=" << *_dst_it;
            return description.str();
        }

        RNNLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_weights_it, *_recurrent_weights_it, *_biases_it, *_dst_it, *_infos_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_weights_it;
            ++_recurrent_weights_it;
            ++_biases_it;
            ++_dst_it;
            ++_infos_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator         _src_it;
        std::vector<TensorShape>::const_iterator         _weights_it;
        std::vector<TensorShape>::const_iterator         _recurrent_weights_it;
        std::vector<TensorShape>::const_iterator         _biases_it;
        std::vector<TensorShape>::const_iterator         _dst_it;
        std::vector<ActivationLayerInfo>::const_iterator _infos_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _weight_shapes.begin(), _recurrent_weight_shapes.begin(), _bias_shapes.begin(), _dst_shapes.begin(), _infos.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_weight_shapes.size(), std::min(_recurrent_weight_shapes.size(), std::min(_bias_shapes.size(), std::min(_dst_shapes.size(), _infos.size())))));
    }

    void add_config(TensorShape src, TensorShape weights, TensorShape recurrent_weights, TensorShape biases, TensorShape dst, ActivationLayerInfo info)
    {
        _src_shapes.emplace_back(std::move(src));
        _weight_shapes.emplace_back(std::move(weights));
        _recurrent_weight_shapes.emplace_back(std::move(recurrent_weights));
        _bias_shapes.emplace_back(std::move(biases));
        _dst_shapes.emplace_back(std::move(dst));
        _infos.emplace_back(std::move(info));
    }

protected:
    RNNLayerDataset()                   = default;
    RNNLayerDataset(RNNLayerDataset &&) = default;

private:
    std::vector<TensorShape>         _src_shapes{};
    std::vector<TensorShape>         _weight_shapes{};
    std::vector<TensorShape>         _recurrent_weight_shapes{};
    std::vector<TensorShape>         _bias_shapes{};
    std::vector<TensorShape>         _dst_shapes{};
    std::vector<ActivationLayerInfo> _infos{};
};

class SmallRNNLayerDataset final : public RNNLayerDataset
{
public:
    SmallRNNLayerDataset()
    {
        add_config(TensorShape(128U, 16U), TensorShape(128U, 32U), TensorShape(32U, 32U), TensorShape(32U), TensorShape(32U, 16U), ActivationLayerInfo());
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_RNN_LAYER_DATASET */
