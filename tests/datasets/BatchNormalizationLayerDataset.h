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
#ifndef ARM_COMPUTE_TEST_BATCHNORMALIZATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_BATCHNORMALIZATION_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class BatchNormalizationLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, float>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator tensor_it,
                 std::vector<TensorShape>::const_iterator param_it,
                 std::vector<float>::const_iterator       epsilon_it)
            : _tensor_it{ std::move(tensor_it) },
              _param_it{ std::move(param_it) },
              _epsilon_it{ std::move(epsilon_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_tensor_it << ":";
            description << "Out=" << *_tensor_it << ":";
            description << "Mean=" << *_param_it << ":";
            description << "Variance=" << *_param_it << ":";
            description << "Beta=" << *_param_it << ":";
            description << "Gamma=" << *_param_it << ":";
            description << "Epsilon=" << *_epsilon_it;
            return description.str();
        }

        BatchNormalizationLayerDataset::type operator*() const
        {
            return std::make_tuple(*_tensor_it, *_param_it, *_epsilon_it);
        }

        iterator &operator++()
        {
            ++_tensor_it;
            ++_param_it;
            ++_epsilon_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _tensor_it;
        std::vector<TensorShape>::const_iterator _param_it;
        std::vector<float>::const_iterator       _epsilon_it;
    };

    iterator begin() const
    {
        return iterator(_tensor_shapes.begin(), _param_shapes.begin(), _epsilons.begin());
    }

    int size() const
    {
        return std::min(_tensor_shapes.size(), std::min(_param_shapes.size(), _epsilons.size()));
    }

    void add_config(TensorShape tensor, TensorShape param, float epsilon)
    {
        _tensor_shapes.emplace_back(std::move(tensor));
        _param_shapes.emplace_back(std::move(param));
        _epsilons.emplace_back(std::move(epsilon));
    }

protected:
    BatchNormalizationLayerDataset()                                  = default;
    BatchNormalizationLayerDataset(BatchNormalizationLayerDataset &&) = default;

private:
    std::vector<TensorShape> _tensor_shapes{};
    std::vector<TensorShape> _param_shapes{};
    std::vector<float>       _epsilons{};
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BATCHNORMALIZATION_LAYER_DATASET */
