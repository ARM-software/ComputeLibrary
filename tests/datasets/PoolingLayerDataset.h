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
#ifndef ARM_COMPUTE_TEST_POOLING_LAYER_DATASET
#define ARM_COMPUTE_TEST_POOLING_LAYER_DATASET

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class PoolingLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, PoolingLayerInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator      src_it,
                 std::vector<TensorShape>::const_iterator      dst_it,
                 std::vector<PoolingLayerInfo>::const_iterator infos_it)
            : _src_it{ std::move(src_it) },
              _dst_it{ std::move(dst_it) },
              _infos_it{ std::move(infos_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "Out=" << *_dst_it << ":";
            description << "Info=" << *_infos_it;
            return description.str();
        }

        PoolingLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_dst_it, *_infos_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_dst_it;
            ++_infos_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator      _src_it;
        std::vector<TensorShape>::const_iterator      _dst_it;
        std::vector<PoolingLayerInfo>::const_iterator _infos_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _dst_shapes.begin(), _infos.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_dst_shapes.size(), _infos.size()));
    }

    void add_config(TensorShape src, TensorShape dst, PoolingLayerInfo info)
    {
        _src_shapes.emplace_back(std::move(src));
        _dst_shapes.emplace_back(std::move(dst));
        _infos.emplace_back(std::move(info));
    }

protected:
    PoolingLayerDataset()                       = default;
    PoolingLayerDataset(PoolingLayerDataset &&) = default;

private:
    std::vector<TensorShape>      _src_shapes{};
    std::vector<TensorShape>      _dst_shapes{};
    std::vector<PoolingLayerInfo> _infos{};
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_POOLING_LAYER_DATASET */
