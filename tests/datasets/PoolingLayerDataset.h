/*
 * Copyright (c) 2017-2021 Arm Limited.
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
    using type = std::tuple<TensorShape, PoolingLayerInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator      src_it,
                 std::vector<PoolingLayerInfo>::const_iterator infos_it)
            : _src_it{ std::move(src_it) },
              _infos_it{ std::move(infos_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "Info=" << *_infos_it << ":";
            return description.str();
        }

        PoolingLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_infos_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_infos_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator      _src_it;
        std::vector<PoolingLayerInfo>::const_iterator _infos_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _infos.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), _infos.size());
    }

    void add_config(TensorShape src, PoolingLayerInfo info)
    {
        _src_shapes.emplace_back(std::move(src));
        _infos.emplace_back(std::move(info));
    }

protected:
    PoolingLayerDataset()                       = default;
    PoolingLayerDataset(PoolingLayerDataset &&) = default;

private:
    std::vector<TensorShape>      _src_shapes{};
    std::vector<PoolingLayerInfo> _infos{};
};

// Special pooling dataset
class PoolingLayerDatasetSpecial final : public PoolingLayerDataset
{
public:
    PoolingLayerDatasetSpecial()
    {
        // Special cases
        add_config(TensorShape(2U, 3U, 4U, 1U), PoolingLayerInfo(PoolingType::AVG, Size2D(2, 2), DataLayout::NCHW, PadStrideInfo(3, 3, 0, 0), true));
        add_config(TensorShape(60U, 52U, 3U, 2U), PoolingLayerInfo(PoolingType::AVG, Size2D(100, 100), DataLayout::NCHW, PadStrideInfo(5, 5, 50, 50), true));
        // Asymmetric padding
        add_config(TensorShape(112U, 112U, 32U), PoolingLayerInfo(PoolingType::MAX, 3, DataLayout::NCHW, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR)));
        add_config(TensorShape(14U, 14U, 832U), PoolingLayerInfo(PoolingType::MAX, 2, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL)));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_POOLING_LAYER_DATASET */
