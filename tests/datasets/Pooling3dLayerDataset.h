/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_POOLING_3D_LAYER_DATASET
#define ARM_COMPUTE_TEST_POOLING_3D_LAYER_DATASET

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class Pooling3dLayerDataset
{
public:
    using type = std::tuple<TensorShape, Pooling3dLayerInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator        src_it,
                 std::vector<Pooling3dLayerInfo>::const_iterator infos_it)
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

        Pooling3dLayerDataset::type operator*() const
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
        std::vector<TensorShape>::const_iterator        _src_it;
        std::vector<Pooling3dLayerInfo>::const_iterator _infos_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _infos.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), _infos.size());
    }

    void add_config(TensorShape src, Pooling3dLayerInfo info)
    {
        _src_shapes.emplace_back(std::move(src));
        _infos.emplace_back(std::move(info));
    }

protected:
    Pooling3dLayerDataset()                         = default;
    Pooling3dLayerDataset(Pooling3dLayerDataset &&) = default;

private:
    std::vector<TensorShape>        _src_shapes{};
    std::vector<Pooling3dLayerInfo> _infos{};
};

// Special pooling dataset
class Pooling3dLayerDatasetSpecial final : public Pooling3dLayerDataset
{
public:
    Pooling3dLayerDatasetSpecial()
    {
        // Special cases
        add_config(TensorShape(2U, 3U, 4U, 2U, 4U), Pooling3dLayerInfo(PoolingType::AVG, /*pool size*/ Size3D(2, 2, 1), /*pool strides*/ Size3D(3, 3, 1), /*pool padding*/ Padding3D(0, 0, 0), true));
        add_config(TensorShape(20U, 22U, 10U, 2U), Pooling3dLayerInfo(PoolingType::AVG, Size3D(100, 100, 100), Size3D(5, 5, 5), Padding3D(50, 50, 50), true));
        add_config(TensorShape(10U, 20U, 32U, 3U, 2U), Pooling3dLayerInfo(PoolingType::MAX, /*pool size*/ 3, /*pool strides*/ Size3D(2, 2, 2), Padding3D(1, 1, 1, 1, 1, 1), false, false,
                                                                    DimensionRoundingType::FLOOR));
        add_config(TensorShape(14U, 10U, 10U, 3U, 5U), Pooling3dLayerInfo(PoolingType::AVG,  Size3D(3, 3, 3), /*pool strides*/ Size3D(3, 3, 3), Padding3D(2, 1, 2), true, false, DimensionRoundingType::CEIL));
        add_config(TensorShape(14U, 10U, 10U, 2U, 4U), Pooling3dLayerInfo(PoolingType::AVG,  Size3D(3, 3, 3), /*pool strides*/ Size3D(3, 3, 3), Padding3D(2, 1, 2), false, false, DimensionRoundingType::CEIL));
        add_config(TensorShape(15U, 13U, 13U, 3U, 5U), Pooling3dLayerInfo(PoolingType::AVG,  Size3D(4, 4, 4), /*pool strides*/ Size3D(2, 2, 2), Padding3D(2, 2, 2), true, false, DimensionRoundingType::CEIL));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_POOLING_3D_LAYER_DATASET */
