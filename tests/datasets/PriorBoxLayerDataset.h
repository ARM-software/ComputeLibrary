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
#ifndef ARM_COMPUTE_TEST_PRIORBOX_LAYER_DATASET
#define ARM_COMPUTE_TEST_PRIORBOX_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class PriorBoxLayerDataset
{
public:
    using type = std::tuple<TensorShape, PriorBoxLayerInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator       src_it,
                 std::vector<PriorBoxLayerInfo>::const_iterator infos_it)
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

        PriorBoxLayerDataset::type operator*() const
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
        std::vector<TensorShape>::const_iterator       _src_it;
        std::vector<PriorBoxLayerInfo>::const_iterator _infos_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _infos.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), _infos.size());
    }

    void add_config(TensorShape src, PriorBoxLayerInfo info)
    {
        _src_shapes.emplace_back(std::move(src));
        _infos.emplace_back(std::move(info));
    }

protected:
    PriorBoxLayerDataset()                        = default;
    PriorBoxLayerDataset(PriorBoxLayerDataset &&) = default;

private:
    std::vector<TensorShape>       _src_shapes{};
    std::vector<PriorBoxLayerInfo> _infos{};
};

class SmallPriorBoxLayerDataset final : public PriorBoxLayerDataset
{
public:
    SmallPriorBoxLayerDataset()
    {
        std::vector<float> min_val      = { 30.f };
        std::vector<float> var          = { 0.1, 0.1, 0.2, 0.2 };
        std::vector<float> max_val      = { 60.f };
        std::vector<float> aspect_ratio = { 2.f };
        std::array<float, 2> steps = { { 8.f, 8.f } };
        add_config(TensorShape(4U, 4U), PriorBoxLayerInfo(min_val, var, 0.5f, true, false, max_val, aspect_ratio, Coordinates2D{ 8, 8 }, steps));
    }
};

class LargePriorBoxLayerDataset final : public PriorBoxLayerDataset
{
public:
    LargePriorBoxLayerDataset()
    {
        std::vector<float> min_val      = { 30.f };
        std::vector<float> var          = { 0.1, 0.1, 0.2, 0.2 };
        std::vector<float> max_val      = { 60.f };
        std::vector<float> aspect_ratio = { 2.f };
        std::array<float, 2> steps = { { 8.f, 8.f } };
        add_config(TensorShape(150U, 245U, 4U, 12U), PriorBoxLayerInfo(min_val, var, 0.5f, true, false, max_val, aspect_ratio, Coordinates2D{ 8, 8 }, steps));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_PRIORBOX_LAYER_DATASET */
