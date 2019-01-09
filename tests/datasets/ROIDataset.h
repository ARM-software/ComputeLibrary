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
#ifndef ARM_COMPUTE_TEST_ROI_ALIGN_LAYER_DATASET
#define ARM_COMPUTE_TEST_ROI_ALIGN_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ROIDataset
{
public:
    using type = std::tuple<TensorShape, ROIPoolingLayerInfo, TensorShape>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator         tensor_shape_it,
                 std::vector<ROIPoolingLayerInfo>::const_iterator infos_it,
                 std::vector<TensorShape>::const_iterator         rois_shape_it)
            : _tensor_shape_it{ std::move(tensor_shape_it) },
              _infos_it{ std::move(infos_it) },
              _rois_shape_it{ std::move(rois_shape_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_tensor_shape_it << ":";
            description << "Info=" << *_infos_it << ":";
            description << "ROIS=" << *_rois_shape_it;
            return description.str();
        }

        ROIDataset::type operator*() const
        {
            return std::make_tuple(*_tensor_shape_it, *_infos_it, *_rois_shape_it);
        }

        iterator &operator++()
        {
            ++_tensor_shape_it;
            ++_infos_it;
            ++_rois_shape_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator         _tensor_shape_it;
        std::vector<ROIPoolingLayerInfo>::const_iterator _infos_it;
        std::vector<TensorShape>::const_iterator         _rois_shape_it;
    };

    iterator begin() const
    {
        return iterator(_tensor_shapes.begin(), _infos.begin(), _rois_shape.begin());
    }

    int size() const
    {
        return std::min(std::min(_tensor_shapes.size(), _infos.size()), _rois_shape.size());
    }

    void add_config(TensorShape tensor_shape, ROIPoolingLayerInfo info, TensorShape rois_shape)
    {
        _tensor_shapes.emplace_back(std::move(tensor_shape));
        _infos.emplace_back(std::move(info));
        _rois_shape.emplace_back(std::move(rois_shape));
    }

protected:
    ROIDataset()              = default;
    ROIDataset(ROIDataset &&) = default;

private:
    std::vector<TensorShape>         _tensor_shapes{};
    std::vector<ROIPoolingLayerInfo> _infos{};
    std::vector<TensorShape>         _rois_shape{};
};

class SmallROIDataset final : public ROIDataset
{
public:
    SmallROIDataset()
    {
        add_config(TensorShape(50U, 47U, 1U, 1U), ROIPoolingLayerInfo(7U, 7U, 1.f / 4.f), TensorShape(5U, 1U));
        add_config(TensorShape(50U, 47U, 3U, 4U), ROIPoolingLayerInfo(7U, 7U, 1.f / 4.f), TensorShape(5U, 1U));
        add_config(TensorShape(50U, 47U, 3U, 1U), ROIPoolingLayerInfo(7U, 7U, 1.f / 4.f), TensorShape(5U, 10U));
        add_config(TensorShape(50U, 47U, 10U, 1U), ROIPoolingLayerInfo(7U, 7U, 1.f / 4.f), TensorShape(5U, 80U));

        //Spatial Scale 1/4
        add_config(TensorShape(50U, 47U, 80U, 4U), ROIPoolingLayerInfo(7U, 7U, 1.f / 4.f), TensorShape(5U, 80U));
        add_config(TensorShape(50U, 47U, 3U, 1U), ROIPoolingLayerInfo(9U, 9U, 1.f / 4.f), TensorShape(5U, 40U));
        add_config(TensorShape(50U, 47U, 10U, 1U), ROIPoolingLayerInfo(9U, 9U, 1.f / 4.f), TensorShape(5U, 80U));
        add_config(TensorShape(50U, 47U, 80U, 8U), ROIPoolingLayerInfo(9U, 9U, 1.f / 4.f), TensorShape(5U, 80U));

        //Spatial Scale 1/8
        add_config(TensorShape(50U, 47U, 80U, 4U), ROIPoolingLayerInfo(7U, 7U, 1.f / 8.f), TensorShape(5U, 80U));
        add_config(TensorShape(50U, 47U, 3U, 1U), ROIPoolingLayerInfo(9U, 9U, 1.f / 8.f), TensorShape(5U, 40U));
        add_config(TensorShape(50U, 47U, 10U, 1U), ROIPoolingLayerInfo(9U, 9U, 1.f / 8.f), TensorShape(5U, 80U));
        add_config(TensorShape(50U, 47U, 80U, 8U), ROIPoolingLayerInfo(9U, 9U, 1.f / 8.f), TensorShape(5U, 80U));

        //Spatial Scale 1/16
        add_config(TensorShape(50U, 47U, 80U, 4U), ROIPoolingLayerInfo(7U, 7U, 1.f / 16.f), TensorShape(5U, 80U));
        add_config(TensorShape(50U, 47U, 3U, 1U), ROIPoolingLayerInfo(9U, 9U, 1.f / 16.f), TensorShape(5U, 40U));
        add_config(TensorShape(50U, 47U, 10U, 1U), ROIPoolingLayerInfo(9U, 9U, 1.f / 16.f), TensorShape(5U, 80U));
        add_config(TensorShape(50U, 47U, 80U, 8U), ROIPoolingLayerInfo(9U, 9U, 1.f / 16.f), TensorShape(5U, 80U));
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ROI_ALIGN_LAYER_DATASET */
