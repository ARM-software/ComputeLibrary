/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_CROP_RESIZE_DATASET
#define ARM_COMPUTE_TEST_CROP_RESIZE_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class CropResizeDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, Coordinates2D, InterpolationPolicy, float>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator         src_shapes_it,
                 std::vector<TensorShape>::const_iterator         boxes_shapes_it,
                 std::vector<Coordinates2D>::const_iterator       crop_size_values_it,
                 std::vector<InterpolationPolicy>::const_iterator method_values_it,
                 std::vector<float>::const_iterator               extrapolation_values_it)
            : _src_shapes_it{ std::move(src_shapes_it) },
              _boxes_shapes_it{ std::move(boxes_shapes_it) },
              _crop_size_values_it{ std::move(crop_size_values_it) },
              _method_values_it{ std::move(method_values_it) },
              _extrapolation_values_it{ std::move(extrapolation_values_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Src_Shape=" << *_src_shapes_it << ":";
            description << "Boxes_Shape=" << *_boxes_shapes_it << ":";
            description << "Crop_Size=(" << (*_crop_size_values_it).x << "," << (*_crop_size_values_it).y << "):";
            description << "Method=" << *_method_values_it << ":";
            description << "Extrapolation_value=" << *_extrapolation_values_it << ":";
            return description.str();
        }

        CropResizeDataset::type operator*() const
        {
            return std::make_tuple(*_src_shapes_it, *_boxes_shapes_it, *_crop_size_values_it, *_method_values_it, *_extrapolation_values_it);
        }

        iterator &operator++()
        {
            ++_src_shapes_it;
            ++_boxes_shapes_it;
            ++_crop_size_values_it;
            ++_method_values_it;
            ++_extrapolation_values_it;
            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator         _src_shapes_it;
        std::vector<TensorShape>::const_iterator         _boxes_shapes_it;
        std::vector<Coordinates2D>::const_iterator       _crop_size_values_it;
        std::vector<InterpolationPolicy>::const_iterator _method_values_it;
        std::vector<float>::const_iterator               _extrapolation_values_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _boxes_shapes.begin(), _crop_size_values.begin(), _method_values.begin(), _extrapolation_values.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_boxes_shapes.size(), std::min(_crop_size_values.size(), std::min(_method_values.size(), _extrapolation_values.size()))));
    }

    void add_config(TensorShape src_shape, TensorShape boxes_shape, Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value)
    {
        _src_shapes.emplace_back(std::move(src_shape));
        _boxes_shapes.emplace_back(std::move(boxes_shape));
        _crop_size_values.emplace_back(std::move(crop_size));
        _method_values.emplace_back(std::move(method));
        _extrapolation_values.emplace_back(std::move(extrapolation_value));
    }

protected:
    CropResizeDataset()                     = default;
    CropResizeDataset(CropResizeDataset &&) = default;

private:
    std::vector<TensorShape>         _src_shapes{};
    std::vector<TensorShape>         _boxes_shapes{};
    std::vector<Coordinates2D>       _crop_size_values{};
    std::vector<InterpolationPolicy> _method_values{};
    std::vector<float>               _extrapolation_values{};
};

class SmallCropResizeDataset final : public CropResizeDataset
{
public:
    SmallCropResizeDataset()
    {
        add_config(TensorShape(1U, 5U, 5U), TensorShape(4, 5), Coordinates2D{ 2, 2 }, InterpolationPolicy::BILINEAR, 100);
        add_config(TensorShape(3U, 5U, 5U), TensorShape(4, 5), Coordinates2D{ 2, 2 }, InterpolationPolicy::BILINEAR, 100);
        add_config(TensorShape(1U, 5U, 5U), TensorShape(4, 5), Coordinates2D{ 10, 10 }, InterpolationPolicy::BILINEAR, 100);
        add_config(TensorShape(15U, 30U, 30U, 10U), TensorShape(4, 20), Coordinates2D{ 10, 10 }, InterpolationPolicy::BILINEAR, 100);

        add_config(TensorShape(1U, 5U, 5U), TensorShape(4, 5), Coordinates2D{ 2, 2 }, InterpolationPolicy::NEAREST_NEIGHBOR, 100);
        add_config(TensorShape(3U, 5U, 5U), TensorShape(4, 5), Coordinates2D{ 2, 2 }, InterpolationPolicy::NEAREST_NEIGHBOR, 100);
        add_config(TensorShape(1U, 5U, 5U), TensorShape(4, 5), Coordinates2D{ 10, 10 }, InterpolationPolicy::NEAREST_NEIGHBOR, 100);
        add_config(TensorShape(15U, 30U, 30U, 10U), TensorShape(4, 20), Coordinates2D{ 10, 10 }, InterpolationPolicy::NEAREST_NEIGHBOR, 100);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CROP_RESIZE_DATASET */