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
#ifndef ARM_COMPUTE_TEST_SCALE_LAYER_DATASET
#define ARM_COMPUTE_TEST_SCALE_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ScaleLayerDataset
{
public:
    using type = std::tuple<TensorShape, InterpolationPolicy, BorderMode, SamplingPolicy, float, float>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator         src_it,
                 std::vector<InterpolationPolicy>::const_iterator policy_it,
                 std::vector<BorderMode>::const_iterator          border_mode_it,
                 std::vector<SamplingPolicy>::const_iterator      sampling_policy_it,
                 std::vector<float>::const_iterator               scale_x_it,
                 std::vector<float>::const_iterator               scale_y_it)
            : _src_it{ std::move(src_it) },
              _policy_it{ std::move(policy_it) },
              _border_mode_it{ std::move(border_mode_it) },
              _sampling_policy_it{ std::move(sampling_policy_it) },
              _scale_x_it{ std::move(scale_x_it) },
              _scale_y_it{ std::move(scale_y_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "InterpolationPolicy=" << *_policy_it << ":";
            description << "BorderMode=" << *_border_mode_it << ":";
            description << "SamplingPolicy=" << *_sampling_policy_it << ":";
            description << "Scale_x=" << *_scale_x_it << ":";
            description << "Scale_y=" << *_scale_y_it;
            return description.str();
        }

        ScaleLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_policy_it, *_border_mode_it, *_sampling_policy_it, *_scale_x_it, *_scale_y_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_policy_it;
            ++_border_mode_it;
            ++_sampling_policy_it;
            ++_scale_x_it;
            ++_scale_y_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator         _src_it;
        std::vector<InterpolationPolicy>::const_iterator _policy_it;
        std::vector<BorderMode>::const_iterator          _border_mode_it;
        std::vector<SamplingPolicy>::const_iterator      _sampling_policy_it;
        std::vector<float>::const_iterator               _scale_x_it;
        std::vector<float>::const_iterator               _scale_y_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _policy.begin(), _border_mode.begin(), _sampling_policy.begin(), _scale_x.begin(), _scale_y.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_policy.size(), std::min(_border_mode.size(), std::min(_sampling_policy.size(), std::min(_scale_x.size(), _scale_y.size())))));
    }

    void add_config(TensorShape src, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy, float scale_x, float scale_y)
    {
        _src_shapes.emplace_back(std::move(src));
        _policy.emplace_back(std::move(policy));
        _border_mode.emplace_back(std::move(border_mode));
        _sampling_policy.emplace_back(std::move(sampling_policy));
        _scale_x.emplace_back(std::move(scale_x));
        _scale_y.emplace_back(std::move(scale_y));
    }

protected:
    ScaleLayerDataset()                     = default;
    ScaleLayerDataset(ScaleLayerDataset &&) = default;

private:
    std::vector<TensorShape>         _src_shapes{};
    std::vector<InterpolationPolicy> _policy{};
    std::vector<BorderMode>          _border_mode{};
    std::vector<SamplingPolicy>      _sampling_policy{};
    std::vector<float>               _scale_x{};
    std::vector<float>               _scale_y{};
};

/** Data set containing small scale layer shapes. */
class SmallScaleLayerShapes final : public ScaleLayerDataset
{
public:
    SmallScaleLayerShapes()
    {
        add_config(TensorShape(128U, 64U, 1U, 3U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 5, 5);
        add_config(TensorShape(9U, 9U, 3U, 4U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 7, 7);
        add_config(TensorShape(27U, 13U, 2U, 4U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 9, 9);
    }
};

/** Data set containing large scale layer shapes. */
class LargeScaleLayerShapes final : public ScaleLayerDataset
{
public:
    LargeScaleLayerShapes()
    {
        add_config(TensorShape(1920U, 1080U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 0.5, 0.5);
        add_config(TensorShape(640U, 480U, 2U, 3U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 0.5, 0.5);
        add_config(TensorShape(4160U, 3120U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 0.5, 0.5);
        add_config(TensorShape(800U, 600U, 1U, 4U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 0.5, 0.5);

        add_config(TensorShape(1920U, 1080U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 2, 2);
        add_config(TensorShape(640U, 480U, 2U, 3U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 2, 2);
        add_config(TensorShape(4160U, 3120U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 2, 2);
        add_config(TensorShape(800U, 600U, 1U, 4U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 2, 2);

        add_config(TensorShape(1920U, 1080U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 3, 3);
        add_config(TensorShape(640U, 480U, 2U, 3U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 3, 3);
        add_config(TensorShape(4160U, 3120U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 3, 3);
        add_config(TensorShape(800U, 600U, 1U, 4U), InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED, SamplingPolicy::CENTER, 3, 3);
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCALE_LAYER_DATASET */
