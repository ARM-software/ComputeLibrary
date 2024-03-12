/*
 * Copyright (c) 2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_OPTICAL_FLOW_DATASET
#define ARM_COMPUTE_TEST_OPTICAL_FLOW_DATASET

#include "tests/TypePrinter.h"
#include "tests/validation/Helpers.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class OpticalFlowDataset
{
public:
    using type = std::tuple<std::string, std::string, OpticalFlowParameters, size_t, size_t>;

    struct iterator
    {
        iterator(std::vector<std::string>::const_iterator           old_image_it,
                 std::vector<std::string>::const_iterator           new_image_it,
                 std::vector<OpticalFlowParameters>::const_iterator params_it,
                 std::vector<size_t>::const_iterator                num_levels_it,
                 std::vector<size_t>::const_iterator                num_keypoints_it)
            : _old_image_it{ std::move(old_image_it) },
              _new_image_it{ std::move(new_image_it) },
              _params_it{ std::move(params_it) },
              _num_levels_it{ std::move(num_levels_it) },
              _num_keypoints_it{ std::move(num_keypoints_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "NumLevels=" << *_num_levels_it << ":";
            description << "NumKeypoints=" << *_num_keypoints_it << ":";
            description << "Termination=" << _params_it->termination << ":";
            description << "Epsilon=" << _params_it->epsilon << ":";
            description << "NumIterations=" << _params_it->num_iterations << ":";
            description << "WindowDimension=" << _params_it->window_dimension << ":";
            description << "InitialEstimate=" << std::boolalpha << _params_it->use_initial_estimate;

            return description.str();
        }

        OpticalFlowDataset::type operator*() const
        {
            return std::make_tuple(*_old_image_it, *_new_image_it, *_params_it, *_num_levels_it, *_num_keypoints_it);
        }

        iterator &operator++()
        {
            ++_old_image_it;
            ++_new_image_it;
            ++_params_it;
            ++_num_levels_it;
            ++_num_keypoints_it;

            return *this;
        }

    private:
        std::vector<std::string>::const_iterator           _old_image_it;
        std::vector<std::string>::const_iterator           _new_image_it;
        std::vector<OpticalFlowParameters>::const_iterator _params_it;
        std::vector<size_t>::const_iterator                _num_levels_it;
        std::vector<size_t>::const_iterator                _num_keypoints_it;
    };

    iterator begin() const
    {
        return iterator(_old_image.begin(),
                        _new_image.begin(),
                        _params.begin(),
                        _num_levels.begin(),
                        _num_keypoints.begin());
    }

    int size() const
    {
        return std::min(_old_image.size(), std::min(_new_image.size(), std::min(_params.size(), std::min(_num_levels.size(), _num_keypoints.size()))));
    }

    void add_config(std::string old_image, std::string new_image, OpticalFlowParameters params, size_t num_levels, size_t num_keypoints)
    {
        _old_image.emplace_back(std::move(old_image));
        _new_image.emplace_back(std::move(new_image));
        _params.emplace_back(params);
        _num_levels.emplace_back(num_levels);
        _num_keypoints.emplace_back(num_keypoints);
    }

protected:
    OpticalFlowDataset()                      = default;
    OpticalFlowDataset(OpticalFlowDataset &&) = default;

private:
    std::vector<std::string>           _old_image{};
    std::vector<std::string>           _new_image{};
    std::vector<OpticalFlowParameters> _params{};
    std::vector<size_t>                _num_levels{};
    std::vector<size_t>                _num_keypoints{};
};

// *INDENT-OFF*
// clang-format off
class SmallOpticalFlowDataset final : public OpticalFlowDataset
{
public:
    SmallOpticalFlowDataset()
    {
        //         old_image              new_image              (termination, epsilon, num_iterations, window_dimension, initial_estimate) levels keypoints
        add_config("opticalflow_old.pgm", "opticalflow_new.pgm", OpticalFlowParameters(Termination::TERM_CRITERIA_BOTH,       0.01f, 3, 5, true), 3, 1000);
        add_config("opticalflow_old.pgm", "opticalflow_new.pgm", OpticalFlowParameters(Termination::TERM_CRITERIA_EPSILON,    0.01f, 3, 5, true), 3, 1000);
        add_config("opticalflow_old.pgm", "opticalflow_new.pgm", OpticalFlowParameters(Termination::TERM_CRITERIA_ITERATIONS, 0.01f, 3, 5, true), 3, 1000);
    }
};

class LargeOpticalFlowDataset final : public OpticalFlowDataset
{
public:
    LargeOpticalFlowDataset()
    {
        //         old_image              new_image              (termination, epsilon, num_iterations, window_dimension, initial_estimate) levels keypoints
        add_config("opticalflow_old.pgm", "opticalflow_new.pgm", OpticalFlowParameters(Termination::TERM_CRITERIA_BOTH,       0.01f, 3, 5, true), 3, 10000);
        add_config("opticalflow_old.pgm", "opticalflow_new.pgm", OpticalFlowParameters(Termination::TERM_CRITERIA_EPSILON,    0.01f, 3, 5, true), 3, 10000);
        add_config("opticalflow_old.pgm", "opticalflow_new.pgm", OpticalFlowParameters(Termination::TERM_CRITERIA_ITERATIONS, 0.01f, 3, 5, true), 3, 10000);

        //         old_image              new_image              (termination, epsilon, num_iterations, window_dimension, initial_estimate) levels keypoints
        add_config("opticalflow_old.pgm", "opticalflow_new.pgm", OpticalFlowParameters(Termination::TERM_CRITERIA_BOTH,       0.01f, 3, 5, false), 3, 10000);
        add_config("opticalflow_old.pgm", "opticalflow_new.pgm", OpticalFlowParameters(Termination::TERM_CRITERIA_EPSILON,    0.01f, 3, 5, false), 3, 10000);
        add_config("opticalflow_old.pgm", "opticalflow_new.pgm", OpticalFlowParameters(Termination::TERM_CRITERIA_ITERATIONS, 0.01f, 3, 5, false), 3, 10000);
    }
};
// clang-format on
// *INDENT-ON*

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_OPTICAL_FLOW_DATASET */
