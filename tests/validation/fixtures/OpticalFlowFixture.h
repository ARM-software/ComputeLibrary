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
#ifndef ARM_COMPUTE_TEST_OPTICAL_FLOW
#define ARM_COMPUTE_TEST_OPTICAL_FLOW

#include "arm_compute/core/PyramidInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/Types.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/OpticalFlow.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType,
          typename AccessorType,
          typename ArrayType,
          typename ArrayAccessorType,
          typename FunctionType,
          typename PyramidType,
          typename PyramidFunctionType,
          typename T>

class OpticalFlowValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string old_image_name, std::string new_image_name, OpticalFlowParameters params,
               size_t num_levels, size_t num_keypoints, Format format, BorderMode border_mode)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> int_dist(0, 255);
        const uint8_t                          constant_border_value = int_dist(gen);

        // Create keypoints
        std::vector<KeyPoint> old_keypoints           = generate_random_keypoints(library->get_image_shape(old_image_name), num_keypoints, library->seed(), num_levels);
        std::vector<KeyPoint> new_keypoints_estimates = old_keypoints;

        _target    = compute_target(old_image_name, new_image_name, params, num_levels, old_keypoints, new_keypoints_estimates, format, border_mode, constant_border_value);
        _reference = compute_reference(old_image_name, new_image_name, params, num_levels, old_keypoints, new_keypoints_estimates, format, border_mode, constant_border_value);
    }

protected:
    template <typename V>
    void fill(V &&tensor, const std::string image, Format format)
    {
        library->fill(tensor, image, format);
    }

    ArrayType compute_target(std::string old_image_name, std::string new_image_name, OpticalFlowParameters params, size_t num_levels,
                             std::vector<KeyPoint> &old_keypoints, std::vector<KeyPoint> &new_keypoints_estimates,
                             Format format, BorderMode border_mode, uint8_t constant_border_value)
    {
        // Get image shapes
        TensorShape old_shape = library->get_image_shape(old_image_name);
        TensorShape new_shape = library->get_image_shape(new_image_name);

        // Create tensors
        auto old_image = create_tensor<TensorType>(old_shape, format);
        auto new_image = create_tensor<TensorType>(new_shape, format);

        // Load keypoints
        ArrayType old_points(old_keypoints.size());
        ArrayType new_points_estimates(new_keypoints_estimates.size());
        ArrayType new_points(old_keypoints.size());

        fill_array(ArrayAccessorType(old_points), old_keypoints);
        fill_array(ArrayAccessorType(new_points_estimates), new_keypoints_estimates);

        // Create pyramid images
        PyramidInfo pyramid_info(num_levels, SCALE_PYRAMID_HALF, old_image.info()->tensor_shape(), format);
        PyramidType old_pyramid = create_pyramid<PyramidType>(pyramid_info);
        PyramidType new_pyramid = create_pyramid<PyramidType>(pyramid_info);

        // Create and configure pyramid functions
        PyramidFunctionType old_gp;
        old_gp.configure(&old_image, &old_pyramid, border_mode, constant_border_value);

        PyramidFunctionType new_gp;
        new_gp.configure(&new_image, &new_pyramid, border_mode, constant_border_value);

        for(size_t i = 0; i < pyramid_info.num_levels(); ++i)
        {
            ARM_COMPUTE_EXPECT(old_pyramid.get_pyramid_level(i)->info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(new_pyramid.get_pyramid_level(i)->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Create and configure optical flow function
        FunctionType optical_flow;

        optical_flow.configure(&old_pyramid,
                               &new_pyramid,
                               &old_points,
                               &new_points_estimates,
                               &new_points,
                               params.termination,
                               params.epsilon,
                               params.num_iterations,
                               params.window_dimension,
                               params.use_initial_estimate,
                               border_mode,
                               constant_border_value);

        ARM_COMPUTE_EXPECT(old_image.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(new_image.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate input tensors
        old_image.allocator()->allocate();
        new_image.allocator()->allocate();

        // Allocate pyramids
        old_pyramid.allocate();
        new_pyramid.allocate();

        ARM_COMPUTE_EXPECT(!old_image.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!new_image.info()->is_resizable(), framework::LogLevel::ERRORS);

        for(size_t i = 0; i < pyramid_info.num_levels(); ++i)
        {
            ARM_COMPUTE_EXPECT(!old_pyramid.get_pyramid_level(i)->info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(!new_pyramid.get_pyramid_level(i)->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Fill tensors
        fill(AccessorType(old_image), old_image_name, format);
        fill(AccessorType(new_image), new_image_name, format);

        // Compute functions
        old_gp.run();
        new_gp.run();
        optical_flow.run();

        return new_points;
    }

    std::vector<KeyPoint> compute_reference(std::string old_image_name, std::string new_image_name,
                                            OpticalFlowParameters params, size_t num_levels,
                                            std::vector<KeyPoint> &old_keypoints, std::vector<KeyPoint> &new_keypoints_estimates,
                                            Format format, BorderMode border_mode, uint8_t constant_border_value)
    {
        SimpleTensor<T> old_image{ library->get_image_shape(old_image_name), data_type_from_format(format) };
        SimpleTensor<T> new_image{ library->get_image_shape(new_image_name), data_type_from_format(format) };

        fill(old_image, old_image_name, format);
        fill(new_image, new_image_name, format);

        return reference::optical_flow<T>(old_image, new_image, params, num_levels, old_keypoints, new_keypoints_estimates,
                                          border_mode, constant_border_value);
    }

    ArrayType             _target{};
    std::vector<KeyPoint> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_OPTICAL_FLOW */
