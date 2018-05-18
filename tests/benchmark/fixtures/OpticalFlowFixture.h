/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_OPTICAL_FLOW_FIXTURE
#define ARM_COMPUTE_TEST_OPTICAL_FLOW_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Types.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename TensorType,
          typename Function,
          typename Accessor,
          typename ArrayType,
          typename ArrayAccessor,
          typename PyramidType,
          typename PyramidFunctionType>
class OpticalFlowFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string old_image_name, std::string new_image_name, OpticalFlowParameters params,
               size_t num_levels, size_t num_keypoints, Format format, BorderMode border_mode)
    {
        const uint8_t                         constant_border_value = 0;
        const std::random_device::result_type seed                  = 0;

        // Create keypoints
        old_keypoints           = generate_random_keypoints(library->get_image_shape(old_image_name), num_keypoints, seed, num_levels);
        new_keypoints_estimates = old_keypoints;

        // Create tensors
        old_image = create_tensor<TensorType>(library->get_image_shape(old_image_name), format);
        new_image = create_tensor<TensorType>(library->get_image_shape(new_image_name), format);

        // Load keypoints
        fill_array(ArrayAccessor(old_points), old_keypoints);
        fill_array(ArrayAccessor(new_points_estimates), new_keypoints_estimates);

        // Create pyramid images
        PyramidInfo pyramid_info(num_levels, SCALE_PYRAMID_HALF, old_image.info()->tensor_shape(), format);
        old_pyramid = create_pyramid<PyramidType>(pyramid_info);
        new_pyramid = create_pyramid<PyramidType>(pyramid_info);

        // Create and configure pyramid functions
        old_gaussian_pyramid_func.configure(&old_image, &old_pyramid, border_mode, constant_border_value);
        new_gaussian_pyramid_func.configure(&new_image, &new_pyramid, border_mode, constant_border_value);

        optical_flow_func.configure(&old_pyramid,
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

        // Allocate input tensors
        old_image.allocator()->allocate();
        new_image.allocator()->allocate();

        // Allocate pyramids
        old_pyramid.allocate();
        new_pyramid.allocate();

        // Copy image data to tensors
        library->fill(Accessor(old_image), old_image_name, format);
        library->fill(Accessor(new_image), new_image_name, format);

        // Compute gaussian pyramids
        old_gaussian_pyramid_func.run();
        new_gaussian_pyramid_func.run();
    }

    void run()
    {
        optical_flow_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
    }

    void teardown()
    {
        old_image.allocator()->free();
        new_image.allocator()->free();
    }

private:
    static const size_t max_keypoints = 10000;

    std::vector<KeyPoint> old_keypoints{};
    std::vector<KeyPoint> new_keypoints_estimates{};

    TensorType old_image{};
    TensorType new_image{};

    ArrayType old_points{ max_keypoints };
    ArrayType new_points{ max_keypoints };
    ArrayType new_points_estimates{ max_keypoints };

    PyramidType old_pyramid{};
    PyramidType new_pyramid{};

    PyramidFunctionType old_gaussian_pyramid_func{};
    PyramidFunctionType new_gaussian_pyramid_func{};

    Function optical_flow_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_OPTICAL_FLOW_FIXTURE */
