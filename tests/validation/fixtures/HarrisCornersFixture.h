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
#ifndef ARM_COMPUTE_TEST_HARRIS_CORNERS_FIXTURE
#define ARM_COMPUTE_TEST_HARRIS_CORNERS_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/HarrisCornerDetector.h"

namespace arm_compute
{
class CLHarrisCorners;
class NEHarrisCorners;

namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename ArrayType, typename FunctionType, typename T>
class HarrisCornersValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, int gradient_size, int block_size, BorderMode border_mode, bool use_fp16, Format format)
    {
        HarrisCornersParameters params = harris_corners_parameters();

        _target = compute_target(shape, gradient_size, block_size, border_mode, use_fp16, format, params);
        _reference = compute_reference(shape, gradient_size, block_size, border_mode, format, params);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    template <typename F, typename std::enable_if<std::is_same<F, NEHarrisCorners>::value, int>::type = 0>
    void configure_target(F &func, TensorType &src, ArrayType &corners, int gradient_size, int block_size, BorderMode border_mode, bool use_fp16, const HarrisCornersParameters &params)
    {
        func.configure(&src, params.threshold, params.min_dist, params.sensitivity, gradient_size, block_size, &corners, border_mode, params.constant_border_value, use_fp16);
    }

    template <typename F, typename std::enable_if<std::is_same<F, CLHarrisCorners>::value, int>::type = 0>
    void configure_target(F &func, TensorType &src, ArrayType &corners, int gradient_size, int block_size, BorderMode border_mode, bool use_fp16, const HarrisCornersParameters &params)
    {
        ARM_COMPUTE_UNUSED(use_fp16);
        ARM_COMPUTE_ERROR_ON(use_fp16);
        func.configure(&src, params.threshold, params.min_dist, params.sensitivity, gradient_size, block_size, &corners, border_mode, params.constant_border_value);
    }

    ArrayType compute_target(const TensorShape &shape, int gradient_size, int block_size, BorderMode border_mode, bool use_fp16, Format format, const HarrisCornersParameters &params)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type_from_format(format));
        src.info()->set_format(format);

        // Create array of keypoints
        ArrayType corners(shape.total_size());

        // Create harris corners configure function
        FunctionType harris_corners;
        configure_target<FunctionType>(harris_corners, src, corners, gradient_size, block_size, border_mode, use_fp16, params);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        harris_corners.run();

        return corners;
    }

    std::vector<KeyPoint> compute_reference(const TensorShape &shape, int gradient_size, int block_size, BorderMode border_mode, Format format, const HarrisCornersParameters &params)
    {
        // Create reference
        SimpleTensor<T> src{ shape, format };

        // Fill reference
        fill(src);

        return reference::harris_corner_detector<T>(src, params.threshold, params.min_dist, params.sensitivity, gradient_size, block_size, border_mode, params.constant_border_value);
    }

    ArrayType             _target{};
    std::vector<KeyPoint> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HARRIS_CORNERS_FIXTURE */
