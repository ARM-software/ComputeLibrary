/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_FAST_CORNERS_FIXTURE
#define ARM_COMPUTE_TEST_FAST_CORNERS_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/FastCorners.h"

#include <random>

namespace arm_compute
{
class CLFastCorners;
class NEFastCorners;

namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename ArrayType, typename FunctionType, typename T>
class FastCornersValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string image, Format format, bool suppress_nonmax, BorderMode border_mode)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> int_dist(0, 255);
        std::uniform_real_distribution<float>  real_dist(0, 255);

        const uint8_t constant_border_value = int_dist(gen);
        const float   threshold             = real_dist(gen);

        _target    = compute_target(image, format, threshold, suppress_nonmax, border_mode, constant_border_value);
        _reference = compute_reference(image, format, threshold, suppress_nonmax, border_mode, constant_border_value);
    }

protected:
    template <typename U>
    void fill(U &&tensor, RawTensor raw)
    {
        library->fill(tensor, raw);
    }

    template <typename F, typename std::enable_if<std::is_same<F, CLFastCorners>::value, int>::type = 0>
    void configure_target(F &func, TensorType &src, ArrayType &corners, unsigned int *num_corners, float threshold, bool suppress_nonmax, BorderMode border_mode, uint8_t constant_border_value)
    {
        func.configure(&src, threshold, suppress_nonmax, &corners, num_corners, border_mode, constant_border_value);
    }

    template <typename F, typename std::enable_if<std::is_same<F, NEFastCorners>::value, int>::type = 0>
    void configure_target(F &func, TensorType &src, ArrayType &corners, unsigned int *num_corners, float threshold, bool suppress_nonmax, BorderMode border_mode, uint8_t constant_border_value)
    {
        ARM_COMPUTE_UNUSED(num_corners);
        func.configure(&src, threshold, suppress_nonmax, &corners, border_mode, constant_border_value);
    }

    ArrayType compute_target(const std::string &image, Format format, float threshold, bool suppress_nonmax, BorderMode border_mode, uint8_t constant_border_value)
    {
        // Load the image (cached by the library if loaded before)
        const RawTensor &raw = library->get(image, format);

        // Create tensors
        TensorType src = create_tensor<TensorType>(raw.shape(), format);

        // Create array of keypoints
        ArrayType    corners(raw.shape().total_size());
        unsigned int num_corners = raw.shape().total_size();

        // Create and configure function
        FunctionType fast_corners;
        configure_target<FunctionType>(fast_corners, src, corners, &num_corners, threshold, suppress_nonmax, border_mode, constant_border_value);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), raw);

        // Compute function
        fast_corners.run();

        return corners;
    }

    std::vector<KeyPoint> compute_reference(const std::string &image, Format format, float threshold, bool suppress_nonmax, BorderMode border_mode, uint8_t constant_border_value)
    {
        // Load the image (cached by the library if loaded before)
        const RawTensor &raw = library->get(image, format);

        // Create reference
        SimpleTensor<T> src{ raw.shape(), format };

        // Fill reference
        fill(src, raw);

        // Compute reference
        return reference::fast_corners<T>(src, threshold, suppress_nonmax, border_mode, constant_border_value);
    }

    ArrayType             _target{};
    std::vector<KeyPoint> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FAST_CORNERS_FIXTURE */
