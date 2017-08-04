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
#include "CL/CLAccessor.h"
#include "Utils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/validation_old/Datasets.h"
#include "tests/validation_old/Reference.h"
#include "tests/validation_old/Validation.h"
#include "tests/validation_old/ValidationUserConfiguration.h"
#include "utils/TypePrinter.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/CL/functions/CLHarrisCorners.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "PaddingCalculator.h"
#include "tests/validation_old/boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
/** Compute CL Harris corners function.
 *
 * @param[in] shape                 Shape of input tensor
 * @param[in] threshold             Minimum threshold with which to eliminate Harris Corner scores (computed using the normalized Sobel kernel).
 * @param[in] min_dist              Radial Euclidean distance for the euclidean distance stage
 * @param[in] sensitivity           Sensitivity threshold k from the Harris-Stephens equation
 * @param[in] gradient_size         The gradient window size to use on the input. The implementation supports 3, 5, and 7
 * @param[in] block_size            The block window size used to compute the Harris Corner score. The implementation supports 3, 5, and 7.
 * @param[in] border_mode           Border mode to use
 * @param[in] constant_border_value Constant value to use for borders if border_mode is set to CONSTANT.
 *
 * @return Computed corners' keypoints.
 */
void compute_harris_corners(const TensorShape &shape, CLKeyPointArray &corners, float threshold, float min_dist, float sensitivity,
                            int32_t gradient_size, int32_t block_size, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U8);
    src.info()->set_format(Format::U8);

    // Create harris corners configure function
    CLHarrisCorners harris_corners;
    harris_corners.configure(&src, threshold, min_dist, sensitivity, gradient_size, block_size, &corners, border_mode, constant_border_value);

    // Allocate tensors
    src.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(CLAccessor(src), 0);

    // Compute function
    harris_corners.run();
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)
BOOST_AUTO_TEST_SUITE(HarrisCorners)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (Small2DShapes() + Large2DShapes()) * BorderModes()
                     * boost::unit_test::data::make({ 3, 5, 7 }) * boost::unit_test::data::make({ 3, 5, 7 }),
                     shape, border_mode, gradient, block)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U8);
    src.info()->set_format(Format::U8);

    CLKeyPointArray corners(shape.total_size());

    uint8_t constant_border_value = 0;

    std::mt19937                          gen(user_config.seed.get());
    std::uniform_real_distribution<float> real_dist(0.01, std::numeric_limits<float>::min());

    const float threshold              = real_dist(gen);
    const float sensitivity            = real_dist(gen);
    const float max_euclidean_distance = 30.f;

    real_dist      = std::uniform_real_distribution<float>(0.f, max_euclidean_distance);
    float min_dist = real_dist(gen);

    // Generate a random constant value if border_mode is constant
    if(border_mode == BorderMode::CONSTANT)
    {
        std::uniform_int_distribution<uint8_t> int_dist(0, 255);
        constant_border_value = int_dist(gen);
    }

    BOOST_TEST(src.info()->is_resizable());

    // Create harris corners configure function
    CLHarrisCorners harris_corners;
    harris_corners.configure(&src, threshold, min_dist, sensitivity, gradient, block, &corners, border_mode, constant_border_value);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);

    validate(src.info()->valid_region(), valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);

    calculator.set_border_mode(border_mode);
    calculator.set_border_size(gradient / 2);
    calculator.set_access_offset(-gradient / 2);
    calculator.set_accessed_elements(16);

    const PaddingSize padding = calculator.required_padding();

    validate(src.info()->padding(), padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, Small2DShapes() * BorderModes() * boost::unit_test::data::make({ 3, 5, 7 }) * boost::unit_test::data::make({ 3, 5, 7 }), shape, border_mode, gradient, block)
{
    uint8_t constant_border_value = 0;

    std::mt19937                          gen(user_config.seed.get());
    std::uniform_real_distribution<float> real_dist(0.01, std::numeric_limits<float>::min());

    const float threshold              = real_dist(gen);
    const float sensitivity            = real_dist(gen);
    const float max_euclidean_distance = 30.f;

    real_dist            = std::uniform_real_distribution<float>(0.f, max_euclidean_distance);
    const float min_dist = real_dist(gen);

    // Generate a random constant value if border_mode is constant
    if(border_mode == BorderMode::CONSTANT)
    {
        std::uniform_int_distribution<uint8_t> int_dist(0, 255);
        constant_border_value = int_dist(gen);
    }

    // Create array of keypoints
    CLKeyPointArray dst(shape.total_size());

    // Compute function
    compute_harris_corners(shape, dst, threshold, min_dist, sensitivity, gradient, block, border_mode, constant_border_value);

    // Compute reference
    KeyPointArray ref_dst = Reference::compute_reference_harris_corners(shape, threshold, min_dist, sensitivity, gradient, block, border_mode, constant_border_value);

    // Validate output
    dst.map();
    validate(dst, ref_dst, 1);
    dst.unmap();
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, Large2DShapes() * BorderModes() * boost::unit_test::data::make({ 3, 5, 7 }) * boost::unit_test::data::make({ 3, 5, 7 }), shape, border_mode, gradient, block)
{
    uint8_t constant_border_value = 0;

    std::mt19937                          gen(user_config.seed.get());
    std::uniform_real_distribution<float> real_dist(0.01, std::numeric_limits<float>::min());

    const float threshold              = real_dist(gen);
    const float sensitivity            = real_dist(gen);
    const float max_euclidean_distance = 30.f;

    real_dist            = std::uniform_real_distribution<float>(0.f, max_euclidean_distance);
    const float min_dist = real_dist(gen);

    // Generate a random constant value if border_mode is constant
    if(border_mode == BorderMode::CONSTANT)
    {
        std::uniform_int_distribution<uint8_t> int_dist(0, 255);
        constant_border_value = int_dist(gen);
    }

    // Create array of keypoints
    CLKeyPointArray dst(shape.total_size());

    // Compute function
    compute_harris_corners(shape, dst, threshold, min_dist, sensitivity, gradient, block, border_mode, constant_border_value);

    // Compute reference
    KeyPointArray ref_dst = Reference::compute_reference_harris_corners(shape, threshold, min_dist, sensitivity, gradient, block, border_mode, constant_border_value);

    // Validate output
    dst.map();
    validate(dst, ref_dst);
    dst.unmap();
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
