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
#include "NEON/Accessor.h"
#include "PaddingCalculator.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/validation_old/Datasets.h"
#include "tests/validation_old/Helpers.h"
#include "tests/validation_old/Reference.h"
#include "tests/validation_old/Validation.h"
#include "tests/validation_old/ValidationUserConfiguration.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NENonLinearFilter.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/validation_old/boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
/** Compute NonLinearFilter function.
 *
     * @param[in] input                 Shape of the input and output tensors.
     * @param[in] function              Non linear function to perform
     * @param[in] mask_size             Mask size. Supported sizes: 3, 5
     * @param[in] pattern               Mask pattern
     * @param[in] mask                  The given mask. Will be used only if pattern is specified to PATTERN_OTHER
     * @param[in] border_mode           Strategy to use for borders.
     * @param[in] constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
 *
 * @return Computed output tensor.
 */
Tensor compute_non_linear_filter(const TensorShape &shape, NonLinearFilterFunction function, unsigned int mask_size,
                                 MatrixPattern pattern, const uint8_t *mask, BorderMode border_mode,
                                 uint8_t constant_border_value)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U8);

    // Create and configure function
    NENonLinearFilter filter;
    filter.configure(&src, &dst, function, mask_size, pattern, mask, border_mode, constant_border_value);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(Accessor(src), 0);

    // Compute function
    filter.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(NonLinearFilter)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes())
                     * NonLinearFilterFunctions() * boost::unit_test::data::make({ 3U, 5U })
                     * MatrixPatterns() * BorderModes(),
                     shape, function, mask_size, pattern, border_mode)
{
    std::mt19937                           generator(user_config.seed.get());
    std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
    const uint8_t                          constant_border_value = distribution_u8(generator);

    // Create the mask
    uint8_t mask[mask_size * mask_size];
    fill_mask_from_pattern(mask, mask_size, mask_size, pattern);
    const auto half_mask_size = static_cast<int>(mask_size / 2);

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U8);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    NENonLinearFilter filter;
    filter.configure(&src, &dst, function, mask_size, pattern, mask, border_mode, constant_border_value);

    // Validate valid region
    const ValidRegion src_valid_region = shape_to_valid_region(shape);
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, BorderSize(half_mask_size));

    validate(src.info()->valid_region(), src_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), ((MatrixPattern::OTHER == pattern) ? 1 : 8));
    calculator.set_border_mode(border_mode);
    calculator.set_border_size(half_mask_size);

    const PaddingSize write_padding = calculator.required_padding(PaddingCalculator::Option::EXCLUDE_BORDER);

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-half_mask_size);

    const PaddingSize read_padding = calculator.required_padding(PaddingCalculator::Option::INCLUDE_BORDER);

    validate(src.info()->padding(), read_padding);
    validate(dst.info()->padding(), write_padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes()
                     * NonLinearFilterFunctions() * boost::unit_test::data::make({ 3U, 5U })
                     * MatrixPatterns() * BorderModes(),
                     shape, function, mask_size, pattern, border_mode)
{
    std::mt19937                           generator(user_config.seed.get());
    std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
    const uint8_t                          constant_border_value = distribution_u8(generator);

    // Create the mask
    uint8_t mask[mask_size * mask_size];
    fill_mask_from_pattern(mask, mask_size, mask_size, pattern);

    // Compute function
    Tensor dst = compute_non_linear_filter(shape, function, mask_size, pattern, mask, border_mode, constant_border_value);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_non_linear_filter(shape, function, mask_size, pattern, mask, border_mode, constant_border_value);

    // Calculate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, BorderSize(static_cast<int>(mask_size / 2)));

    // Validate output
    validate(Accessor(dst), ref_dst, valid_region);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes()
                     * NonLinearFilterFunctions() * boost::unit_test::data::make({ 3U, 5U })
                     * MatrixPatterns() * BorderModes(),
                     shape, function, mask_size, pattern, border_mode)
{
    std::mt19937                           generator(user_config.seed.get());
    std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
    const uint8_t                          constant_border_value = distribution_u8(generator);

    // Create the mask
    uint8_t mask[mask_size * mask_size];
    fill_mask_from_pattern(mask, mask_size, mask_size, pattern);

    // Compute function
    Tensor dst = compute_non_linear_filter(shape, function, mask_size, pattern, mask, border_mode, constant_border_value);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_non_linear_filter(shape, function, mask_size, pattern, mask, border_mode, constant_border_value);

    // Calculate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, BorderSize(static_cast<int>(mask_size / 2)));

    // Validate output
    validate(Accessor(dst), ref_dst, valid_region);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
