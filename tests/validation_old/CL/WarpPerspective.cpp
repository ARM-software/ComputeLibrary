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
#include "arm_compute/runtime/CL/functions/CLWarpPerspective.h"
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
/** Compute Warp Perspective function.
 *
 * @param[in] input                 Shape of the input and output tensors.
 * @param[in] matrix                The perspective matrix. Must be 3x3 of type float.
 * @param[in] policy                The interpolation type.
 * @param[in] border_mode           Strategy to use for borders.
 * @param[in] constant_border_value Constant value to use for borders if border_mode is set to CONSTANT.
 *
 * @return Computed output tensor.
 */
CLTensor compute_warp_perspective(const TensorShape &shape, const float *matrix, InterpolationPolicy policy,
                                  BorderMode border_mode, uint8_t constant_border_value)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U8);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::U8);

    // Create and configure function
    CLWarpPerspective warp_perspective;
    warp_perspective.configure(&src, &dst, matrix, policy, border_mode, constant_border_value);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(CLAccessor(src), 0);

    // Compute function
    warp_perspective.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)
BOOST_AUTO_TEST_SUITE(WarpPerspective)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes())
                     * boost::unit_test::data::make({ InterpolationPolicy::BILINEAR, InterpolationPolicy::NEAREST_NEIGHBOR }) * BorderModes(),
                     shape, policy, border_mode)
{
    uint8_t constant_border_value = 0;

    // Generate a random constant value if border_mode is constant
    if(border_mode == BorderMode::CONSTANT)
    {
        std::mt19937                           gen(user_config.seed.get());
        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        constant_border_value = distribution_u8(gen);
    }

    std::array<float, 9> matrix;
    fill_warp_matrix<9>(matrix, 3, 3);

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U8);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::U8);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    CLWarpPerspective warp_perspective;
    warp_perspective.configure(&src, &dst, matrix.data(), policy, border_mode, constant_border_value);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);

    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 4);
    calculator.set_border_mode(border_mode);

    const PaddingSize read_padding(1);
    const PaddingSize write_padding = calculator.required_padding(PaddingCalculator::Option::EXCLUDE_BORDER);

    validate(src.info()->padding(), read_padding);
    validate(dst.info()->padding(), write_padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes()
                     * boost::unit_test::data::make({ InterpolationPolicy::BILINEAR, InterpolationPolicy::NEAREST_NEIGHBOR })
                     * BorderModes(),
                     shape, policy, border_mode)
{
    uint8_t constant_border_value = 0;

    // Generate a random constant value if border_mode is constant
    if(border_mode == BorderMode::CONSTANT)
    {
        std::mt19937                           gen(user_config.seed.get());
        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        constant_border_value = distribution_u8(gen);
    }

    // Create the valid mask Tensor
    RawTensor valid_mask(shape, DataType::U8);

    // Create the matrix
    std::array<float, 9> matrix;
    fill_warp_matrix<9>(matrix, 3, 3);

    // Compute function
    CLTensor dst = compute_warp_perspective(shape, matrix.data(), policy, border_mode, constant_border_value);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_warp_perspective(shape, valid_mask, matrix.data(), policy, border_mode, constant_border_value);

    // Validate output
    validate(CLAccessor(dst), ref_dst, valid_mask, 1, 0.2f);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes()
                     * boost::unit_test::data::make({ InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR }) * BorderModes(),
                     shape, policy, border_mode)
{
    uint8_t constant_border_value = 0;

    // Generate a random constant value if border_mode is constant
    if(border_mode == BorderMode::CONSTANT)
    {
        std::mt19937                           gen(user_config.seed.get());
        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        constant_border_value = distribution_u8(gen);
    }

    // Create the valid mask Tensor
    RawTensor valid_mask(shape, DataType::U8);

    // Create the matrix
    std::array<float, 9> matrix;
    fill_warp_matrix<9>(matrix, 3, 3);

    // Compute function
    CLTensor dst = compute_warp_perspective(shape, matrix.data(), policy, border_mode, constant_border_value);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_warp_perspective(shape, valid_mask, matrix.data(), policy, border_mode, constant_border_value);

    // Validate output
    validate(CLAccessor(dst), ref_dst, valid_mask, 1, 0.2f);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
