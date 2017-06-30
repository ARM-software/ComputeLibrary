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

#include "Globals.h"
#include "NEON/NEAccessor.h"
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"
#include "validation/ValidationUserConfiguration.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NESobel3x3.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "PaddingCalculator.h"
#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::neon;
using namespace arm_compute::test::validation;

namespace
{
constexpr unsigned int filter_size = 3;              /** Size of the kernel/filter in number of elements. */
constexpr BorderSize   border_size(filter_size / 2); /** Border size of the kernel/filter around its central element. */

/** Compute Neon Sobel 3x3 function.
 *
 * @param[in] shape                 Shape of the input and output tensors.
 * @param[in] border_mode           BorderMode used by the input tensor
 * @param[in] constant_border_value Constant to use if @p border_mode == CONSTANT
 *
 * @return Computed output tensor.
 */
std::pair<Tensor, Tensor> compute_sobel_3x3(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create tensors
    Tensor src   = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst_x = create_tensor<Tensor>(shape, DataType::S16);
    Tensor dst_y = create_tensor<Tensor>(shape, DataType::S16);

    src.info()->set_format(Format::U8);
    dst_x.info()->set_format(Format::S16);
    dst_y.info()->set_format(Format::S16);

    // Create sobel image configure function
    NESobel3x3 sobel_3x3;
    sobel_3x3.configure(&src, &dst_x, &dst_y, border_mode, constant_border_value);

    // Allocate tensors
    src.allocator()->allocate();
    dst_x.allocator()->allocate();
    dst_y.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst_x.info()->is_resizable());
    BOOST_TEST(!dst_y.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(NEAccessor(src), 0);

    // Compute function
    sobel_3x3.run();

    return std::make_pair(std::move(dst_x), std::move(dst_y));
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(Sobel3x3)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * BorderModes(), shape, border_mode)
{
    // Create tensors
    Tensor src   = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst_x = create_tensor<Tensor>(shape, DataType::S16);
    Tensor dst_y = create_tensor<Tensor>(shape, DataType::S16);

    src.info()->set_format(Format::U8);
    dst_x.info()->set_format(Format::S16);
    dst_y.info()->set_format(Format::S16);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst_x.info()->is_resizable());
    BOOST_TEST(dst_y.info()->is_resizable());

    // Create sobel 3x3 configure function
    NESobel3x3 sobel_3x3;
    sobel_3x3.configure(&src, &dst_x, &dst_y, border_mode);

    // Validate valid region
    const ValidRegion src_valid_region = shape_to_valid_region(shape);
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, border_size);

    validate(src.info()->valid_region(), src_valid_region);
    validate(dst_x.info()->valid_region(), dst_valid_region);
    validate(dst_y.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);

    calculator.set_border_mode(border_mode);
    calculator.set_border_size(1);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-1);

    const PaddingSize src_padding = calculator.required_padding();

    validate(src.info()->padding(), src_padding);
    validate(dst_x.info()->padding(), dst_padding);
    validate(dst_y.info()->padding(), dst_padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * BorderModes(), shape, border_mode)
{
    uint8_t constant_border_value = 0;

    // Generate a random constant value if border_mode is constant
    if(border_mode == BorderMode::CONSTANT)
    {
        std::mt19937                           gen(user_config.seed.get());
        std::uniform_int_distribution<uint8_t> distribution(0, 255);
        constant_border_value = distribution(gen);
    }

    // Compute function
    std::pair<Tensor, Tensor> dst = compute_sobel_3x3(shape, border_mode, constant_border_value);

    // Compute reference
    std::pair<RawTensor, RawTensor> ref_dst = Reference::compute_reference_sobel_3x3(shape, border_mode, constant_border_value);

    // Calculate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, border_size);

    // Validate output
    validate(NEAccessor(dst.first), ref_dst.first, valid_region);
    validate(NEAccessor(dst.second), ref_dst.second, valid_region);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * BorderModes(), shape, border_mode)
{
    uint8_t constant_border_value = 0;

    // Generate a random constant value if border_mode is constant
    if(border_mode == BorderMode::CONSTANT)
    {
        std::mt19937                           gen(user_config.seed.get());
        std::uniform_int_distribution<uint8_t> distribution(0, 255);
        constant_border_value = distribution(gen);
    }

    // Compute function
    std::pair<Tensor, Tensor> dst = compute_sobel_3x3(shape, border_mode, constant_border_value);

    // Compute reference
    std::pair<RawTensor, RawTensor> ref_dst = Reference::compute_reference_sobel_3x3(shape, border_mode, constant_border_value);

    // Calculate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, border_size);

    // Validate output
    validate(NEAccessor(dst.first), ref_dst.first, valid_region);
    validate(NEAccessor(dst.second), ref_dst.second, valid_region);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
