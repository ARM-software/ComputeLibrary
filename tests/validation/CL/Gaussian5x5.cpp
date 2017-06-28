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
#include "CL/Helper.h"
#include "Globals.h"
#include "PaddingCalculator.h"
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"
#include "validation/ValidationUserConfiguration.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLGaussian5x5.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::cl;
using namespace arm_compute::test::validation;

namespace
{
constexpr unsigned int filter_size = 5;              /** Size of the kernel/filter in number of elements. */
constexpr BorderSize   border_size(filter_size / 2); /** Border size of the kernel/filter around its central element. */

/** Compute CL gaussian5x5 filter.
 *
 * @param[in] shape                 Shape of the input and output tensors.
 * @param[in] border_mode           BorderMode used by the input tensor.
 * @param[in] constant_border_value Constant to use if @p border_mode == CONSTANT.
 *
 * @return Computed output tensor.
 */
CLTensor compute_gaussian5x5(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create tensors
    CLTensor src = create_tensor(shape, DataType::U8);
    CLTensor dst = create_tensor(shape, DataType::U8);

    // Create and configure function
    CLGaussian5x5 gaussian5x5;
    gaussian5x5.configure(&src, &dst, border_mode, constant_border_value);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(CLAccessor(src), 0);

    // Compute function
    gaussian5x5.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)
BOOST_AUTO_TEST_SUITE(Gaussian5x5)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * BorderModes(), shape, border_mode)
{
    // Create tensors
    CLTensor src = create_tensor(shape, DataType::U8);
    CLTensor dst = create_tensor(shape, DataType::U8);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    CLGaussian5x5 gaussian5x5;
    gaussian5x5.configure(&src, &dst, border_mode);

    // Validate valid region
    const ValidRegion src_valid_region = shape_to_valid_region(shape);
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, border_size);
    validate(src.info()->valid_region(), src_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);
    calculator.set_border_size(2);
    calculator.set_border_mode(border_mode);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-2);

    const PaddingSize src_padding = calculator.required_padding();

    validate(src.info()->padding(), src_padding);
    validate(dst.info()->padding(), dst_padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * BorderModes(), shape, border_mode)
{
    std::mt19937                           gen(user_config.seed.get());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);
    const uint8_t                          border_value = distribution(gen);

    // Compute function
    CLTensor dst = compute_gaussian5x5(shape, border_mode, border_value);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_gaussian5x5(shape, border_mode, border_value);

    // Validate output
    validate(CLAccessor(dst), ref_dst, shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, border_size));
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * BorderModes(), shape, border_mode)
{
    std::mt19937                           gen(user_config.seed.get());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);
    const uint8_t                          border_value = distribution(gen);

    // Compute function
    CLTensor dst = compute_gaussian5x5(shape, border_mode, border_value);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_gaussian5x5(shape, border_mode, border_value);

    // Validate output
    validate(CLAccessor(dst), ref_dst, shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, border_size));
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
