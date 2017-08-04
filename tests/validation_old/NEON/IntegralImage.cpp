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
#include "Utils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/validation_old/Datasets.h"
#include "tests/validation_old/Reference.h"
#include "tests/validation_old/Validation.h"
#include "utils/TypePrinter.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEIntegralImage.h"
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
/** Compute Neon integral image function.
 *
 * @param[in] shape Shape of the input and output tensors.
 *
 * @return Computed output tensor.
 */
Tensor compute_integral_image(const TensorShape &shape)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U32);

    // Create integral image configure function
    NEIntegralImage integral_image;
    integral_image.configure(&src, &dst);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(Accessor(src), 0);

    // Compute function
    integral_image.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(IntegralImage)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, SmallShapes() + LargeShapes(), shape)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U32);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create integral image configure function
    NEIntegralImage integral_image;
    integral_image.configure(&src, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize src_padding = PaddingCalculator(shape.x(), 16).required_padding();
    const PaddingSize dst_padding(1, src_padding.right, 0, 1);

    validate(src.info()->padding(), src_padding);
    validate(dst.info()->padding(), dst_padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes(), shape)
{
    // Compute function
    Tensor dst = compute_integral_image(shape);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_integral_image(shape);

    // Validate output
    validate(Accessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes(), shape)
{
    // Compute function
    Tensor dst = compute_integral_image(shape);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_integral_image(shape);

    // Validate output
    validate(Accessor(dst), ref_dst);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
