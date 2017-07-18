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
#include "AssetsLibrary.h"
#include "Globals.h"
#include "NEON/NEAccessor.h"
#include "PaddingCalculator.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseXor.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::neon;
using namespace arm_compute::test::validation;

namespace
{
/** Compute Neon bitwise xor function.
 *
 * @param[in] shape Shape of the input and output tensors.
 *
 * @return Computed output tensor.
 */
Tensor compute_bitwise_xor(const TensorShape &shape)
{
    // Create tensors
    Tensor src1 = create_tensor<Tensor>(shape, DataType::U8);
    Tensor src2 = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst  = create_tensor<Tensor>(shape, DataType::U8);

    // Create xor configure function
    NEBitwiseXor bxor;
    bxor.configure(&src1, &src2, &dst);

    // Allocate tensors
    src1.allocator()->allocate();
    src2.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src1.info()->is_resizable());
    BOOST_TEST(!src2.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(NEAccessor(src1), 0);
    library->fill_tensor_uniform(NEAccessor(src2), 1);

    // Compute function
    bxor.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(BitwiseXor)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, SmallShapes() + LargeShapes(), shape)
{
    // Create tensors
    Tensor src1 = create_tensor<Tensor>(shape, DataType::U8);
    Tensor src2 = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst  = create_tensor<Tensor>(shape, DataType::U8);

    BOOST_TEST(src1.info()->is_resizable());
    BOOST_TEST(src2.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create xor configure function
    NEBitwiseXor bxor;
    bxor.configure(&src1, &src2, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src1.info()->valid_region(), valid_region);
    validate(src2.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src1.info()->padding(), padding);
    validate(src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes(), shape)
{
    // Compute function
    Tensor dst = compute_bitwise_xor(shape);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_bitwise_xor(shape);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes(), shape)
{
    // Compute function
    Tensor dst = compute_bitwise_xor(shape);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_bitwise_xor(shape);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
