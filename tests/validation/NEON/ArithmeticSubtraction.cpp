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
#include "NEON/Helper.h"
#include "NEON/NEAccessor.h"
#include "PaddingCalculator.h"
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h"
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
/** Compute Neon arithmetic subtraction function.
 *
 * @param[in] shape  Shape of the input and output tensors.
 * @param[in] dt_in0 Data type of first input tensor.
 * @param[in] dt_in1 Data type of second input tensor.
 * @param[in] dt_out Data type of the output tensor.
 * @param[in] policy Overflow policy of the operation.
 *
 * @return Computed output tensor.
 */
Tensor compute_arithmetic_subtraction(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, ConvertPolicy policy)
{
    // Create tensors
    Tensor src1 = create_tensor(shape, dt_in0);
    Tensor src2 = create_tensor(shape, dt_in1);
    Tensor dst  = create_tensor(shape, dt_out);

    // Create and configure function
    NEArithmeticSubtraction sub;
    sub.configure(&src1, &src2, &dst, policy);

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
    sub.run();

    return dst;
}

void validate_configuration(const Tensor &src1, const Tensor &src2, Tensor &dst, TensorShape shape, ConvertPolicy policy)
{
    BOOST_TEST(src1.info()->is_resizable());
    BOOST_TEST(src2.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    NEArithmeticSubtraction sub;
    sub.configure(&src1, &src2, &dst, policy);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src1.info()->valid_region(), valid_region);
    validate(src2.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding(0, PaddingCalculator(shape.x(), 16).required_padding(), 0, 0);
    validate(src1.info()->padding(), padding);
    validate(src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(ArithmeticSubtraction)

BOOST_AUTO_TEST_SUITE(U8)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, policy)
{
    // Create tensors
    Tensor src1 = create_tensor(shape, DataType::U8);
    Tensor src2 = create_tensor(shape, DataType::U8);
    Tensor dst  = create_tensor(shape, DataType::U8);

    validate_configuration(src1, src2, dst, shape, policy);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, policy)
{
    // Compute function
    Tensor dst = compute_arithmetic_subtraction(shape, DataType::U8, DataType::U8, DataType::U8, policy);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_subtraction(shape, DataType::U8, DataType::U8, DataType::U8, policy);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(S16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ DataType::U8, DataType::S16 }) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, dt, policy)
{
    // Create tensors
    Tensor src1 = create_tensor(shape, dt);
    Tensor src2 = create_tensor(shape, DataType::S16);
    Tensor dst  = create_tensor(shape, DataType::S16);

    validate_configuration(src1, src2, dst, shape, policy);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ DataType::U8, DataType::S16 }) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, dt, policy)
{
    // Compute function
    Tensor dst = compute_arithmetic_subtraction(shape, dt, DataType::S16, DataType::S16, policy);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_subtraction(shape, dt, DataType::S16, DataType::S16, policy);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ DataType::U8, DataType::S16 }) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, dt, policy)
{
    // Compute function
    Tensor dst = compute_arithmetic_subtraction(shape, dt, DataType::S16, DataType::S16, policy);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_subtraction(shape, dt, DataType::S16, DataType::S16, policy);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(F32)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, policy)
{
    // Create tensors
    Tensor src1 = create_tensor(shape, DataType::F32);
    Tensor src2 = create_tensor(shape, DataType::F32);
    Tensor dst  = create_tensor(shape, DataType::F32);

    validate_configuration(src1, src2, dst, shape, policy);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes(), shape)
{
    // Compute function
    Tensor dst = compute_arithmetic_subtraction(shape, DataType::F32, DataType::F32, DataType::F32, ConvertPolicy::WRAP);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_subtraction(shape, DataType::F32, DataType::F32, DataType::F32, ConvertPolicy::WRAP);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, policy)
{
    // Compute function
    Tensor dst = compute_arithmetic_subtraction(shape, DataType::F32, DataType::F32, DataType::F32, policy);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_subtraction(shape, DataType::F32, DataType::F32, DataType::F32, policy);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
