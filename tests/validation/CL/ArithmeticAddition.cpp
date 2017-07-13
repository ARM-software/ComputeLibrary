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
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLArithmeticAddition.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::cl;
using namespace arm_compute::test::validation;

namespace
{
/** Compute Neon arithmetic addition function.
 *
 * @param[in] shape                Shape of the input and output tensors.
 * @param[in] dt_in0               Data type of first input tensor.
 * @param[in] dt_in1               Data type of second input tensor.
 * @param[in] dt_out               Data type of the output tensor.
 * @param[in] policy               Overflow policy of the operation.
 * @param[in] fixed_point_position (Optional) Fixed point position that expresses the number of bits for the fractional part of the number when the tensor's data type is QS8 or QS16 (default = 0).
 *
 * @return Computed output tensor.
 */
CLTensor compute_arithmetic_addition(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, ConvertPolicy policy, int fixed_point_position = 0)
{
    // Create tensors
    CLTensor src1 = create_tensor<CLTensor>(shape, dt_in0, 1, fixed_point_position);
    CLTensor src2 = create_tensor<CLTensor>(shape, dt_in1, 1, fixed_point_position);
    CLTensor dst  = create_tensor<CLTensor>(shape, dt_out, 1, fixed_point_position);

    // Create and configure function
    CLArithmeticAddition add;
    add.configure(&src1, &src2, &dst, policy);

    // Allocate tensors
    src1.allocator()->allocate();
    src2.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src1.info()->is_resizable());
    BOOST_TEST(!src2.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(CLAccessor(src1), 0);
    library->fill_tensor_uniform(CLAccessor(src2), 1);

    // Compute function
    add.run();

    return dst;
}

void validate_configuration(const CLTensor &src1, const CLTensor &src2, CLTensor &dst, TensorShape shape, ConvertPolicy policy)
{
    BOOST_TEST(src1.info()->is_resizable());
    BOOST_TEST(src2.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    CLArithmeticAddition add;
    add.configure(&src1, &src2, &dst, policy);

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
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)
BOOST_AUTO_TEST_SUITE(ArithmeticAddition)

BOOST_AUTO_TEST_SUITE(U8)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, policy)
{
    // Create tensors
    CLTensor src1 = create_tensor<CLTensor>(shape, DataType::U8);
    CLTensor src2 = create_tensor<CLTensor>(shape, DataType::U8);
    CLTensor dst  = create_tensor<CLTensor>(shape, DataType::U8);

    validate_configuration(src1, src2, dst, shape, policy);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, policy)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, DataType::U8, DataType::U8, DataType::U8, policy);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, DataType::U8, DataType::U8, DataType::U8, policy);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(S16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ DataType::U8, DataType::S16 }) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, dt, policy)
{
    // Create tensors
    CLTensor src1 = create_tensor<CLTensor>(shape, dt);
    CLTensor src2 = create_tensor<CLTensor>(shape, DataType::S16);
    CLTensor dst  = create_tensor<CLTensor>(shape, DataType::S16);

    validate_configuration(src1, src2, dst, shape, policy);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ DataType::U8, DataType::S16 }) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, dt, policy)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, dt, DataType::S16, DataType::S16, policy);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, dt, DataType::S16, DataType::S16, policy);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ DataType::U8, DataType::S16 }) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, dt, policy)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, dt, DataType::S16, DataType::S16, policy);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, dt, DataType::S16, DataType::S16, policy);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_AUTO_TEST_SUITE(QS8)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * ConvertPolicies() * boost::unit_test::data::xrange(1, 7),
                     shape, policy, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, DataType::QS8, DataType::QS8, DataType::QS8, policy, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, DataType::QS8, DataType::QS8, DataType::QS8, policy, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * ConvertPolicies() * boost::unit_test::data::xrange(1, 7),
                     shape, policy, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, DataType::QS8, DataType::QS8, DataType::QS8, policy, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, DataType::QS8, DataType::QS8, DataType::QS8, policy, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(QS16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * ConvertPolicies() * boost::unit_test::data::xrange(1, 15),
                     shape, policy, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, DataType::QS16, DataType::QS16, DataType::QS16, policy, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, DataType::QS16, DataType::QS16, DataType::QS16, policy, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * ConvertPolicies() * boost::unit_test::data::xrange(1, 15),
                     shape, policy, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, DataType::QS16, DataType::QS16, DataType::QS16, policy, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, DataType::QS16, DataType::QS16, DataType::QS16, policy, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()

#ifdef ARM_COMPUTE_ENABLE_FP16
BOOST_AUTO_TEST_SUITE(F16)
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes(), shape)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, DataType::F16, DataType::F16, DataType::F16, ConvertPolicy::WRAP);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, DataType::F16, DataType::F16, DataType::F16, ConvertPolicy::WRAP);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()
#endif /* ARM_COMPUTE_ENABLE_FP16 */

BOOST_AUTO_TEST_SUITE(F32)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, policy)
{
    // Create tensors
    CLTensor src1 = create_tensor<CLTensor>(shape, DataType::F32);
    CLTensor src2 = create_tensor<CLTensor>(shape, DataType::F32);
    CLTensor dst  = create_tensor<CLTensor>(shape, DataType::F32);

    validate_configuration(src1, src2, dst, shape, policy);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes(), shape)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, DataType::F32, DataType::F32, DataType::F32, ConvertPolicy::WRAP);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, DataType::F32, DataType::F32, DataType::F32, ConvertPolicy::WRAP);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }),
                     shape, policy)
{
    // Compute function
    CLTensor dst = compute_arithmetic_addition(shape, DataType::F32, DataType::F32, DataType::F32, policy);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_arithmetic_addition(shape, DataType::F32, DataType::F32, DataType::F32, policy);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
