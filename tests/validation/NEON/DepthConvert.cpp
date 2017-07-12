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
#include "PaddingCalculator.h"
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConvert.h"
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
/** Compute Neon depth convert function.
 *
 * @param[in] shape                    Shape of the input and output tensors.
 * @param[in] dt_in                    Data type of input tensor.
 * @param[in] dt_out                   Data type of the output tensor.
 * @param[in] policy                   Conversion policy.
 * @param[in] shift                    Value for down/up conversions. Must be 0 <= shift < 8.
 * @param[in] fixed_point_position_in  (Optional) Fixed point position for the input tensor.
 * @param[in] fixed_point_position_out (Optional) Fixed point position for the output tensor.
 *
 * @return Computed output tensor.
 */
Tensor compute_depth_convert(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy,
                             uint32_t shift, uint32_t fixed_point_position_in = 0, uint32_t fixed_point_position_out = 0)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, dt_in, 1, fixed_point_position_in);
    Tensor dst = create_tensor<Tensor>(shape, dt_out, 1, fixed_point_position_out);

    // Create and configure function
    NEDepthConvert depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(NEAccessor(src), 0);

    // Compute function
    depth_convert.run();

    return dst;
}
/** Configure and validate region/padding function.
 *
 * @param[in] shape                    Shape of the input and output tensors.
 * @param[in] dt_in                    Data type of input tensor.
 * @param[in] dt_out                   Data type of the output tensor.
 * @param[in] policy                   Conversion policy.
 * @param[in] shift                    Value for down/up conversions. Must be 0 <= shift < 8.
 * @param[in] fixed_point_position_in  (Optional) Fixed point position for the input tensor.
 * @param[in] fixed_point_position_out (Optional) Fixed point position for the output tensor.
 *
 */

void compute_configure_validate(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy,
                                uint32_t shift, uint32_t fixed_point_position_in = 0, uint32_t fixed_point_position_out = 0)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, dt_in, 1, fixed_point_position_in);
    Tensor dst = create_tensor<Tensor>(shape, dt_out, 1, fixed_point_position_out);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    NEDepthConvert depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(DepthConvert)

BOOST_AUTO_TEST_SUITE(QS8_to_QS8)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * (boost::unit_test::data::make({ 1, 3, 5, 6 }) ^ boost::unit_test::data::make({ 6, 5, 1, 3 })),
                     shape, policy, fixed_point_position_in, fixed_point_position_out)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::QS8, DataType::QS8, policy, 0, fixed_point_position_in, fixed_point_position_out);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * (boost::unit_test::data::make({ 1, 3, 5, 6 }) ^ boost::unit_test::data::make({ 6, 5, 1, 3 })),
                     shape, policy, fixed_point_position_in, fixed_point_position_out)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::QS8, DataType::QS8, policy, 0, fixed_point_position_in, fixed_point_position_out);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::QS8, DataType::QS8, policy, 0, fixed_point_position_in, fixed_point_position_out);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(QS8_to_F32)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 7, 1),
                     shape, policy, fixed_point_position)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::QS8, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 7, 1),
                     shape, policy, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::QS8, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::QS8, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 7, 1),
                     shape, policy, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::QS8, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::QS8, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(F32_to_QS8)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 7, 1),
                     shape, policy, fixed_point_position)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::F32, DataType::QS8, policy, 0, fixed_point_position, fixed_point_position);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 7, 1),
                     shape, policy, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::F32, DataType::QS8, policy, 0, fixed_point_position, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::F32, DataType::QS8, policy, 0, fixed_point_position, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 7, 1),
                     shape, policy, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::F32, DataType::QS8, policy, 0, fixed_point_position, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::F32, DataType::QS8, policy, 0, fixed_point_position, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(QS16_to_QS16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * (boost::unit_test::data::make({ 3, 6, 7, 13, 14 }) ^ boost::unit_test::data::make({ 5, 10, 14, 4, 7 })),
                     shape, policy, fixed_point_position_in, fixed_point_position_out)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::QS16, DataType::QS16, policy, 0, fixed_point_position_in, fixed_point_position_out);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * (boost::unit_test::data::make({ 3, 6, 7, 13, 14 }) ^ boost::unit_test::data::make({ 5, 10, 14, 4, 7 })),
                     shape, policy, fixed_point_position_in, fixed_point_position_out)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::QS16, DataType::QS16, policy, 0, fixed_point_position_in, fixed_point_position_out);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::QS16, DataType::QS16, policy, 0, fixed_point_position_in, fixed_point_position_out);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(QS16_to_F32)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 15, 1),
                     shape, policy, fixed_point_position)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::QS16, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 15, 1),
                     shape, policy, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::QS16, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::QS16, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 15, 1),
                     shape, policy, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::QS16, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::QS16, DataType::F32, policy, 0, fixed_point_position, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(F32_to_QS16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 7, 1),
                     shape, policy, fixed_point_position)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::F32, DataType::QS16, policy, 0, fixed_point_position, fixed_point_position);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 15, 1),
                     shape, policy, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::F32, DataType::QS16, policy, 0, fixed_point_position, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::F32, DataType::QS16, policy, 0, fixed_point_position, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE })
                     * boost::unit_test::data::xrange(1, 15, 1),
                     shape, policy, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::F32, DataType::QS16, policy, 0, fixed_point_position, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::F32, DataType::QS16, policy, 0, fixed_point_position, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(U8_to_U16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))

BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::U8, DataType::U16, policy, shift, 0);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U8, DataType::U16, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U8, DataType::U16, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U8, DataType::U16, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U8, DataType::U16, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(U8_to_S16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::U8, DataType::S16, policy, shift);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U8, DataType::S16, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U8, DataType::S16, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U8, DataType::S16, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U8, DataType::S16, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(U8_to_S32)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::U8, DataType::S32, policy, shift);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U8, DataType::S32, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U8, DataType::S32, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U8, DataType::S32, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U8, DataType::S32, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(U16_to_U8)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::U16, DataType::U8, policy, shift);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U16, DataType::U8, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U16, DataType::U8, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U16, DataType::U8, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U16, DataType::U8, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(U16_to_U32)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::U16, DataType::U32, policy, shift);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U16, DataType::U32, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U16, DataType::U32, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::U16, DataType::U32, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::U16, DataType::U32, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(S16_to_U8)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::S16, DataType::U8, policy, shift);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::S16, DataType::U8, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::S16, DataType::U8, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::S16, DataType::U8, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::S16, DataType::U8, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(S16_to_S32)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute configure and validate region/padding
    compute_configure_validate(shape, DataType::S16, DataType::S32, policy, shift);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::S16, DataType::S32, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::S16, DataType::S32, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     shape, policy, shift)
{
    // Compute function
    Tensor dst = compute_depth_convert(shape, DataType::S16, DataType::S32, policy, shift);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_convert(shape, DataType::S16, DataType::S32, policy, shift);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
