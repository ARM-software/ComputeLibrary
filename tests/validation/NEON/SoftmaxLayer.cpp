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
#include "NEON/Accessor.h"
#include "PaddingCalculator.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
/** Tolerance for float operations */
const float tolerance = 0.000001f;
/** Tolerance for fixed point operations */
const float tolerance_fixed_point = 2.f;

/** Compute Neon softmax layer function.
 *
 * @param[in] shape                Shape of the input and output tensors.
 * @param[in] dt                   Shape Data type of tensors.
 * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of fixed point numbers.
 *
 * @return Computed output tensor.
 */
Tensor compute_softmax_layer(const TensorShape &shape, DataType dt, int fixed_point_position = 0)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);

    // Create and configure function
    NESoftmaxLayer smx_layer;
    smx_layer.configure(&src, &dst);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    if(arm_compute::is_data_type_float(dt))
    {
        std::uniform_real_distribution<> distribution(-1000.f, 1000.f);
        library->fill(Accessor(src), distribution, 0);
    }
    else
    {
        int                             one_fixed = 1 << fixed_point_position;
        std::uniform_int_distribution<> distribution(-one_fixed, one_fixed);
        library->fill(Accessor(src), distribution, 0);
    }

    // Compute function
    smx_layer.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(SoftmaxLayer)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * CNNDataTypes(), shape, dt)
{
    // Set fixed point position data type allowed
    int fixed_point_position = (arm_compute::is_data_type_fixed_point(dt)) ? 3 : 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    NESoftmaxLayer smx_layer;
    smx_layer.configure(&src, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const int         step    = 16 / arm_compute::data_size_from_type(dt);
    const PaddingSize padding = PaddingCalculator(shape.x(), step).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * CNNFloatDataTypes(), shape, dt)
{
    // Compute function
    Tensor dst = compute_softmax_layer(shape, dt);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, dt);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * CNNFloatDataTypes(), shape, dt)
{
    // Compute function
    Tensor dst = compute_softmax_layer(shape, dt);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, dt);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_AUTO_TEST_SUITE(QS8)
// Testing for fixed point position [1,6) as reciprocal limits the maximum fixed point position to 5
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::xrange(1, 6),
                     shape, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_softmax_layer(shape, DataType::QS8, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, DataType::QS8, fixed_point_position);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_fixed_point);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::xrange(1, 6),
                     shape, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_softmax_layer(shape, DataType::QS8, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, DataType::QS8, fixed_point_position);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_fixed_point);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(QS16)
// Testing for fixed point position [1,14) as reciprocal limits the maximum fixed point position to 14
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * boost::unit_test::data::xrange(1, 14),
                     shape, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_softmax_layer(shape, DataType::QS16, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, DataType::QS16, fixed_point_position);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_fixed_point);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * boost::unit_test::data::xrange(1, 14),
                     shape, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_softmax_layer(shape, DataType::QS16, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, DataType::QS16, fixed_point_position);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_fixed_point);
}
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
