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

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::cl;
using namespace arm_compute::test::validation;

namespace
{
/** Tolerance for float operations */
const float tolerance = 0.000001f;
/** Tolerance for fixed point operations */
const float tolerance_fixed_point = 2.f;

/** Compute OpenCL softmax layer function.
 *
 * @param[in] shape                Shape of the input and output tensors.
 * @param[in] dt                   Shape Data type of tensors.
 * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of fixed point numbers.
 *
 * @return Computed output tensor.
 */
CLTensor compute_softmax_layer(const TensorShape &shape, DataType dt, int fixed_point_position = 0)
{
    // Create tensors
    CLTensor src = create_tensor(shape, dt, 1, fixed_point_position);
    CLTensor dst = create_tensor(shape, dt, 1, fixed_point_position);

    // Create and configure function
    CLSoftmaxLayer smx_layer;
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
        library->fill(CLAccessor(src), distribution, 0);
    }
    else
    {
        int                             one_fixed = 1 << fixed_point_position;
        std::uniform_int_distribution<> distribution(-one_fixed, one_fixed);
        library->fill(CLAccessor(src), distribution, 0);
    }

    // Compute function
    smx_layer.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)
BOOST_AUTO_TEST_SUITE(SoftmaxLayer)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * CNNDataTypes(), shape, dt)
{
    // Set fixed point position data type allowed
    int fixed_point_position = (arm_compute::is_data_type_fixed_point(dt)) ? 3 : 0;

    // Create tensors
    CLTensor src = create_tensor(shape, dt, 1, fixed_point_position);
    CLTensor dst = create_tensor(shape, dt, 1, fixed_point_position);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    CLSoftmaxLayer smx_layer;
    smx_layer.configure(&src, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * CNNFloatDataTypes(), shape, dt)
{
    // Compute function
    CLTensor dst = compute_softmax_layer(shape, dt);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, dt);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * CNNFloatDataTypes(), shape, dt)
{
    // Compute function
    CLTensor dst = compute_softmax_layer(shape, dt);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, dt);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * CNNFixedPointDataTypes() * boost::unit_test::data::xrange(1, 6),
                     shape, dt, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_softmax_layer(shape, dt, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, dt, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_fixed_point);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeShapes() * CNNFixedPointDataTypes() * boost::unit_test::data::xrange(1, 6),
                     shape, dt, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_softmax_layer(shape, dt, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_softmax_layer(shape, dt, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_fixed_point);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
