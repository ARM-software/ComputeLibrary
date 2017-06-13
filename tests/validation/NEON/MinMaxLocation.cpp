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
#include "arm_compute/runtime/NEON/functions/NEMinMaxLocation.h"
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
/** Compute Neon MinMaxLocation function.
     *
     * @param[in]  shape     Shape of the input and output tensors.
     * @param[in]  dt_in     Data type of first input tensor.
     * @param[out] min       Minimum value of tensor
     * @param[out] max       Maximum value of tensor
     * @param[out] min_loc   Array with locations of minimum values
     * @param[out] max_loc   Array with locations of maximum values
     * @param[out] min_count Number of minimum values found
     * @param[out] max_count Number of maximum values found
     *
     * @return Computed output tensor.
     */

void compute_min_max_location(const TensorShape &shape, DataType dt_in, int32_t &min, int32_t &max,
                              Coordinates2DArray &min_loc, Coordinates2DArray &max_loc, uint32_t &min_count, uint32_t &max_count)
{
    // Create tensor
    Tensor src = create_tensor<Tensor>(shape, dt_in);
    src.info()->set_format((dt_in == DataType::U8) ? Format::U8 : Format::S16);

    // Create and configure min_max_location configure function
    NEMinMaxLocation min_max_loc;
    min_max_loc.configure(&src, &min, &max, &min_loc, &max_loc, &min_count, &max_count);

    // Allocate tensors
    src.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(NEAccessor(src), 0);

    // Compute function
    min_max_loc.run();
}

void validate_configuration(const Tensor &src, TensorShape shape)
{
    BOOST_TEST(src.info()->is_resizable());

    // Create output storage
    int32_t            min;
    int32_t            max;
    Coordinates2DArray min_loc;
    Coordinates2DArray max_loc;
    uint32_t           min_count;
    uint32_t           max_count;

    // Create and configure function
    NEMinMaxLocation min_max_loc;
    min_max_loc.configure(&src, &min, &max, &min_loc, &max_loc, &min_count, &max_count);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 1).required_padding();
    validate(src.info()->padding(), padding);
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(MinMaxLocation)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (Small2DShapes() + Large2DShapes()) * boost::unit_test::data::make({ DataType::U8, DataType::S16 }),
                     shape, dt)
{
    // Create tensor
    Tensor src = create_tensor<Tensor>(shape, dt);
    src.info()->set_format(dt == DataType::U8 ? Format::U8 : Format::S16);

    validate_configuration(src, shape);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, Small2DShapes() * boost::unit_test::data::make({ DataType::U8, DataType::S16 }),
                     shape, dt)
{
    // Create output storage
    int32_t            min;
    int32_t            max;
    Coordinates2DArray min_loc(shape.total_size());
    Coordinates2DArray max_loc(shape.total_size());
    uint32_t           min_count;
    uint32_t           max_count;

    int32_t            ref_min;
    int32_t            ref_max;
    Coordinates2DArray ref_min_loc(shape.total_size());
    Coordinates2DArray ref_max_loc(shape.total_size());
    uint32_t           ref_min_count;
    uint32_t           ref_max_count;

    // Compute function
    compute_min_max_location(shape, dt, min, max, min_loc, max_loc, min_count, max_count);

    // Compute reference
    Reference::compute_reference_min_max_location(shape, dt, ref_min, ref_max, ref_min_loc, ref_max_loc, ref_min_count, ref_max_count);

    // Validate output
    validate_min_max_loc(min, ref_min, max, ref_max, min_loc, ref_min_loc, max_loc, ref_max_loc, min_count, ref_min_count, max_count, ref_max_count);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, Large2DShapes() * boost::unit_test::data::make({ DataType::U8, DataType::S16 }),
                     shape, dt)
{
    // Create output storage
    int32_t            min;
    int32_t            max;
    Coordinates2DArray min_loc(shape.total_size());
    Coordinates2DArray max_loc(shape.total_size());
    uint32_t           min_count;
    uint32_t           max_count;

    int32_t            ref_min;
    int32_t            ref_max;
    Coordinates2DArray ref_min_loc(shape.total_size());
    Coordinates2DArray ref_max_loc(shape.total_size());
    uint32_t           ref_min_count;
    uint32_t           ref_max_count;

    // Compute function
    compute_min_max_location(shape, dt, min, max, min_loc, max_loc, min_count, max_count);

    // Compute reference
    Reference::compute_reference_min_max_location(shape, dt, ref_min, ref_max, ref_min_loc, ref_max_loc, ref_min_count, ref_max_count);

    // Validate output
    validate_min_max_loc(min, ref_min, max, ref_max, min_loc, ref_min_loc, max_loc, ref_max_loc, min_count, ref_min_count, max_count, ref_max_count);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */