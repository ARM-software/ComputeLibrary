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
#include "tests/validation_old/Validation.h"
#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"

#include "tests/validation_old/boost_wrapper.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(UNIT)
BOOST_AUTO_TEST_SUITE(TensorShapeValidation)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Construction,
                     boost::unit_test::data::make({ TensorShape{},
                                                    TensorShape{ 1U },
                                                    TensorShape{ 2U },
                                                    TensorShape{ 2U, 3U },
                                                    TensorShape{ 2U, 3U, 5U },
                                                    TensorShape{ 2U, 3U, 5U, 7U },
                                                    TensorShape{ 2U, 3U, 5U, 7U, 11U },
                                                    TensorShape{ 2U, 3U, 5U, 7U, 11U, 13U }
                                                  })
                     ^ boost::unit_test::data::make({ 0, 0, 1, 2, 3, 4, 5, 6 }) ^ boost::unit_test::data::make({ 0, 1, 2, 6, 30, 210, 2310, 30030 }),
                     shape, num_dimensions, total_size)
{
    BOOST_TEST(shape.num_dimensions() == num_dimensions);
    BOOST_TEST(shape.total_size() == total_size);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(SetEmpty, boost::unit_test::data::make({ 0, 1, 2, 3, 4, 5 }), dimension)
{
    TensorShape shape;

    shape.set(dimension, 10);

    BOOST_TEST(shape.num_dimensions() == dimension + 1);
    BOOST_TEST(shape.total_size() == 10);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
