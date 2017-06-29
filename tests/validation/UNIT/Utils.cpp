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
#include "Utils.h"

#include "TypePrinter.h"
#include "validation/Validation.h"

#include "boost_wrapper.h"

#include <stdexcept>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(UNIT)
BOOST_AUTO_TEST_SUITE(Utils)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RoundHalfUp, boost::unit_test::data::make({ 1.f, 1.2f, 1.5f, 2.5f, 2.9f, -3.f, -3.5f, -3.8f, -4.3f, -4.5f }) ^ boost::unit_test::data::make({ 1.f, 1.f, 2.f, 3.f, 3.f, -3.f, -3.f, -4.f, -4.f, -4.f }),
                     value, result)
{
    BOOST_TEST(round_half_up(value) == result);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RoundHalfEven, boost::unit_test::data::make({ 1.f, 1.2f, 1.5f, 2.5f, 2.9f, -3.f, -3.5f, -3.8f, -4.3f, -4.5f }) ^ boost::unit_test::data::make({ 1.f, 1.f, 2.f, 2.f, 3.f, -3.f, -4.f, -4.f, -4.f, -4.f }),
                     value, result)
{
    BOOST_TEST(round_half_even(value) == result);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Index2Coord, boost::unit_test::data::make({ TensorShape{ 1U }, TensorShape{ 2U }, TensorShape{ 2U, 3U } }) ^ boost::unit_test::data::make({ 0, 1, 2 }) ^
                     boost::unit_test::data::make({ Coordinates{ 0 }, Coordinates{ 1 }, Coordinates{ 0, 1 } }), shape, index, ref_coordinate)
{
    Coordinates coordinate = index2coord(shape, index);

    BOOST_TEST(compare_dimensions(coordinate, ref_coordinate));
}

//FIXME: Negative tests only work in debug mode
#if 0
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Index2CoordFail, boost::unit_test::data::make({ TensorShape{}, TensorShape{ 2U }, TensorShape{ 2U } }) ^ boost::unit_test::data::make({ 0, -1, 2 }), shape, index)
{
    BOOST_CHECK_THROW(index2coord(shape, index), std::runtime_error);
}
#endif

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Coord2Index, boost::unit_test::data::make({ TensorShape{ 1U }, TensorShape{ 2U }, TensorShape{ 2U, 3U } }) ^ boost::unit_test::data::make({ Coordinates{ 0 }, Coordinates{ 1 }, Coordinates{ 0, 1 } })
                     ^ boost::unit_test::data::make({ 0, 1, 2 }),
                     shape, coordinate, ref_index)
{
    int index = coord2index(shape, coordinate);

    BOOST_TEST(index == ref_index);
}

//FIXME: Negative tests only work in debug mode
#if 0
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Coord2IndexFail, boost::unit_test::data::make({ TensorShape{}, TensorShape{ 2U } }) ^ boost::unit_test::data::make({ Coordinates{ 0 }, Coordinates{} }), shape, coordinate)
{
    BOOST_CHECK_THROW(coord2index(shape, coordinate), std::runtime_error);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
