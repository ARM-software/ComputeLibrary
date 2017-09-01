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
#include "TypePrinter.h"
#include "tests/validation_old/Validation.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "tests/validation_old/boost_wrapper.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(UNIT)
BOOST_AUTO_TEST_SUITE(TensorInfoValidation)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(AutoPadding,
                     boost::unit_test::data::make({ TensorShape{},
                                                    TensorShape{ 10U },
                                                    TensorShape{ 10U, 10U },
                                                    TensorShape{ 10U, 10U, 10U },
                                                    TensorShape{ 10U, 10U, 10U, 10U },
                                                    TensorShape{ 10U, 10U, 10U, 10U, 10U },
                                                    TensorShape{ 10U, 10U, 10U, 10U, 10U, 10U }
                                                  })
                     ^ boost::unit_test::data::make({ PaddingSize{ 0, 0, 0, 0 },
                                                      PaddingSize{ 0, 36, 0, 4 },
                                                      PaddingSize{ 4, 36, 4, 4 },
                                                      PaddingSize{ 4, 36, 4, 4 },
                                                      PaddingSize{ 4, 36, 4, 4 },
                                                      PaddingSize{ 4, 36, 4, 4 },
                                                      PaddingSize{ 4, 36, 4, 4 }
                                                    })
                     ^ boost::unit_test::data::make({ Strides{},
                                                      Strides{ 1U },
                                                      Strides{ 1U, 50U },
                                                      Strides{ 1U, 50U, 900U },
                                                      Strides{ 1U, 50U, 900U, 9000U },
                                                      Strides{ 1U, 50U, 900U, 9000U, 90000U },
                                                      Strides{ 1U, 50U, 900U, 9000U, 90000U, 900000U }
                                                    })
                     ^ boost::unit_test::data::make(
{
    0,
    4,
    204,
    204,
    204,
    204,
    204,
}),
shape, auto_padding, strides, offset)
{
    TensorInfo info{ shape, Format::U8 };

    BOOST_TEST(!info.has_padding());

    info.auto_padding();

    validate(info.padding(), auto_padding);
    BOOST_TEST(compare_dimensions(info.strides_in_bytes(), strides));
    BOOST_TEST(info.offset_first_element_in_bytes() == offset);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
