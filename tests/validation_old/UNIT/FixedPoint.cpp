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
#include "tests/validation_old/FixedPoint.h"

#include "Utils.h"
#include "support/ToolchainSupport.h"
#include "tests/validation_old/Validation.h"
#include "tests/validation_old/ValidationUserConfiguration.h"
#include "utils/TypePrinter.h"

#include "tests/validation_old/boost_wrapper.h"

#include <fstream>
#include <vector>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
std::string func_names[] =
{
    "add", "sub", "mul", "exp", "log", "inv_sqrt"
};
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(UNIT)
BOOST_AUTO_TEST_SUITE(FixedPoint)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(FixedPointQS8Inputs, boost::unit_test::data::make(func_names) * boost::unit_test::data::xrange(1, 7), func_name, frac_bits)
{
    const std::string base_file_name = user_config.path.get() + "/dumps/" + func_name + "_Q8." + support::cpp11::to_string(frac_bits);
    std::ifstream     inputs_file{ base_file_name + ".in", std::ios::binary | std::ios::in };

    BOOST_TEST_INFO(base_file_name + ".in");
    BOOST_TEST_REQUIRE(inputs_file.good()); //FIXME: When moving to new framework: throw a FileNotFound exception

    float float_val = 0.f;

    // Read first value
    inputs_file.read(reinterpret_cast<char *>(&float_val), sizeof(float_val));

    while(inputs_file.good())
    {
        // Convert to fixed point
        fixed_point_arithmetic::fixed_point<int8_t> in_val(float_val, frac_bits);

        // Check that the value didn't change
        BOOST_TEST(static_cast<float>(in_val) == float_val);

        // Read next value
        inputs_file.read(reinterpret_cast<char *>(&float_val), sizeof(float_val));
    }
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
//FIXME: Figure out how to handle expected failures properly
// The last input argument specifies the expected number of failures for a
// given combination of (function name, number of fractional bits) as defined
// by the first two arguments.
BOOST_DATA_TEST_CASE(FixedPointQS8Outputs, (boost::unit_test::data::make(func_names) * boost::unit_test::data::xrange(1, 7)) ^ (boost::unit_test::data::make({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 13, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 33, 96 })),
                     func_name, frac_bits, expected_failures)
{
    const std::string base_file_name = user_config.path.get() + "/dumps/" + func_name + "_Q8." + support::cpp11::to_string(frac_bits);
    std::ifstream     inputs_file{ base_file_name + ".in", std::ios::binary | std::ios::in };
    std::ifstream     reference_file{ base_file_name + ".out", std::ios::binary | std::ios::in };

    BOOST_TEST_INFO(base_file_name + ".in");
    BOOST_TEST_REQUIRE(inputs_file.good()); //FIXME: When moving to new framework: throw a FileNotFound exception
    BOOST_TEST_INFO(base_file_name + ".out");
    BOOST_TEST_REQUIRE(reference_file.good()); //FIXME: When moving to new framework: throw a FileNotFound exception

    const float step_size = std::pow(2.f, -frac_bits);

    float   float_val      = 0.f;
    float   ref_val        = 0.f;
    int64_t num_mismatches = 0;

    // Read first values
    inputs_file.read(reinterpret_cast<char *>(&float_val), sizeof(float_val));
    reference_file.read(reinterpret_cast<char *>(&ref_val), sizeof(ref_val));

    while(inputs_file.good() && reference_file.good())
    {
        fixed_point_arithmetic::fixed_point<int8_t> in_val(float_val, frac_bits);
        fixed_point_arithmetic::fixed_point<int8_t> out_val(0.f, frac_bits);

        float tolerance = 0.f;

        if(func_name == "add")
        {
            out_val = in_val + in_val;
        }
        else if(func_name == "sub")
        {
            out_val = in_val - in_val; //NOLINT
        }
        else if(func_name == "mul")
        {
            tolerance = 1.f * step_size;
            out_val   = in_val * in_val;
        }
        else if(func_name == "exp")
        {
            tolerance = 2.f * step_size;
            out_val   = fixed_point_arithmetic::exp(in_val);
        }
        else if(func_name == "log")
        {
            tolerance = 4.f * step_size;
            out_val   = fixed_point_arithmetic::log(in_val);
        }
        else if(func_name == "inv_sqrt")
        {
            tolerance = 5.f * step_size;
            out_val   = fixed_point_arithmetic::inv_sqrt(in_val);
        }

        if(std::abs(static_cast<float>(out_val) - ref_val) > tolerance)
        {
            BOOST_TEST_INFO("input = " << in_val);
            BOOST_TEST_INFO("output = " << out_val);
            BOOST_TEST_INFO("reference = " << ref_val);
            BOOST_TEST_INFO("tolerance = " << tolerance);
            BOOST_TEST_WARN((std::abs(static_cast<float>(out_val) - ref_val) <= tolerance));

            ++num_mismatches;
        }

        // Read next values
        inputs_file.read(reinterpret_cast<char *>(&float_val), sizeof(float_val));
        reference_file.read(reinterpret_cast<char *>(&ref_val), sizeof(ref_val));
    }

    BOOST_TEST(num_mismatches == expected_failures);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
