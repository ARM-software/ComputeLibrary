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
#include "tests/validation/FixedPoint.h"

#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto FuncNamesDataset = framework::dataset::make("FunctionNames", { FixedPointOp::ADD,
                                                                          FixedPointOp::SUB,
                                                                          FixedPointOp::MUL,
                                                                          FixedPointOp::EXP,
                                                                          FixedPointOp::LOG,
                                                                          FixedPointOp::INV_SQRT
                                                                        });

template <typename T>
void load_array_from_numpy(const std::string &file, std::vector<unsigned long> &shape, std::vector<T> &data) // NOLINT
{
    try
    {
        npy::LoadArrayFromNumpy(file, shape, data);
    }
    catch(const std::runtime_error &e)
    {
        throw framework::FileNotFound("Could not load npy file: " + file + " (" + e.what() + ")");
    }
}
} // namespace

TEST_SUITE(UNIT)
TEST_SUITE(FixedPoint)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(FixedPointQS8Inputs, framework::DatasetMode::ALL, combine(
               FuncNamesDataset,
               framework::dataset::make("FractionalBits", 1, 7)),
               func_name, frac_bits)
// clang-format on
// *INDENT-ON*
{
    std::vector<double>        data;
    std::vector<unsigned long> shape; //NOLINT

    std::string func_name_lower = to_string(func_name);
    std::transform(func_name_lower.begin(), func_name_lower.end(), func_name_lower.begin(), ::tolower);

    const std::string inputs_file = library->path()
    + "fixed_point/"
    + func_name_lower
    + "_Q8."
    + support::cpp11::to_string(frac_bits)
    + ".in.npy";

    load_array_from_numpy(inputs_file, shape, data);

    // Values stored as doubles so reinterpret as floats
    const auto *float_val    = reinterpret_cast<float *>(&data[0]);
    const size_t num_elements = data.size() * sizeof(double) / sizeof(float);

    for(unsigned int i = 0; i < num_elements; ++i)
    {
        // Convert to fixed point
        fixed_point_arithmetic::fixed_point<int8_t> in_val(float_val[i], frac_bits);

        // Check that the value didn't change
        ARM_COMPUTE_EXPECT(static_cast<float>(in_val) == float_val[i], framework::LogLevel::ERRORS);
    }
}

// The last input argument specifies the expected number of failures for a
// given combination of (function name, number of fractional bits) as defined
// by the first two arguments.

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(FixedPointQS8Outputs, framework::DatasetMode::ALL, zip(combine(
               FuncNamesDataset,
               framework::dataset::make("FractionalBits", 1, 7)),
               framework::dataset::make("ExpectedFailures", { 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0,
                                                              7, 8, 13, 2, 0, 0,
                                                              0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 5, 33, 96 })),
               func_name, frac_bits, expected_failures)
// clang-format on
// *INDENT-ON*
{
    std::vector<double>        in_data;
    std::vector<unsigned long> in_shape; //NOLINT

    std::vector<double>        out_data;
    std::vector<unsigned long> out_shape; //NOLINT

    std::string func_name_lower = to_string(func_name);
    std::transform(func_name_lower.begin(), func_name_lower.end(), func_name_lower.begin(), ::tolower);

    const std::string base_file_name = library->path()
    + "fixed_point/"
    + func_name_lower
    + "_Q8."
    + support::cpp11::to_string(frac_bits);

    const std::string inputs_file    = base_file_name + ".in.npy";
    const std::string reference_file = base_file_name + ".out.npy";

    load_array_from_numpy(inputs_file, in_shape, in_data);
    load_array_from_numpy(reference_file, out_shape, out_data);

    ARM_COMPUTE_EXPECT(in_shape.front() == out_shape.front(), framework::LogLevel::ERRORS);

    const float step_size      = std::pow(2.f, -frac_bits);
    int64_t     num_mismatches = 0;

    // Values stored as doubles so reinterpret as floats
    const auto *float_val = reinterpret_cast<float *>(&in_data[0]);
    const auto *ref_val   = reinterpret_cast<float *>(&out_data[0]);

    const size_t num_elements = in_data.size() * sizeof(double) / sizeof(float);

    for(unsigned int i = 0; i < num_elements; ++i)
    {
        fixed_point_arithmetic::fixed_point<int8_t> in_val(float_val[i], frac_bits);
        fixed_point_arithmetic::fixed_point<int8_t> out_val(0.f, frac_bits);

        float tolerance = 0.f;

        if(func_name == FixedPointOp::ADD)
        {
            out_val = in_val + in_val;
        }
        else if(func_name == FixedPointOp::SUB)
        {
            out_val = in_val - in_val; //NOLINT
        }
        else if(func_name == FixedPointOp::MUL)
        {
            tolerance = 1.f * step_size;
            out_val   = in_val * in_val;
        }
        else if(func_name == FixedPointOp::EXP)
        {
            tolerance = 2.f * step_size;
            out_val   = fixed_point_arithmetic::exp(in_val);
        }
        else if(func_name == FixedPointOp::LOG)
        {
            tolerance = 4.f * step_size;
            out_val   = fixed_point_arithmetic::log(in_val);
        }
        else if(func_name == FixedPointOp::INV_SQRT)
        {
            tolerance = 5.f * step_size;
            out_val   = fixed_point_arithmetic::inv_sqrt(in_val);
        }

        if(std::abs(static_cast<float>(out_val) - ref_val[i]) > tolerance)
        {
            ARM_COMPUTE_TEST_INFO("input = " << in_val);
            ARM_COMPUTE_TEST_INFO("output = " << out_val);
            ARM_COMPUTE_TEST_INFO("reference = " << ref_val[i]);
            ARM_COMPUTE_TEST_INFO("tolerance = " << tolerance);

            ARM_COMPUTE_TEST_INFO((std::abs(static_cast<float>(out_val) - ref_val[i]) <= tolerance));

            ++num_mismatches;
        }
    }

    ARM_COMPUTE_EXPECT(num_mismatches == expected_failures, framework::LogLevel::ERRORS);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
