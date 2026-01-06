/*
 * Copyright (c) 2024-2026 Arm Limited.
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
#include "src/core/helpers/LUTManager.h"
#include "support/Half.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
#ifdef ARM_COMPUTE_ENABLE_FP16
// Take fp16 value and output as uint16_t without changing bits.
inline uint16_t read_as_bf16(const float16_t tmp)
{
    uint16_t out = 0;
    memcpy(&out, &tmp, sizeof(tmp));
    return out;
}
#endif // ARM_COMPUTE_ENABLE_FP16

// Check if difference in values is within tolerance range
template <typename U>
bool equal_values_relative(const U target, const U reference, const float tolerance)
{
    if (are_equal_infs(target, reference))
    {
        return true;
    }
    else if (target == reference)
    {
        return true;
    }
    else if (half_float::detail::builtin_isnan(target) &&
             half_float::detail::builtin_isnan(reference)) // determine if nan values using existing function
    {
        return true;
    }

    const U epsilon = (std::is_same<half, typename std::remove_cv<U>::type>::value || (reference == 0))
                          ? static_cast<U>(0.01)
                          : static_cast<U>(1e-05);
    if (std::abs(static_cast<double>(reference) - static_cast<double>(target)) <= epsilon)
    {
        return true;
    }
    else
    {
        if (static_cast<double>(reference) == 0.0f)
        {
            return false; // We have checked whether _reference and _target is close. If _reference is 0 but not close to _target, it should return false
        }
        const double relative_change =
            std::abs((static_cast<double>(target) - static_cast<double>(reference)) / reference);
        return relative_change <= static_cast<U>(tolerance);
    }
}
} // namespace

TEST_SUITE(CPP)
TEST_SUITE(LUTManager)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(BF16)
TEST_CASE(LUTValueTest, framework::DatasetMode::ALL)
{
    // Define values for test
    constexpr float beta           = -1.0f;
    constexpr float rel_tolerance  = 0.01f;
    constexpr int   num_elements   = 65536;
    unsigned int    num_mismatches = 0;

    // Create lutinfo, use to get lut
    LUTInfo    info = {LUTType::Exponential, beta, DataType::BFLOAT16, UniformQuantizationInfo()};
    LUTManager lman = LUTManager::get_instance();

    if (cpu_supports_dtypes({DataType::BFLOAT16}))
    {
        // Retrieve lut, Assert lut exists and is retrieved successfully.
        std::shared_ptr<LookupTable65536> lut = lman.get_lut_table<LookupTable65536>(info);
        ARM_COMPUTE_EXPECT(lut != nullptr, framework::LogLevel::ALL);

        // Check each value in lut
        for (int i = 0; i < num_elements; i++)
        {
            // Calculate reference in fp32. Convert lut value to fp32.
            const float    fref        = std::exp(bf16_to_float(i) * beta);
            const uint16_t target_bf16 = read_as_bf16((*lut)[i]);
            const float    target      = bf16_to_float(target_bf16);

            // Compare and increment mismatch count if needed.
            if (!equal_values_relative(target, fref, rel_tolerance))
            {
                ARM_COMPUTE_TEST_INFO("id = " << i);
                ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << framework::make_printable(target));
                ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << framework::make_printable(fref));
                ARM_COMPUTE_TEST_INFO("relative tolerance = " << std::setprecision(5)
                                                              << framework::make_printable(rel_tolerance));
                framework::ARM_COMPUTE_PRINT_INFO();
                ++num_mismatches;
            }
        }

        if (num_mismatches != 0)
        {
            const float percent_mismatches = static_cast<float>(num_mismatches) / num_elements * 100.f;
            ARM_COMPUTE_TEST_INFO(num_mismatches << " values (" << std::fixed << std::setprecision(2)
                                                 << percent_mismatches << "%) mismatched ");
        }

        // Check if passed tests
        ARM_COMPUTE_EXPECT(num_mismatches == 0, framework::LogLevel::ERRORS);
    }
}

TEST_CASE(CheckLutReuse, framework::DatasetMode::ALL)
{
    if (cpu_supports_dtypes({DataType::BFLOAT16}))
    {
        LUTInfo    info   = {LUTType::Exponential, -1.0f, DataType::BFLOAT16, UniformQuantizationInfo()};
        LUTManager lman   = LUTManager::get_instance();
        auto       first  = lman.get_lut_table<LookupTable65536>(info);
        auto       second = lman.get_lut_table<LookupTable65536>(info);
        ARM_COMPUTE_EXPECT(first == second, framework::LogLevel::ERRORS);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support BFLOAT16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

TEST_SUITE_END() // BF16
#endif           // ARM_COMPUTE_ENABLE_FP16

TEST_SUITE_END() // LUTManager
TEST_SUITE_END() // CPP

} // namespace validation
} // namespace test
} // namespace arm_compute
