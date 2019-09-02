/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_UTILS_HELPERS_FLOAT_OPS_H__
#define __ARM_COMPUTE_UTILS_HELPERS_FLOAT_OPS_H__

namespace arm_compute
{
namespace helpers
{
namespace float_ops
{
union RawFloat
{
    /** Constructor
     *
     * @param[in] val Floating-point value
     */
    explicit RawFloat(float val)
        : f32(val)
    {
    }
    /** Extract sign of floating point number
     *
     * @return Sign of floating point number
     */
    int32_t sign() const
    {
        return i32 >> 31;
    }
    /** Extract exponent of floating point number
     *
     * @return Exponent of floating point number
     */
    int32_t exponent() const
    {
        return (i32 >> 23) & 0xFF;
    }
    /** Extract mantissa of floating point number
     *
     * @return Mantissa of floating point number
     */
    int32_t mantissa() const
    {
        return i32 & 0x007FFFFF;
    }

    int32_t i32;
    float   f32;
};

/** Checks if two floating point numbers are equal given an allowed number of ULPs
 *
 * @param[in] a                First number to compare
 * @param[in] b                Second number to compare
 * @param[in] max_allowed_ulps (Optional) Number of allowed ULPs
 *
 * @return True if number is close else false
 */
inline bool is_equal_ulps(float a, float b, int max_allowed_ulps = 0)
{
    RawFloat ra(a);
    RawFloat rb(b);

    // Check ULP distance
    const int ulps = std::abs(ra.i32 - rb.i32);
    return ulps <= max_allowed_ulps;
}

/** Checks if the input floating point number is 1.0f checking if the difference is within a range defined with epsilon
 *
 * @param[in] a       Input floating point number
 * @param[in] epsilon (Optional) Epsilon used to define the error bounds
 *
 * @return True if number is close to 1.0f
 */
inline bool is_one(float a, float epsilon = 0.00001f)
{
    return std::abs(1.0f - a) <= epsilon;
}

/** Checks if the input floating point number is 0.0f checking if the difference is within a range defined with epsilon
 *
 * @param[in] a       Input floating point number
 * @param[in] epsilon (Optional) Epsilon used to define the error bounds
 *
 * @return True if number is close to 0.0f
 */
inline bool is_zero(float a, float epsilon = 0.00001f)
{
    return std::abs(0.0f - a) <= epsilon;
}
} // namespace float_ops
} // namespace helpers
} // namespace arm_compute
#endif /* __ARM_COMPUTE_UTILS_HELPERS_FLOAT_OPS_H__ */
