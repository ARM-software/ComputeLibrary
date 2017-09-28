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
#ifndef __ARM_COMPUTE_TEST_VALIDATION_FIXEDPOINT_H__
#define __ARM_COMPUTE_TEST_VALIDATION_FIXEDPOINT_H__

#include "support/ToolchainSupport.h"
#include "tests/Utils.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace fixed_point_arithmetic
{
namespace detail
{
// Forward declare structs
struct functions;
template <typename T>
struct constant_expr;
}

/** Fixed point traits */
namespace traits
{
// Promote types
// *INDENT-OFF*
// clang-format off
template <typename T> struct promote { };
template <> struct promote<uint8_t> { using type = uint16_t; };
template <> struct promote<int8_t> { using type = int16_t; };
template <> struct promote<uint16_t> { using type = uint32_t; };
template <> struct promote<int16_t> { using type = int32_t; };
template <> struct promote<uint32_t> { using type = uint64_t; };
template <> struct promote<int32_t> { using type = int64_t; };
template <> struct promote<uint64_t> { using type = uint64_t; };
template <> struct promote<int64_t> { using type = int64_t; };
template <typename T>
using promote_t = typename promote<T>::type;
// clang-format on
// *INDENT-ON*
}

/** Strongly typed enum class representing the overflow policy */
enum class OverflowPolicy
{
    WRAP,    /**< Wrap policy */
    SATURATE /**< Saturate policy */
};
/** Strongly typed enum class representing the rounding policy */
enum class RoundingPolicy
{
    TO_ZERO,        /**< Round to zero policy */
    TO_NEAREST_EVEN /**< Round to nearest even policy */
};

/** Arbitrary fixed-point arithmetic class */
template <typename T>
class fixed_point
{
public:
    // Static Checks
    static_assert(std::is_integral<T>::value, "Type is not an integer");

    /** Constructor (from different fixed point type)
     *
     * @param[in] val Fixed point
     * @param[in] p   Fixed point precision
     */
    template <typename U>
    fixed_point(fixed_point<U> val, uint8_t p)
        : _value(0), _fixed_point_position(p)
    {
        assert(p > 0 && p < std::numeric_limits<T>::digits);
        T v = 0;

        if(std::numeric_limits<T>::digits < std::numeric_limits<U>::digits)
        {
            val.rescale(p);
            v = detail::constant_expr<T>::saturate_cast(val.raw());
        }
        else
        {
            auto v_cast = static_cast<fixed_point<T>>(val);
            v_cast.rescale(p);
            v = v_cast.raw();
        }
        _value = static_cast<T>(v);
    }
    /** Constructor (from integer)
     *
     * @param[in] val    Integer value to be represented as fixed point
     * @param[in] p      Fixed point precision
     * @param[in] is_raw If true val is a raw fixed point value else an integer
     */
    template <typename U, typename = typename std::enable_if<std::is_integral<U>::value>::type>
    fixed_point(U val, uint8_t p, bool is_raw = false)
        : _value(val << p), _fixed_point_position(p)
    {
        if(is_raw)
        {
            _value = val;
        }
    }
    /** Constructor (from float)
     *
     * @param[in] val Float value to be represented as fixed point
     * @param[in] p   Fixed point precision
     */
    fixed_point(float val, uint8_t p)
        : _value(detail::constant_expr<T>::to_fixed(val, p)), _fixed_point_position(p)
    {
        assert(p > 0 && p < std::numeric_limits<T>::digits);
    }
    /** Constructor (from float string)
     *
     * @param[in] str Float string to be represented as fixed point
     * @param[in] p   Fixed point precision
     */
    fixed_point(std::string str, uint8_t p)
        : _value(detail::constant_expr<T>::to_fixed(support::cpp11::stof(str), p)), _fixed_point_position(p)
    {
        assert(p > 0 && p < std::numeric_limits<T>::digits);
    }
    /** Default copy constructor */
    fixed_point &operator=(const fixed_point &) = default;
    /** Default move constructor */
    fixed_point &operator=(fixed_point &&) = default;
    /** Default copy assignment operator */
    fixed_point(const fixed_point &) = default;
    /** Default move assignment operator */
    fixed_point(fixed_point &&) = default;

    /** Float conversion operator
     *
     * @return Float representation of fixed point
     */
    operator float() const
    {
        return detail::constant_expr<T>::to_float(_value, _fixed_point_position);
    }
    /** Integer conversion operator
     *
     * @return Integer representation of fixed point
     */
    template <typename U, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    operator U() const
    {
        return detail::constant_expr<T>::to_int(_value, _fixed_point_position);
    }
    /** Convert to different fixed point of different type but same precision
     *
     * @note Down-conversion might fail.
     */
    template <typename U>
    operator fixed_point<U>()
    {
        U val = static_cast<U>(_value);
        if(std::numeric_limits<U>::digits < std::numeric_limits<T>::digits)
        {
            val = detail::constant_expr<U>::saturate_cast(_value);
        }
        return fixed_point<U>(val, _fixed_point_position, true);
    }

    /** Arithmetic += assignment operator
     *
     * @param[in] rhs Fixed point operand
     *
     * @return Reference to this fixed point
     */
    template <typename U>
    fixed_point<T> &operator+=(const fixed_point<U> &rhs)
    {
        fixed_point<T> val(rhs, _fixed_point_position);
        _value += val.raw();
        return *this;
    }
    /** Arithmetic -= assignment operator
     *
     * @param[in] rhs Fixed point operand
     *
     * @return Reference to this fixed point
     */
    template <typename U>
    fixed_point<T> &operator-=(const fixed_point<U> &rhs)
    {
        fixed_point<T> val(rhs, _fixed_point_position);
        _value -= val.raw();
        return *this;
    }

    /** Raw value accessor
     *
     * @return Raw fixed point value
     */
    T raw() const
    {
        return _value;
    }
    /** Precision accessor
     *
     * @return Precision of fixed point
     */
    uint8_t precision() const
    {
        return _fixed_point_position;
    }
    /** Rescale a fixed point to a new precision
     *
     * @param[in] p New fixed point precision
     */
    void rescale(uint8_t p)
    {
        assert(p > 0 && p < std::numeric_limits<T>::digits);

        using promoted_T = typename traits::promote<T>::type;
        promoted_T val   = _value;
        if(p > _fixed_point_position)
        {
            val <<= (p - _fixed_point_position);
        }
        else if(p < _fixed_point_position)
        {
            uint8_t pbar = _fixed_point_position - p;
            val += (pbar != 0) ? (1 << (pbar - 1)) : 0;
            val >>= pbar;
        }

        _value                = detail::constant_expr<T>::saturate_cast(val);
        _fixed_point_position = p;
    }

private:
    T       _value;                /**< Fixed point raw value */
    uint8_t _fixed_point_position; /**< Fixed point precision */
};

namespace detail
{
/** Count the number of leading zero bits in the given value.
 *
 * @param[in] value Input value.
 *
 * @return Number of leading zero bits.
 */
template <typename T>
constexpr int clz(T value)
{
    using unsigned_T = typename std::make_unsigned<T>::type;
    // __builtin_clz is available for int. Need to correct reported number to
    // match the original type.
    return __builtin_clz(value) - (32 - std::numeric_limits<unsigned_T>::digits);
}

template <typename T>
struct constant_expr
{
    /** Calculate representation of 1 in fixed point given a fixed point precision
     *
     * @param[in] p Fixed point precision
     *
     * @return Representation of value 1 in fixed point.
     */
    static constexpr T fixed_one(uint8_t p)
    {
        return (1 << p);
    }
    /** Calculate fixed point precision step given a fixed point precision
     *
     * @param[in] p Fixed point precision
     *
     * @return Fixed point precision step
     */
    static constexpr float fixed_step(uint8_t p)
    {
        return (1.0f / static_cast<float>(1 << p));
    }

    /** Convert a fixed point value to float given its precision.
     *
     * @param[in] val Fixed point value
     * @param[in] p   Fixed point precision
     *
     * @return Float representation of the fixed point number
     */
    static constexpr float to_float(T val, uint8_t p)
    {
        return static_cast<float>(val * fixed_step(p));
    }
    /** Convert a fixed point value to integer given its precision.
     *
     * @param[in] val Fixed point value
     * @param[in] p   Fixed point precision
     *
     * @return Integer of the fixed point number
     */
    static constexpr T to_int(T val, uint8_t p)
    {
        return val >> p;
    }
    /** Convert a single precision floating point value to a fixed point representation given its precision.
     *
     * @param[in] val Floating point value
     * @param[in] p   Fixed point precision
     *
     * @return The raw fixed point representation
     */
    static constexpr T to_fixed(float val, uint8_t p)
    {
        return static_cast<T>(saturate_cast<float>(val * fixed_one(p) + ((val >= 0) ? 0.5 : -0.5)));
    }
    /** Clamp value between two ranges
     *
     * @param[in] val Value to clamp
     * @param[in] min Minimum value to clamp to
     * @param[in] max Maximum value to clamp to
     *
     * @return clamped value
     */
    static constexpr T clamp(T val, T min, T max)
    {
        return std::min(std::max(val, min), max);
    }
    /** Saturate given number
     *
     * @param[in] val Value to saturate
     *
     * @return Saturated value
     */
    template <typename U>
    static constexpr T saturate_cast(U val)
    {
        return static_cast<T>(std::min<U>(std::max<U>(val, static_cast<U>(std::numeric_limits<T>::min())), static_cast<U>(std::numeric_limits<T>::max())));
    }
};
struct functions
{
    /** Output stream operator
     *
     * @param[in] s Output stream
     * @param[in] x Fixed point value
     *
     * @return Reference output to updated stream
     */
    template <typename T, typename U, typename traits>
    static std::basic_ostream<T, traits> &write(std::basic_ostream<T, traits> &s, fixed_point<U> &x)
    {
        return s << static_cast<float>(x);
    }
    /** Signbit of a fixed point number.
     *
     * @param[in] x Fixed point number
     *
     * @return True if negative else false.
     */
    template <typename T>
    static bool signbit(fixed_point<T> x)
    {
        return ((x.raw() >> std::numeric_limits<T>::digits) != 0);
    }
    /** Checks if two fixed point numbers are equal
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return True if fixed points are equal else false
     */
    template <typename T>
    static bool isequal(fixed_point<T> x, fixed_point<T> y)
    {
        uint8_t p = std::min(x.precision(), y.precision());
        x.rescale(p);
        y.rescale(p);
        return (x.raw() == y.raw());
    }
    /** Checks if two fixed point number are not equal
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return True if fixed points are not equal else false
     */
    template <typename T>
    static bool isnotequal(fixed_point<T> x, fixed_point<T> y)
    {
        return !isequal(x, y);
    }
    /** Checks if one fixed point is greater than the other
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return True if fixed point is greater than other
     */
    template <typename T>
    static bool isgreater(fixed_point<T> x, fixed_point<T> y)
    {
        uint8_t p = std::min(x.precision(), y.precision());
        x.rescale(p);
        y.rescale(p);
        return (x.raw() > y.raw());
    }
    /** Checks if one fixed point is greater or equal than the other
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return True if fixed point is greater or equal than other
     */
    template <typename T>
    static bool isgreaterequal(fixed_point<T> x, fixed_point<T> y)
    {
        uint8_t p = std::min(x.precision(), y.precision());
        x.rescale(p);
        y.rescale(p);
        return (x.raw() >= y.raw());
    }
    /** Checks if one fixed point is less than the other
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return True if fixed point is less than other
     */
    template <typename T>
    static bool isless(fixed_point<T> x, fixed_point<T> y)
    {
        uint8_t p = std::min(x.precision(), y.precision());
        x.rescale(p);
        y.rescale(p);
        return (x.raw() < y.raw());
    }
    /** Checks if one fixed point is less or equal than the other
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return True if fixed point is less or equal than other
     */
    template <typename T>
    static bool islessequal(fixed_point<T> x, fixed_point<T> y)
    {
        uint8_t p = std::min(x.precision(), y.precision());
        x.rescale(p);
        y.rescale(p);
        return (x.raw() <= y.raw());
    }
    /** Checks if one fixed point is less or greater than the other
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return True if fixed point is less or greater than other
     */
    template <typename T>
    static bool islessgreater(fixed_point<T> x, fixed_point<T> y)
    {
        return isnotequal(x, y);
    }
    /** Clamp fixed point to specific range.
     *
     * @param[in] x   Fixed point operand
     * @param[in] min Minimum value to clamp to
     * @param[in] max Maximum value to clamp to
     *
     * @return Clamped result
     */
    template <typename T>
    static fixed_point<T> clamp(fixed_point<T> x, T min, T max)
    {
        return fixed_point<T>(constant_expr<T>::clamp(x.raw(), min, max), x.precision(), true);
    }
    /** Negate number
     *
     * @param[in] x Fixed point operand
     *
     * @return Negated fixed point result
     */
    template <OverflowPolicy OP = OverflowPolicy::SATURATE, typename T>
    static fixed_point<T> negate(fixed_point<T> x)
    {
        using promoted_T = typename traits::promote<T>::type;
        promoted_T val   = -x.raw();
        if(OP == OverflowPolicy::SATURATE)
        {
            val = constant_expr<T>::saturate_cast(val);
        }
        return fixed_point<T>(static_cast<T>(val), x.precision(), true);
    }
    /** Perform addition among two fixed point numbers
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return Result fixed point with precision equal to minimum precision of both operands
     */
    template <OverflowPolicy OP = OverflowPolicy::SATURATE, typename T>
    static fixed_point<T> add(fixed_point<T> x, fixed_point<T> y)
    {
        uint8_t p = std::min(x.precision(), y.precision());
        x.rescale(p);
        y.rescale(p);
        if(OP == OverflowPolicy::SATURATE)
        {
            using type = typename traits::promote<T>::type;
            type val   = static_cast<type>(x.raw()) + static_cast<type>(y.raw());
            val        = constant_expr<T>::saturate_cast(val);
            return fixed_point<T>(static_cast<T>(val), p, true);
        }
        else
        {
            return fixed_point<T>(x.raw() + y.raw(), p, true);
        }
    }
    /** Perform subtraction among two fixed point numbers
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return Result fixed point with precision equal to minimum precision of both operands
     */
    template <OverflowPolicy OP = OverflowPolicy::SATURATE, typename T>
    static fixed_point<T> sub(fixed_point<T> x, fixed_point<T> y)
    {
        uint8_t p = std::min(x.precision(), y.precision());
        x.rescale(p);
        y.rescale(p);
        if(OP == OverflowPolicy::SATURATE)
        {
            using type = typename traits::promote<T>::type;
            type val   = static_cast<type>(x.raw()) - static_cast<type>(y.raw());
            val        = constant_expr<T>::saturate_cast(val);
            return fixed_point<T>(static_cast<T>(val), p, true);
        }
        else
        {
            return fixed_point<T>(x.raw() - y.raw(), p, true);
        }
    }
    /** Perform multiplication among two fixed point numbers
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return Result fixed point with precision equal to minimum precision of both operands
     */
    template <OverflowPolicy OP = OverflowPolicy::SATURATE, typename T>
    static fixed_point<T> mul(fixed_point<T> x, fixed_point<T> y)
    {
        using promoted_T        = typename traits::promote<T>::type;
        uint8_t    p_min        = std::min(x.precision(), y.precision());
        uint8_t    p_max        = std::max(x.precision(), y.precision());
        promoted_T round_factor = (1 << (p_max - 1));
        promoted_T val          = ((static_cast<promoted_T>(x.raw()) * static_cast<promoted_T>(y.raw())) + round_factor) >> p_max;
        if(OP == OverflowPolicy::SATURATE)
        {
            val = constant_expr<T>::saturate_cast(val);
        }
        return fixed_point<T>(static_cast<T>(val), p_min, true);
    }
    /** Perform division among two fixed point numbers
     *
     * @param[in] x First fixed point operand
     * @param[in] y Second fixed point operand
     *
     * @return Result fixed point with precision equal to minimum precision of both operands
     */
    template <OverflowPolicy OP = OverflowPolicy::SATURATE, typename T>
    static fixed_point<T> div(fixed_point<T> x, fixed_point<T> y)
    {
        using promoted_T = typename traits::promote<T>::type;
        uint8_t    p     = std::min(x.precision(), y.precision());
        promoted_T denom = static_cast<promoted_T>(y.raw());
        if(denom != 0)
        {
            promoted_T val = (static_cast<promoted_T>(x.raw()) << std::max(x.precision(), y.precision())) / denom;
            if(OP == OverflowPolicy::SATURATE)
            {
                val = constant_expr<T>::saturate_cast(val);
            }
            return fixed_point<T>(static_cast<T>(val), p, true);
        }
        else
        {
            T val = (x.raw() < 0) ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
            return fixed_point<T>(val, p, true);
        }
    }
    /** Shift left
     *
     * @param[in] x     Fixed point operand
     * @param[in] shift Shift value
     *
     * @return Shifted value
     */
    template <OverflowPolicy OP = OverflowPolicy::SATURATE, typename T>
    static fixed_point<T> shift_left(fixed_point<T> x, size_t shift)
    {
        using promoted_T = typename traits::promote<T>::type;
        promoted_T val   = static_cast<promoted_T>(x.raw()) << shift;
        if(OP == OverflowPolicy::SATURATE)
        {
            val = constant_expr<T>::saturate_cast(val);
        }
        return fixed_point<T>(static_cast<T>(val), x.precision(), true);
    }
    /** Shift right
     *
     * @param[in] x     Fixed point operand
     * @param[in] shift Shift value
     *
     * @return Shifted value
     */
    template <typename T>
    static fixed_point<T> shift_right(fixed_point<T> x, size_t shift)
    {
        return fixed_point<T>(x.raw() >> shift, x.precision(), true);
    }
    /** Calculate absolute value
     *
     * @param[in] x Fixed point operand
     *
     * @return Absolute value of operand
     */
    template <typename T>
    static fixed_point<T> abs(fixed_point<T> x)
    {
        using promoted_T = typename traits::promote<T>::type;
        T val            = (x.raw() < 0) ? constant_expr<T>::saturate_cast(-static_cast<promoted_T>(x.raw())) : x.raw();
        return fixed_point<T>(val, x.precision(), true);
    }
    /** Calculate the logarithm of a fixed point number
     *
     * @param[in] x Fixed point operand
     *
     * @return Logarithm value of operand
     */
    template <typename T>
    static fixed_point<T> log(fixed_point<T> x)
    {
        uint8_t p         = x.precision();
        auto    const_one = fixed_point<T>(static_cast<T>(1), p);

        // Logarithm of 1 is zero and logarithm of negative values is not defined in R, so return 0.
        // Also, log(x) == -log(1/x) for 0 < x < 1.
        if(isequal(x, const_one) || islessequal(x, fixed_point<T>(static_cast<T>(0), p)))
        {
            return fixed_point<T>(static_cast<T>(0), p, true);
        }
        else if(isless(x, const_one))
        {
            return mul(log(div(const_one, x)), fixed_point<T>(-1, p));
        }

        // Remove even powers of 2
        T shift_val = 31 - __builtin_clz(x.raw() >> p);
        x           = shift_right(x, shift_val);
        x           = sub(x, const_one);

        // Constants
        auto ln2 = fixed_point<T>(0.6931471, p);
        auto A   = fixed_point<T>(1.4384189, p);
        auto B   = fixed_point<T>(-0.67719, p);
        auto C   = fixed_point<T>(0.3218538, p);
        auto D   = fixed_point<T>(-0.0832229, p);

        // Polynomial expansion
        auto sum = add(mul(x, D), C);
        sum      = add(mul(x, sum), B);
        sum      = add(mul(x, sum), A);
        sum      = mul(x, sum);

        return mul(add(sum, fixed_point<T>(static_cast<T>(shift_val), p)), ln2);
    }
    /** Calculate the exponential of a fixed point number.
     *
     * exp(x) = exp(floor(x)) * exp(x - floor(x))
     *        = pow(2, floor(x) / ln(2)) * exp(x - floor(x))
     *        = exp(x - floor(x)) << (floor(x) / ln(2))
     *
     * @param[in] x Fixed point operand
     *
     * @return Exponential value of operand
     */
    template <typename T>
    static fixed_point<T> exp(fixed_point<T> x)
    {
        uint8_t p = x.precision();
        // Constants
        auto const_one = fixed_point<T>(1, p);
        auto ln2       = fixed_point<T>(0.6931471, p);
        auto inv_ln2   = fixed_point<T>(1.442695, p);
        auto A         = fixed_point<T>(0.9978546, p);
        auto B         = fixed_point<T>(0.4994721, p);
        auto C         = fixed_point<T>(0.1763723, p);
        auto D         = fixed_point<T>(0.0435108, p);

        T scaled_int_part = detail::constant_expr<T>::to_int(mul(x, inv_ln2).raw(), p);

        // Polynomial expansion
        auto frac_part = sub(x, mul(ln2, fixed_point<T>(scaled_int_part, p)));
        auto taylor    = add(mul(frac_part, D), C);
        taylor         = add(mul(frac_part, taylor), B);
        taylor         = add(mul(frac_part, taylor), A);
        taylor         = mul(frac_part, taylor);
        taylor         = add(taylor, const_one);

        // Saturate value
        if(static_cast<T>(clz(taylor.raw())) <= scaled_int_part)
        {
            return fixed_point<T>(std::numeric_limits<T>::max(), p, true);
        }

        return (scaled_int_part < 0) ? shift_right(taylor, -scaled_int_part) : shift_left(taylor, scaled_int_part);
    }
    /** Calculate the inverse square root of a fixed point number
     *
     * @param[in] x Fixed point operand
     *
     * @return Inverse square root value of operand
     */
    template <typename T>
    static fixed_point<T> inv_sqrt(fixed_point<T> x)
    {
        const uint8_t p     = x.precision();
        int8_t        shift = std::numeric_limits<T>::digits - (p + detail::clz(x.raw()));

        shift += std::numeric_limits<T>::is_signed ? 1 : 0;

        // Use volatile to restrict compiler optimizations on shift as compiler reports maybe-uninitialized error on Android
        volatile int8_t *shift_ptr = &shift;

        auto           const_three = fixed_point<T>(3, p);
        auto           a           = (*shift_ptr < 0) ? shift_left(x, -(shift)) : shift_right(x, shift);
        fixed_point<T> x2          = a;

        // We need three iterations to find the result for QS8 and five for QS16
        constexpr int num_iterations = std::is_same<T, int8_t>::value ? 3 : 5;
        for(int i = 0; i < num_iterations; ++i)
        {
            fixed_point<T> three_minus_dx = sub(const_three, mul(a, mul(x2, x2)));
            x2                            = shift_right(mul(x2, three_minus_dx), 1);
        }

        return (shift < 0) ? shift_left(x2, (-shift) >> 1) : shift_right(x2, shift >> 1);
    }
    /** Calculate the hyperbolic tangent of a fixed point number
     *
     * @param[in] x Fixed point operand
     *
     * @return Hyperbolic tangent of the operand
     */
    template <typename T>
    static fixed_point<T> tanh(fixed_point<T> x)
    {
        uint8_t p = x.precision();
        // Constants
        auto const_one = fixed_point<T>(1, p);
        auto const_two = fixed_point<T>(2, p);

        auto exp2x = exp(const_two * x);
        auto num   = exp2x - const_one;
        auto den   = exp2x + const_one;
        auto tanh  = num / den;

        return tanh;
    }
    /** Calculate the a-th power of a fixed point number.
     *
     *  The power is computed as x^a = e^(log(x) * a)
     *
     * @param[in] x Fixed point operand
     * @param[in] a Fixed point exponent
     *
     * @return a-th power of the operand
     */
    template <typename T>
    static fixed_point<T> pow(fixed_point<T> x, fixed_point<T> a)
    {
        return exp(log(x) * a);
    }
};

template <typename T>
bool operator==(const fixed_point<T> &lhs, const fixed_point<T> &rhs)
{
    return functions::isequal(lhs, rhs);
}
template <typename T>
bool operator!=(const fixed_point<T> &lhs, const fixed_point<T> &rhs)
{
    return !operator==(lhs, rhs);
}
template <typename T>
bool operator<(const fixed_point<T> &lhs, const fixed_point<T> &rhs)
{
    return functions::isless(lhs, rhs);
}
template <typename T>
bool operator>(const fixed_point<T> &lhs, const fixed_point<T> &rhs)
{
    return operator<(rhs, lhs);
}
template <typename T>
bool operator<=(const fixed_point<T> &lhs, const fixed_point<T> &rhs)
{
    return !operator>(lhs, rhs);
}
template <typename T>
bool operator>=(const fixed_point<T> &lhs, const fixed_point<T> &rhs)
{
    return !operator<(lhs, rhs);
}
template <typename T>
fixed_point<T> operator+(const fixed_point<T> &lhs, const fixed_point<T> &rhs)
{
    return functions::add(lhs, rhs);
}
template <typename T>
fixed_point<T> operator-(const fixed_point<T> &lhs, const fixed_point<T> &rhs)
{
    return functions::sub(lhs, rhs);
}
template <typename T>
fixed_point<T> operator-(const fixed_point<T> &rhs)
{
    return functions::negate(rhs);
}
template <typename T>
fixed_point<T> operator*(fixed_point<T> x, fixed_point<T> y)
{
    return functions::mul(x, y);
}
template <typename T>
fixed_point<T> operator/(fixed_point<T> x, fixed_point<T> y)
{
    return functions::div(x, y);
}
template <typename T>
fixed_point<T> operator>>(fixed_point<T> x, size_t shift)
{
    return functions::shift_right(x, shift);
}
template <typename T>
fixed_point<T> operator<<(fixed_point<T> x, size_t shift)
{
    return functions::shift_left(x, shift);
}
template <typename T, typename U, typename traits>
std::basic_ostream<T, traits> &operator<<(std::basic_ostream<T, traits> &s, fixed_point<U> x)
{
    return functions::write(s, x);
}
template <typename T>
inline fixed_point<T> min(fixed_point<T> x, fixed_point<T> y)
{
    return x > y ? y : x;
}
template <typename T>
inline fixed_point<T> max(fixed_point<T> x, fixed_point<T> y)
{
    return x > y ? x : y;
}
template <OverflowPolicy OP = OverflowPolicy::SATURATE, typename T>
inline fixed_point<T> add(fixed_point<T> x, fixed_point<T> y)
{
    return functions::add<OP>(x, y);
}
template <OverflowPolicy OP = OverflowPolicy::SATURATE, typename T>
inline fixed_point<T> sub(fixed_point<T> x, fixed_point<T> y)
{
    return functions::sub<OP>(x, y);
}
template <OverflowPolicy OP = OverflowPolicy::SATURATE, typename T>
inline fixed_point<T> mul(fixed_point<T> x, fixed_point<T> y)
{
    return functions::mul<OP>(x, y);
}
template <typename T>
inline fixed_point<T> div(fixed_point<T> x, fixed_point<T> y)
{
    return functions::div(x, y);
}
template <typename T>
inline fixed_point<T> abs(fixed_point<T> x)
{
    return functions::abs(x);
}
template <typename T>
inline fixed_point<T> clamp(fixed_point<T> x, T min, T max)
{
    return functions::clamp(x, min, max);
}
template <typename T>
inline fixed_point<T> exp(fixed_point<T> x)
{
    return functions::exp(x);
}
template <typename T>
inline fixed_point<T> log(fixed_point<T> x)
{
    return functions::log(x);
}
template <typename T>
inline fixed_point<T> inv_sqrt(fixed_point<T> x)
{
    return functions::inv_sqrt(x);
}
template <typename T>
inline fixed_point<T> tanh(fixed_point<T> x)
{
    return functions::tanh(x);
}
template <typename T>
inline fixed_point<T> pow(fixed_point<T> x, fixed_point<T> a)
{
    return functions::pow(x, a);
}
} // namespace detail

// Expose operators
using detail::operator==;
using detail::operator!=;
using detail::operator<;
using detail::operator>;
using detail::operator<=;
using detail::operator>=;
using detail::operator+;
using detail::operator-;
using detail::operator*;
using detail::operator/;
using detail::operator>>;
using detail::operator<<;

// Expose additional functions
using detail::min;
using detail::max;
using detail::add;
using detail::sub;
using detail::mul;
using detail::div;
using detail::abs;
using detail::clamp;
using detail::exp;
using detail::log;
using detail::inv_sqrt;
using detail::tanh;
using detail::pow;
} // namespace fixed_point_arithmetic
} // namespace test
} // namespace arm_compute
#endif /*__ARM_COMPUTE_TEST_VALIDATION_FIXEDPOINT_H__ */
