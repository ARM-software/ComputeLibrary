/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "PixelWiseMultiplication.h"

#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <class T>
struct is_floating_point
    : std::integral_constant < bool,
      std::is_same<float, typename std::remove_cv<T>::type>::value || std::is_same<half_float::half, typename std::remove_cv<T>::type>::value
      || std::is_same<double, typename std::remove_cv<T>::type>::value || std::is_same<long double, typename std::remove_cv<T>::type>::value >
{
};

namespace
{
constexpr float scale1_constant = 1.f;

/** Compute the result of `src1 * src2 * scale`. The result type always matches the type of @p src2.
 *
 * @param[in] src1            An input value. Data types supported: U8/S16/F16/F32.
 * @param[in] src2            An input value. Data types supported: same as @p src1.
 * @param[in] scale           Scale to apply after multiplication.
 *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
 * @param[in] convert_policy  Overflow policy. Supported overflow policies: Wrap, Saturate
 * @param[in] rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest even.
 */
template <typename T1, typename T2, typename T3>
T3 mul(const T1 src1, const T2 src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    using intermediate_type = typename common_promoted_signed_type<T1, T2, T3>::intermediate_type;

    const double val = static_cast<intermediate_type>(src1) * static_cast<intermediate_type>(src2) * static_cast<double>(scale);

    if(is_floating_point<T3>::value)
    {
        const auto result = static_cast<T3>(val);

        return result;
    }
    else
    {
        double rounded_val = 0;
        switch(rounding_policy)
        {
            case(RoundingPolicy::TO_ZERO):
                rounded_val = support::cpp11::trunc(val);
                break;
            case(RoundingPolicy::TO_NEAREST_UP):
                rounded_val = round_half_up(val);
                break;
            case(RoundingPolicy::TO_NEAREST_EVEN):
                rounded_val = round_half_even(val);
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported rounding policy");
        }

        const auto result = static_cast<T3>((convert_policy == ConvertPolicy::SATURATE) ? saturate_cast<T3>(rounded_val) : rounded_val);

        return result;
    }
}

template <>
int32_t mul(const int32_t src1, const int32_t src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    const int64_t intermediate_val = static_cast<int64_t>(src1) * static_cast<int64_t>(src2);

    if(std::abs(scale - scale1_constant) < 0.00001f)
    {
        // Use bit-accurate integer arithmetic for scale == 1
        // Apply conversion
        if(convert_policy == ConvertPolicy::SATURATE)
        {
            return saturate_cast<int32_t>(intermediate_val);
        }
        else
        {
            // Correct wrapping behaviour for int32_t
            const auto i32_hi              = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
            const auto i32_lo              = static_cast<int64_t>(std::numeric_limits<int32_t>::lowest());
            const auto i32_wi              = static_cast<int64_t>(1) << 32;
            int64_t    wrapped_rounded_val = intermediate_val - i32_wi * static_cast<int64_t>(support::cpp11::trunc(static_cast<double>(intermediate_val) / i32_wi));
            if(wrapped_rounded_val <= i32_hi)
            {
                return static_cast<int32_t>(wrapped_rounded_val);
            }
            else
            {
                // Values beyond i32_hi wrap around to negatives
                return static_cast<int32_t>((wrapped_rounded_val - i32_hi) + i32_lo - 1);
            }
        }
    }
    else
    {
        // Use double arithmetic for scale != 1; may not be bit-accurate
        // Apply scaling
        // scale == 1 / 2^scale_exponent
        int scale_exponent = 0;
        std::frexp(scale, &scale_exponent);
        // Store the positive exponent. We know that we compute 1/2^n
        // Additionally we need to subtract 1 to compensate that frexp used a mantissa of 0.5
        scale_exponent         = std::abs(scale_exponent - 1);
        const double scale_inv = static_cast<int64_t>(1) << scale_exponent;
        const double val       = intermediate_val / scale_inv;
        // Apply rounding
        double rounded_val = 0;
        switch(rounding_policy)
        {
            case(RoundingPolicy::TO_ZERO):
                rounded_val = support::cpp11::trunc(val);
                break;
            case(RoundingPolicy::TO_NEAREST_UP):
                rounded_val = round_half_up(val);
                break;
            case(RoundingPolicy::TO_NEAREST_EVEN):
                rounded_val = round_half_even(val);
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported rounding policy");
        }
        // Apply conversion
        if(convert_policy == ConvertPolicy::SATURATE)
        {
            return saturate_cast<int32_t>(rounded_val);
        }
        else
        {
            // Correct wrapping behaviour for int32_t
            const auto i32_hi              = static_cast<double>(std::numeric_limits<int32_t>::max());
            const auto i32_lo              = static_cast<double>(std::numeric_limits<int32_t>::lowest());
            const auto i32_wi              = static_cast<double>(static_cast<int64_t>(1) << 32);
            double     wrapped_rounded_val = rounded_val - i32_wi * std::floor(rounded_val / i32_wi);
            if(wrapped_rounded_val <= i32_hi)
            {
                return static_cast<int32_t>(wrapped_rounded_val);
            }
            else
            {
                // Values beyond i32_hi wrap around to negatives
                return static_cast<int32_t>((wrapped_rounded_val - i32_hi) + i32_lo - 1);
            }
        }
    }
}

template <size_t dim>
struct BroadcastUnroll
{
    template <typename T1, typename T2, typename T3>
    static void unroll(const SimpleTensor<T1> &src1, const SimpleTensor<T2> &src2, SimpleTensor<T3> &dst,
                       float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
                       Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        const bool src1_is_broadcast = (src1.shape()[dim - 1] != dst.shape()[dim - 1]);
        const bool src2_is_broadcast = (src2.shape()[dim - 1] != dst.shape()[dim - 1]);

        id_src1.set(dim - 1, 0);
        id_src2.set(dim - 1, 0);
        id_dst.set(dim - 1, 0);

        for(size_t i = 0; i < dst.shape()[dim - 1]; ++i, ++id_dst[dim - 1])
        {
            BroadcastUnroll < dim - 1 >::unroll(src1, src2, dst, scale, convert_policy, rounding_policy, id_src1, id_src2, id_dst);

            id_src1[dim - 1] += !src1_is_broadcast;
            id_src2[dim - 1] += !src2_is_broadcast;
        }
    }
};

template <>
struct BroadcastUnroll<0>
{
    template <typename T1, typename T2, typename T3>
    static void unroll(const SimpleTensor<T1> &src1, const SimpleTensor<T2> &src2, SimpleTensor<T3> &dst,
                       float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
                       Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        dst[coord2index(dst.shape(), id_dst)] = mul<T1, T2, T3>(src1[coord2index(src1.shape(), id_src1)], src2[coord2index(src2.shape(), id_src2)], scale, convert_policy, rounding_policy);
    }
};
} // namespace

template <typename T1, typename T2, typename T3>
SimpleTensor<T3> pixel_wise_multiplication(const SimpleTensor<T1> &src1, const SimpleTensor<T2> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
                                           DataType dt_out, const QuantizationInfo &qout)
{
    ARM_COMPUTE_UNUSED(qout);

    SimpleTensor<T3> dst(TensorShape::broadcast_shape(src1.shape(), src2.shape()), dt_out);

    if(scale < 0)
    {
        ARM_COMPUTE_ERROR("Scale of pixel-wise multiplication must be non-negative");
    }

    Coordinates id_src1{};
    Coordinates id_src2{};
    Coordinates id_dst{};

    BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(src1, src2, dst, scale, convert_policy, rounding_policy, id_src1, id_src2, id_dst);

    return dst;
}

template <>
SimpleTensor<uint8_t> pixel_wise_multiplication(const SimpleTensor<uint8_t> &src1, const SimpleTensor<uint8_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
                                                DataType dt_out, const QuantizationInfo &qout)
{
    SimpleTensor<uint8_t> dst(TensorShape::broadcast_shape(src1.shape(), src2.shape()), dt_out, 1, qout);

    if(src1.data_type() == DataType::QASYMM8 && src2.data_type() == DataType::QASYMM8)
    {
        SimpleTensor<float> src1_tmp = convert_from_asymmetric(src1);
        SimpleTensor<float> src2_tmp = convert_from_asymmetric(src2);
        SimpleTensor<float> dst_tmp  = pixel_wise_multiplication<float, float, float>(src1_tmp, src2_tmp, scale, convert_policy, rounding_policy, DataType::F32, qout);
        dst                          = convert_to_asymmetric<uint8_t>(dst_tmp, qout);
    }
    else
    {
        if(scale < 0)
        {
            ARM_COMPUTE_ERROR("Scale of pixel-wise multiplication must be non-negative");
        }

        Coordinates id_src1{};
        Coordinates id_src2{};
        Coordinates id_dst{};
        BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(src1, src2, dst, scale, convert_policy, rounding_policy, id_src1, id_src2, id_dst);
    }
    return dst;
}

template <>
SimpleTensor<int16_t> pixel_wise_multiplication(const SimpleTensor<uint8_t> &src1, const SimpleTensor<uint8_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
                                                DataType dt_out, const QuantizationInfo &qout)
{
    SimpleTensor<int16_t> dst(TensorShape::broadcast_shape(src1.shape(), src2.shape()), dt_out, 1, qout);

    if(src1.data_type() == DataType::QASYMM8 && src2.data_type() == DataType::QASYMM8)
    {
        SimpleTensor<float> src1_tmp = convert_from_asymmetric(src1);
        SimpleTensor<float> src2_tmp = convert_from_asymmetric(src2);
        SimpleTensor<float> dst_tmp  = pixel_wise_multiplication<float, float, float>(src1_tmp, src2_tmp, scale, convert_policy, rounding_policy, DataType::F32, qout);
        dst                          = convert_to_symmetric<int16_t>(dst_tmp, qout);
    }
    else
    {
        if(scale < 0)
        {
            ARM_COMPUTE_ERROR("Scale of pixel-wise multiplication must be non-negative");
        }

        Coordinates id_src1{};
        Coordinates id_src2{};
        Coordinates id_dst{};
        BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(src1, src2, dst, scale, convert_policy, rounding_policy, id_src1, id_src2, id_dst);
    }
    return dst;
}

template <>
SimpleTensor<int8_t> pixel_wise_multiplication(const SimpleTensor<int8_t> &src1, const SimpleTensor<int8_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
                                               DataType dt_out, const QuantizationInfo &qout)
{
    SimpleTensor<int8_t> dst(TensorShape::broadcast_shape(src1.shape(), src2.shape()), dt_out, 1, qout);

    if(src1.data_type() == DataType::QASYMM8_SIGNED && src2.data_type() == DataType::QASYMM8_SIGNED)
    {
        SimpleTensor<float> src1_tmp = convert_from_asymmetric(src1);
        SimpleTensor<float> src2_tmp = convert_from_asymmetric(src2);
        SimpleTensor<float> dst_tmp  = pixel_wise_multiplication<float, float, float>(src1_tmp, src2_tmp, scale, convert_policy, rounding_policy, DataType::F32, qout);
        dst                          = convert_to_asymmetric<int8_t>(dst_tmp, qout);
    }
    else
    {
        if(scale < 0)
        {
            ARM_COMPUTE_ERROR("Scale of pixel-wise multiplication must be non-negative");
        }

        Coordinates id_src1{};
        Coordinates id_src2{};
        Coordinates id_dst{};
        BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(src1, src2, dst, scale, convert_policy, rounding_policy, id_src1, id_src2, id_dst);
    }
    return dst;
}

template <>
SimpleTensor<int16_t> pixel_wise_multiplication(const SimpleTensor<int16_t> &src1, const SimpleTensor<int16_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
                                                DataType dt_out, const QuantizationInfo &qout)
{
    SimpleTensor<int16_t> dst(TensorShape::broadcast_shape(src1.shape(), src2.shape()), dt_out, 1, qout);

    if(src1.data_type() == DataType::QSYMM16 && src2.data_type() == DataType::QSYMM16)
    {
        SimpleTensor<float> src1_tmp = convert_from_symmetric<int16_t>(src1);
        SimpleTensor<float> src2_tmp = convert_from_symmetric<int16_t>(src2);
        SimpleTensor<float> dst_tmp  = pixel_wise_multiplication<float, float, float>(src1_tmp, src2_tmp, scale, convert_policy, rounding_policy, DataType::F32, qout);
        dst                          = convert_to_symmetric<int16_t>(dst_tmp, qout);
    }
    else
    {
        if(scale < 0)
        {
            ARM_COMPUTE_ERROR("Scale of pixel-wise multiplication must be non-negative");
        }

        Coordinates id_src1{};
        Coordinates id_src2{};
        Coordinates id_dst{};
        BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(src1, src2, dst, scale, convert_policy, rounding_policy, id_src1, id_src2, id_dst);
    }
    return dst;
}
// *INDENT-OFF*
// clang-format off
template SimpleTensor<int16_t> pixel_wise_multiplication(const SimpleTensor<uint8_t> &src1, const SimpleTensor<int16_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy, DataType dt_out, const QuantizationInfo &qout);
template SimpleTensor<int32_t> pixel_wise_multiplication(const SimpleTensor<int16_t> &src1, const SimpleTensor<int16_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy, DataType dt_out, const QuantizationInfo &qout);
template SimpleTensor<int32_t> pixel_wise_multiplication(const SimpleTensor<int32_t> &src1, const SimpleTensor<int32_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy, DataType dt_out, const QuantizationInfo &qout);
template SimpleTensor<float> pixel_wise_multiplication(const SimpleTensor<float> &src1, const SimpleTensor<float> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy, DataType dt_out, const QuantizationInfo &qout);
template SimpleTensor<half_float::half> pixel_wise_multiplication(const SimpleTensor<half_float::half> &src1, const SimpleTensor<half_float::half> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy, DataType dt_out, const QuantizationInfo &qout);
// clang-format on
// *INDENT-ON*
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
