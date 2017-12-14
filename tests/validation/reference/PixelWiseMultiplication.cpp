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
 * dst OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "PixelWiseMultiplication.h"

#include "tests/validation/FixedPoint.h"

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

template <typename T1, typename T2>
SimpleTensor<T2> pixel_wise_multiplication(const SimpleTensor<T1> &src1, const SimpleTensor<T2> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    SimpleTensor<T2> dst(src2.shape(), src2.data_type());

    if(scale < 0)
    {
        ARM_COMPUTE_ERROR("Scale of pixel-wise multiplication must be non-negative");
    }

    using intermediate_type = typename common_promoted_signed_type<T1, T2, T2>::intermediate_type;

    for(int i = 0; i < src1.num_elements(); ++i)
    {
        double val = static_cast<intermediate_type>(src1[i]) * static_cast<intermediate_type>(src2[i]) * static_cast<double>(scale);
        if(is_floating_point<T2>::value)
        {
            dst[i] = val;
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

            dst[i] = (convert_policy == ConvertPolicy::SATURATE) ? saturate_cast<T2>(rounded_val) : static_cast<T2>(rounded_val);
        }
    }

    return dst;
}

// *INDENT-OFF*
// clang-format off
template SimpleTensor<uint8_t> pixel_wise_multiplication(const SimpleTensor<uint8_t> &src1, const SimpleTensor<uint8_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy);
template SimpleTensor<int16_t> pixel_wise_multiplication(const SimpleTensor<uint8_t> &src1, const SimpleTensor<int16_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy);
template SimpleTensor<int16_t> pixel_wise_multiplication(const SimpleTensor<int16_t> &src1, const SimpleTensor<int16_t> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy);
template SimpleTensor<float> pixel_wise_multiplication(const SimpleTensor<float> &src1, const SimpleTensor<float> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy);
template SimpleTensor<half_float::half> pixel_wise_multiplication(const SimpleTensor<half_float::half> &src1, const SimpleTensor<half_float::half> &src2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy);
// clang-format on
// *INDENT-ON*
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
