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
#ifndef __ARM_COMPUTE_TEST_TENSOR_FACTORY_H__
#define __ARM_COMPUTE_TEST_TENSOR_FACTORY_H__

#include "RawTensor.h"
#include "Tensor.h"
#include "arm_compute/core/Error.h"

#include "boost_wrapper.h"

#if ARM_COMPUTE_ENABLE_FP16
#include <arm_fp16.h> // needed for float16_t
#endif

namespace arm_compute
{
namespace test
{
namespace validation
{
using TensorVariant = boost::variant < Tensor<uint8_t>, Tensor<int8_t>,
      Tensor<uint16_t>, Tensor<int16_t>,
      Tensor<uint32_t>, Tensor<int32_t>,
#ifdef ARM_COMPUTE_ENABLE_FP16
      Tensor<float16_t>,
#endif
      Tensor<float >>;

/** Helper to create a constant type if the passed reference is constant. */
template <typename R, typename T>
struct match_const
{
    using type = typename std::conditional<std::is_const<typename std::remove_reference<R>::type>::value, const T, T>::type;
};

class TensorFactory
{
public:
    template <typename R>
    static TensorVariant get_tensor(R &&raw)
    {
        TensorVariant v;
        DataType      dt                   = raw.data_type();
        int           fixed_point_position = raw.fixed_point_position();
        auto          shape                = raw.shape();
        auto          data                 = raw.data();

        switch(dt)
        {
            case DataType::U8:
                using value_type_u8 = typename match_const<R, uint8_t>::type;
                v                   = Tensor<uint8_t>(shape, dt, fixed_point_position, reinterpret_cast<value_type_u8 *>(data));
                break;
            case DataType::S8:
            case DataType::QS8:
                using value_type_s8 = typename match_const<R, int8_t>::type;
                v                   = Tensor<int8_t>(shape, dt, fixed_point_position, reinterpret_cast<value_type_s8 *>(data));
                break;
            case DataType::U16:
                using value_type_u16 = typename match_const<R, uint16_t>::type;
                v                    = Tensor<uint16_t>(shape, dt, fixed_point_position, reinterpret_cast<value_type_u16 *>(data));
                break;
            case DataType::S16:
            case DataType::QS16:
                using value_type_s16 = typename match_const<R, int16_t>::type;
                v                    = Tensor<int16_t>(shape, dt, fixed_point_position, reinterpret_cast<value_type_s16 *>(data));
                break;
            case DataType::U32:
                using value_type_u32 = typename match_const<R, uint32_t>::type;
                v                    = Tensor<uint32_t>(shape, dt, fixed_point_position, reinterpret_cast<value_type_u32 *>(data));
                break;
            case DataType::S32:
                using value_type_s32 = typename match_const<R, int32_t>::type;
                v                    = Tensor<int32_t>(shape, dt, fixed_point_position, reinterpret_cast<value_type_s32 *>(data));
                break;
#ifdef ARM_COMPUTE_ENABLE_FP16
            case DataType::F16:
                using value_type_f16 = typename match_const<R, float16_t>::type;
                v                    = Tensor<float16_t>(shape, dt, fixed_point_position, reinterpret_cast<value_type_f16 *>(data));
                break;
#endif
            case DataType::F32:
                using value_type_f32 = typename match_const<R, float>::type;
                v                    = Tensor<float>(shape, dt, fixed_point_position, reinterpret_cast<value_type_f32 *>(data));
                break;
            default:
                ARM_COMPUTE_ERROR("NOT SUPPORTED!");
        }
        return v;
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif /* __ARM_COMPUTE_TEST_TENSOR_FACTORY_H__ */
