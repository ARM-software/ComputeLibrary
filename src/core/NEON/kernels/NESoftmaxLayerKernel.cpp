/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/SaturateCast.h"

#include <algorithm>
#include <arm_neon.h>
#include <cfloat>
#include <functional>

namespace arm_compute
{
template <typename T, int N>
struct vec_n_type;

#define DECLARE_NEON_VEC_TYPE(T, N, V) \
    template <>                        \
    struct vec_n_type<T, N>            \
    {                                  \
        using type = V;                \
    };

DECLARE_NEON_VEC_TYPE(uint8_t, 16, uint8x16_t)
DECLARE_NEON_VEC_TYPE(uint8_t, 8, uint8x8_t)

DECLARE_NEON_VEC_TYPE(int8_t, 16, int8x16_t)
DECLARE_NEON_VEC_TYPE(int8_t, 8, int8x8_t)

DECLARE_NEON_VEC_TYPE(uint16_t, 8, uint16x8_t)
DECLARE_NEON_VEC_TYPE(uint16_t, 4, uint16x4_t)

DECLARE_NEON_VEC_TYPE(int16_t, 8, int16x8_t)
DECLARE_NEON_VEC_TYPE(int16_t, 4, int16x4_t)

DECLARE_NEON_VEC_TYPE(int32_t, 4, int32x4_t)
DECLARE_NEON_VEC_TYPE(int32_t, 2, int32x2_t)

DECLARE_NEON_VEC_TYPE(uint32_t, 4, uint32x4_t)
DECLARE_NEON_VEC_TYPE(uint32_t, 2, uint32x2_t)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
DECLARE_NEON_VEC_TYPE(float16_t, 8, float16x8_t)
DECLARE_NEON_VEC_TYPE(float16_t, 4, float16x4_t)
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

DECLARE_NEON_VEC_TYPE(float, 4, float32x4_t)
DECLARE_NEON_VEC_TYPE(float, 2, float32x2_t)

template <typename T, int N>
using vec_n_t = typename vec_n_type<T, N>::type;

template <typename T, int N>
using vec_n_byte_t = vec_n_t < T, N / sizeof(T) >;

template <typename T>
using vec_16_byte_t = vec_n_byte_t<T, 16>;

template <typename T>
using vec_8_byte_t = vec_n_byte_t<T, 8>;

template <typename T>
using const_ptr_t = const T *;

template <typename T>
using ptr_t = T *;

#define FORWARD_DECLARE_VGET_LANE_FOR_TYPE(TYPE) \
    template <int lane>                          \
    TYPE vget_lane(vec_8_byte_t<TYPE> vec);      \
    template <int lane>                          \
    TYPE vget_lane(vec_16_byte_t<TYPE> vec);

FORWARD_DECLARE_VGET_LANE_FOR_TYPE(uint8_t)
FORWARD_DECLARE_VGET_LANE_FOR_TYPE(int8_t)
FORWARD_DECLARE_VGET_LANE_FOR_TYPE(uint16_t)
FORWARD_DECLARE_VGET_LANE_FOR_TYPE(int16_t)
FORWARD_DECLARE_VGET_LANE_FOR_TYPE(uint32_t)
FORWARD_DECLARE_VGET_LANE_FOR_TYPE(int32_t)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
FORWARD_DECLARE_VGET_LANE_FOR_TYPE(float16_t)
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
FORWARD_DECLARE_VGET_LANE_FOR_TYPE(float)
template <int lane>
float vget_lane(float32x4x4_t vec);

template <typename V>
using elem_type_t = decltype(vget_lane<0>(std::declval<V>()));

template <typename V>
constexpr size_t vec_size_of(const V &vec)
{
    return sizeof(vec) / sizeof(elem_type_t<V>);
}

template <typename V>
V vdup_n(elem_type_t<V> val);
template <typename V>
V vld(const_ptr_t<elem_type_t<V>> ptr);

#define DECLARE_NEON_FUNCTIONS_FOR_TYPE(TYPE, TAG)                                \
    template <>                                                                   \
    inline vec_8_byte_t<TYPE> vdup_n<vec_8_byte_t<TYPE>>(TYPE val)                \
    {                                                                             \
        return vdup_n_##TAG(val);                                                 \
    }                                                                             \
    template <>                                                                   \
    inline vec_16_byte_t<TYPE> vdup_n<vec_16_byte_t<TYPE>>(TYPE val)              \
    {                                                                             \
        return vdupq_n_##TAG(val);                                                \
    }                                                                             \
    template <>                                                                   \
    inline vec_8_byte_t<TYPE> vld<vec_8_byte_t<TYPE>>(const_ptr_t<TYPE> ptr)      \
    {                                                                             \
        return vld1_##TAG(ptr);                                                   \
    }                                                                             \
    template <>                                                                   \
    inline vec_16_byte_t<TYPE> vld<vec_16_byte_t<TYPE>>(const_ptr_t<TYPE> ptr)    \
    {                                                                             \
        return vld1q_##TAG(ptr);                                                  \
    }                                                                             \
    inline void vst(ptr_t<TYPE> ptr, vec_8_byte_t<TYPE> vec)                      \
    {                                                                             \
        vst1_##TAG(ptr, vec);                                                     \
    }                                                                             \
    inline void vst(ptr_t<TYPE> ptr, vec_16_byte_t<TYPE> vec)                     \
    {                                                                             \
        vst1q_##TAG(ptr, vec);                                                    \
    }                                                                             \
    inline vec_16_byte_t<TYPE> vmax(vec_16_byte_t<TYPE> a, vec_16_byte_t<TYPE> b) \
    {                                                                             \
        return vmaxq_##TAG(a, b);                                                 \
    }                                                                             \
    inline vec_8_byte_t<TYPE> vpmax(vec_8_byte_t<TYPE> a, vec_8_byte_t<TYPE> b)   \
    {                                                                             \
        return vpmax_##TAG(a, b);                                                 \
    }                                                                             \
    inline vec_8_byte_t<TYPE> vget_low(vec_16_byte_t<TYPE> vec)                   \
    {                                                                             \
        return vget_low_##TAG(vec);                                               \
    }                                                                             \
    inline vec_8_byte_t<TYPE> vget_high(vec_16_byte_t<TYPE> vec)                  \
    {                                                                             \
        return vget_high_##TAG(vec);                                              \
    }                                                                             \
    template <int lane>                                                           \
    inline TYPE vget_lane(vec_8_byte_t<TYPE> vec)                                 \
    {                                                                             \
        static_assert(lane >= 0, "lane is out of bounds");                        \
        static_assert(lane < vec_size_of(vec), "lane is out of bounds");          \
        return vget_lane_##TAG(vec, lane);                                        \
    }                                                                             \
    template <int lane>                                                           \
    inline TYPE vget_lane(vec_16_byte_t<TYPE> vec)                                \
    {                                                                             \
        static_assert(lane >= 0, "lane is out of bounds");                        \
        static_assert(lane < vec_size_of(vec), "lane is out of bounds");          \
        return vgetq_lane_##TAG(vec, lane);                                       \
    }

template <typename T>
T sqadd(T a, T b);
template <typename T>
T sqsub(T a, T b);
template <typename T>
T sqmul(T a, T b);

#define DECLARE_NEON_FUNCTIONS_FOR_FLOAT(TYPE, TAG)                               \
    inline vec_8_byte_t<TYPE> vadd(vec_8_byte_t<TYPE> a, vec_8_byte_t<TYPE> b)    \
    {                                                                             \
        return vadd_##TAG(a, b);                                                  \
    }                                                                             \
    inline vec_16_byte_t<TYPE> vadd(vec_16_byte_t<TYPE> a, vec_16_byte_t<TYPE> b) \
    {                                                                             \
        return vaddq_##TAG(a, b);                                                 \
    }                                                                             \
    inline vec_16_byte_t<TYPE> vsub(vec_16_byte_t<TYPE> a, vec_16_byte_t<TYPE> b) \
    {                                                                             \
        return vsubq_##TAG(a, b);                                                 \
    }                                                                             \
    inline vec_16_byte_t<TYPE> vmul_n(vec_16_byte_t<TYPE> vec, TYPE val)          \
    {                                                                             \
        return vmulq_n_##TAG(vec, val);                                           \
    }

DECLARE_NEON_FUNCTIONS_FOR_TYPE(uint8_t, u8)
DECLARE_NEON_FUNCTIONS_FOR_TYPE(int8_t, s8)
DECLARE_NEON_FUNCTIONS_FOR_TYPE(uint16_t, u16)
DECLARE_NEON_FUNCTIONS_FOR_TYPE(int16_t, s16)
DECLARE_NEON_FUNCTIONS_FOR_TYPE(uint32_t, u32)
DECLARE_NEON_FUNCTIONS_FOR_TYPE(int32_t, s32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
DECLARE_NEON_FUNCTIONS_FOR_TYPE(float16_t, f16)
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
DECLARE_NEON_FUNCTIONS_FOR_TYPE(float, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
DECLARE_NEON_FUNCTIONS_FOR_FLOAT(float16_t, f16)
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
DECLARE_NEON_FUNCTIONS_FOR_FLOAT(float, f32)

template <typename VO, typename VI>
VO vcvt(VI vec);

template <>
float32x4x4_t vcvt<float32x4x4_t>(uint8x16_t vec)
{
    const auto    low  = vmovl_u8(vget_low(vec));
    const auto    high = vmovl_u8(vget_high(vec));
    float32x4x4_t res  = { {
            vcvtq_f32_u32(vmovl_u16(vget_low(low))),
            vcvtq_f32_u32(vmovl_u16(vget_high(low))),
            vcvtq_f32_u32(vmovl_u16(vget_low(high))),
            vcvtq_f32_u32(vmovl_u16(vget_high(high)))
        }
    };
    return res;
}

template <>
uint8x16_t vcvt<uint8x16_t>(float32x4x4_t vec)
{
    uint16x8x2_t resU16 = { {
            vcombine_u16(vqmovn_u32(vcvtq_u32_f32(vec.val[0])),
            vqmovn_u32(vcvtq_u32_f32(vec.val[1]))),
            vcombine_u16(vqmovn_u32(vcvtq_u32_f32(vec.val[2])),
            vqmovn_u32(vcvtq_u32_f32(vec.val[3])))
        }
    };

    uint8x16_t res = vcombine_u8(vqmovn_u16(resU16.val[0]), vqmovn_u16(resU16.val[1]));
    return res;
}

float32x4x4_t vexp(float32x4x4_t vec)
{
    float32x4x4_t res = { {
            vexpq_f32(vec.val[0]),
            vexpq_f32(vec.val[1]),
            vexpq_f32(vec.val[2]),
            vexpq_f32(vec.val[3])
        }
    };
    return res;
}

float32x4_t vexp(const float32x4_t &vec)
{
    return vexpq_f32(vec);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
// TODO (COMPMID-1535) : Revisit FP16 approximations
float16x8_t vexp(const float16x8_t &vec)
{
    float16x4x2_t res =
    {
        {
            vcvt_f16_f32(vexpq_f32(vcvt_f32_f16(vget_low_f16(vec)))),
            vcvt_f16_f32(vexpq_f32(vcvt_f32_f16(vget_high_f16(vec))))
        }
    };
    return vcombine_f16(res.val[0], res.val[1]);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <>
float32x4x4_t vdup_n<float32x4x4_t>(float val)
{
    float32x4x4_t res = { {
            vdupq_n_f32(val),
            vdupq_n_f32(val),
            vdupq_n_f32(val),
            vdupq_n_f32(val)
        }
    };
    return res;
}

float32x4x4_t vmul_n(float32x4x4_t vec, float val)
{
    float32x4x4_t res = { {
            vmulq_n_f32(vec.val[0], val),
            vmulq_n_f32(vec.val[1], val),
            vmulq_n_f32(vec.val[2], val),
            vmulq_n_f32(vec.val[3], val)
        }
    };
    return res;
}

float32x4x4_t vadd(float32x4x4_t a, float32x4x4_t b)
{
    float32x4x4_t res = { {
            vaddq_f32(a.val[0], b.val[0]),
            vaddq_f32(a.val[1], b.val[1]),
            vaddq_f32(a.val[2], b.val[2]),
            vaddq_f32(a.val[3], b.val[3])
        }
    };
    return res;
}

namespace
{
Status validate_arguments_logits_1d_max(const ITensorInfo &input, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);

    // Validate in case of configured output
    if(output.total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output.tensor_shape(), TensorShape(input.tensor_shape()).set(0, 1));
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_logits_1d_max(ITensorInfo &input, ITensorInfo &output)
{
    // Softmax across the x dimension
    const TensorShape output_shape = TensorShape(input.tensor_shape()).set(0, 1);
    // Output auto initialization if not yet initialized
    auto_init_if_empty(output, output_shape, 1, input.data_type(), input.quantization_info());

    // Configure kernel window
    const int input_width                       = input.valid_region().shape.x();
    const int num_elems_processed_per_iteration = 16U / data_size_from_type(input.data_type());
    const int num_elems_read_per_iteration      = ceil_to_multiple(input_width, num_elems_processed_per_iteration);

    const ValidRegion out_valid_region(ValidRegion(input.valid_region()).set(0, 0, 1));
    output.set_valid_region(out_valid_region);

    Window win = calculate_max_window(output);

    AccessWindowHorizontal input_access(&input, input.valid_region().anchor.x(), num_elems_read_per_iteration);
    AccessWindowHorizontal output_access(&output, 0, 1);

    const bool window_changed = update_window_and_padding(win, input_access, output_access);

    const Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

template <typename V>
auto reduce_max(V vec) -> elem_type_t<V>
{
    constexpr int N = vec_size_of(vec);

    auto carry_max = vpmax(vget_high(vec), vget_low(vec));

    for(int k = N / 2; k > 1; k /= 2)
    {
        carry_max = vpmax(carry_max, carry_max);
    }

    return vget_lane<0>(carry_max);
}

template <typename T>
void logits_1d_max(const ITensor &in, ITensor &out, const Window &window)
{
    const auto   start_x     = in.info()->valid_region().anchor.x();
    const size_t input_width = in.info()->valid_region().shape.x();

    Iterator input(&in, window);
    Iterator output(&out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        // Get pointers
        const auto in_ptr  = reinterpret_cast<const T *>(input.ptr()) + start_x;
        const auto out_ptr = reinterpret_cast<T *>(output.ptr());

        // Init max value
        auto vec_max = vdup_n<vec_16_byte_t<T>>(support::cpp11::lowest<T>());

        // Loop over input row
        for(const T *it = in_ptr; it < (in_ptr + input_width); it += vec_size_of(vec_max))
        {
            const auto current_value = vld<vec_16_byte_t<T>>(it);
            vec_max                  = vmax(vec_max, current_value);
        }

        const T max_val = reduce_max(vec_max);
        *out_ptr        = max_val;
    },
    input, output);
}
} // namespace

NELogits1DMaxKernel::NELogits1DMaxKernel()
    : _func(nullptr), _border_size()
{
}

BorderSize NELogits1DMaxKernel::border_size() const
{
    return _border_size;
}

void NELogits1DMaxKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input->info(), output->info());
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_1d_max(*input->info(), *output->info()));
    // Configure kernel window
    auto win_config = validate_and_configure_window_logits_1d_max(*input->info(), *output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    switch(input->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = &logits_1d_max<qasymm8_t>;
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &logits_1d_max<float16_t>;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
            _func = &logits_1d_max<float>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }

    _input  = input;
    _output = output;

    const int input_width                       = input->info()->valid_region().shape.x();
    const int num_elems_processed_per_iteration = 16U / data_size_from_type(input->info()->data_type());
    const int num_elems_read_per_iteration      = ceil_to_multiple(input_width, num_elems_processed_per_iteration);

    _border_size = BorderSize(0, num_elems_read_per_iteration - input_width, 0, 0);

    INEKernel::configure(win_config.second);
}

Status NELogits1DMaxKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_logits_1d_max(*input, *output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_logits_1d_max(*input->clone(), *output->clone()).first);

    return Status{};
}

void NELogits1DMaxKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(*_input, *_output, window);
}

namespace
{
Status validate_arguments_logits_softmax(const ITensorInfo &input, const ITensorInfo &max,
                                         const ITensorInfo &output, const float beta, const ITensorInfo &tmp)
{
    ARM_COMPUTE_UNUSED(beta);
    // Check input
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);

    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(input.data_type());

    // Check max
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &max);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(TensorShape(input.tensor_shape()).set(0, 1), max.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&input, &max);

    // Check output if configured
    if(output.total_size() != 0)
    {
        const QuantizationInfo output_quantization = is_quantized_asymmetric ? QuantizationInfo(1.f / 256.f, 0) : output.quantization_info();
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON(output.quantization_info() != output_quantization);
    }

    // Check tmp if configured
    if(tmp.total_size() != 0)
    {
        const DataType tmp_data_type = is_quantized_asymmetric ? DataType::F32 : input.data_type();
        ARM_COMPUTE_RETURN_ERROR_ON(tmp.data_type() != tmp_data_type);
        // We could potentially reduce tmp memory if we could predict or make an assumption
        // on the maximum number of threads that will run in parallel.
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&input, &tmp);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_logits_softmax(ITensorInfo &input, ITensorInfo &max,
                                                                       ITensorInfo &output, ITensorInfo &tmp)
{
    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(input.data_type());

    // Output auto initialization if not yet initialized
    const QuantizationInfo output_quantization = is_quantized_asymmetric ? QuantizationInfo(1.f / 256.f, 0) : output.quantization_info();
    auto_init_if_empty(output, TensorInfo(input).set_quantization_info(output_quantization).reset_padding());

    // Tmp auto initialization if not yet initialized
    const DataType tmp_data_type = is_quantized_asymmetric ? DataType::F32 : input.data_type();
    auto_init_if_empty(tmp, TensorInfo(input).set_data_type(tmp_data_type).reset_padding());

    const int input_width = input.valid_region().shape.x();

    Window win = calculate_max_window(max);

    AccessWindowHorizontal input_access(&input, input.valid_region().anchor.x(), input_width);
    AccessWindowHorizontal max_access(&input, 0, 1);
    AccessWindowHorizontal output_access(&output, input.valid_region().anchor.x(), input_width);
    AccessWindowHorizontal tmp_access(&tmp, input.valid_region().anchor.x(), input_width);

    const bool window_changed = update_window_and_padding(win, input_access, max_access, output_access, tmp_access);

    output.set_valid_region(input.valid_region());

    const Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

template <typename T, int N, int S, int E>
struct reduce_add_impl
{
    template <typename F>
    static T reduce(F add_fn, vec_n_t<T, N> vec)
    {
        constexpr int H            = (S + E + 1) / 2;
        const auto    reduced_high = reduce_add_impl < T, N, S, H - 1 >::reduce(add_fn, vec);
        const auto    reduced_low  = reduce_add_impl<T, N, H, E>::reduce(add_fn, vec);
        return add_fn(reduced_high, reduced_low);
    }
};
template <typename T, int N, int I>
struct reduce_add_impl<T, N, I, I>
{
    template <typename F>
    static T reduce(F /*add_fn*/, vec_n_t<T, N> vec)
    {
        return vget_lane<I>(vec);
    }
};
template <typename V, typename F>
elem_type_t<V> reduce_add(F add_fn, V vec)
{
    constexpr int N = vec_size_of(vec);
    return reduce_add_impl < elem_type_t<V>, N, 0, N - 1 >::reduce(add_fn, vec);
}

void logits_1d_softmax_qasymm8(const ITensor &in, const ITensor &max, void *const tmp, ITensor &out, const float beta, const Window &window)
{
    const int start_x     = in.info()->valid_region().anchor.x();
    const int input_width = in.info()->valid_region().shape.x();

    const float scale_beta = -beta * in.info()->quantization_info().scale;

    Iterator in_it(&in, window);
    Iterator max_it(&max, window);
    Iterator out_it(&out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        /* Get pointers */
        const auto in_ptr  = reinterpret_cast<const qasymm8_t *>(in_it.ptr()) + start_x;
        const auto out_ptr = reinterpret_cast<qasymm8_t *>(out_it.ptr()) + start_x;
        const auto tmp_ptr = reinterpret_cast<float *>(tmp);

        float sum_inversed;

        /* Compute exponentials and sum */
        {
            /* Get max value */
            const auto max_val = *reinterpret_cast<const qasymm8_t *>(max_it.ptr());
            const auto vec_max = vdup_n<vec_16_byte_t<qasymm8_t>>(max_val);

            /* Init sum to zero */
            auto vec_sum = vdup_n<float32x4x4_t>(0.f);

            /* Loop over row and compute exponentials and sum */
            int           i        = 0;
            constexpr int vec_size = vec_size_of(vec_max);
            for(; i <= (input_width - vec_size); i += vec_size)
            {
                auto vec_elements = vld<vec_16_byte_t<qasymm8_t>>(in_ptr + i);
                vec_elements      = vsubq_u8(vec_max, vec_elements);

                auto vec_elements_flt = vcvt<float32x4x4_t>(vec_elements);
                vec_elements_flt      = vexp(vmul_n(vec_elements_flt, scale_beta));

                vec_sum = vadd(vec_sum, vec_elements_flt);

                vst4q_f32(tmp_ptr + i, vec_elements_flt);
            }
            /* Reduce sum */
            const auto sum_16_byte = vaddq_f32(vaddq_f32(vec_sum.val[0], vec_sum.val[1]),
                                               vaddq_f32(vec_sum.val[2], vec_sum.val[3]));
            const auto sum_8_byte = vadd_f32(vget_low(sum_16_byte), vget_high(sum_16_byte));
            float      sum        = reduce_add(std::plus<float>(), sum_8_byte);

            /* Run remaining elements */
            for(; i < input_width; ++i)
            {
                const float element = std::exp((max_val - in_ptr[i]) * scale_beta);
                sum += element;
                tmp_ptr[i] = element;
            }

            sum_inversed = 256.f / sum;
        }

        /* Normalize exponentials */
        {
            /* Loop over row and compute softmax */
            int i = 0;
            {
                constexpr int vec_size = 16;
                for(; i <= (input_width - vec_size); i += vec_size)
                {
                    float32x4x4_t vec_in           = vld4q_f32(tmp_ptr + i);
                    auto          normalized_value = vcvt<vec_16_byte_t<qasymm8_t>>(vmul_n(vec_in, sum_inversed));
                    vst(out_ptr + i, normalized_value);
                }
            }
            /* Run remaining elements */
            for(; i < input_width; ++i)
            {
                out_ptr[i] = utils::cast::saturate_cast<qasymm8_t>(tmp_ptr[i] * sum_inversed);
            }
        }
    },
    in_it, max_it, out_it);
}

template <typename T>
void logits_1d_softmax_float(const ITensor &in, const ITensor &max, void *const tmp,
                             ITensor &out, const float beta, const Window &window)
{
    const int start_x     = in.info()->valid_region().anchor.x();
    const int input_width = in.info()->valid_region().shape.x();

    Iterator in_it(&in, window);
    Iterator max_it(&max, window);
    Iterator out_it(&out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        /* Get pointers */
        const auto in_ptr  = reinterpret_cast<const T *>(in_it.ptr()) + start_x;
        const auto out_ptr = reinterpret_cast<T *>(out_it.ptr()) + start_x;
        const auto tmp_ptr = reinterpret_cast<T *>(tmp);

        T sum_inversed;

        /* Compute exponentials and sum */
        {
            /* Get max value */
            const auto max_val = *reinterpret_cast<const T *>(max_it.ptr());
            const auto vec_max = vdup_n<vec_16_byte_t<T>>(max_val);

            /* Init sum to zero */
            auto vec_sum = vdup_n<vec_16_byte_t<T>>(0);

            /* Loop over row and compute exponentials and sum */
            int           i        = 0;
            constexpr int vec_size = vec_size_of(vec_sum);
            for(; i <= (input_width - vec_size); i += vec_size)
            {
                auto vec_elements = vld<vec_16_byte_t<T>>(in_ptr + i);
                vec_elements      = vsub(vec_elements, vec_max);
                vec_elements      = vexp(vmul_n(vec_elements, static_cast<T>(beta)));
                vec_sum           = vadd(vec_sum, vec_elements);
                vst(tmp_ptr + i, vec_elements);
            }
            /* Reduce sum */
            const auto sum_8_byte = vadd(vget_high(vec_sum), vget_low(vec_sum));
            T sum                 = reduce_add([](T a, T b) -> T { return a + b; }, sum_8_byte);

            /* Run remaining elements */
            for(; i < input_width; ++i)
            {
                T element = std::exp((in_ptr[i] - max_val) * beta);
                sum += element;
                tmp_ptr[i] = element;
            }

            sum_inversed = T(1) / sum;
        }

        /* Normalize exponentials */
        {
            /* Loop over row and compute softmax */
            int i = 0;
            {
                constexpr int vec_size = vec_size_of(vec_16_byte_t<T> {});
                for(; i <= (input_width - vec_size); i += vec_size)
                {
                    auto             vec_in           = vld<vec_16_byte_t<T>>(tmp_ptr + i);
                    vec_16_byte_t<T> normalized_value = vmul_n(vec_in, sum_inversed);
                    vst(out_ptr + i, normalized_value);
                }
            }
            /* Run remaining elements */
            for(; i < input_width; ++i)
            {
                out_ptr[i] = tmp_ptr[i] * sum_inversed;
            }
        }
    },
    in_it, max_it, out_it);
}
} // namespace

NELogits1DSoftmaxKernel::NELogits1DSoftmaxKernel()
    : _func(nullptr), _input(nullptr), _max(nullptr), _output(nullptr), _beta(1.0f), _tmp(nullptr)
{
}

void NELogits1DSoftmaxKernel::configure(const ITensor *input, const ITensor *max, ITensor *output, const float beta, ITensor *tmp)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, max, output, tmp);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input->info(), max->info(), output->info(), tmp->info());
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_softmax(*input->info(), *max->info(), *output->info(), beta, *tmp->info()));
    // Configure kernel window
    auto win_config = validate_and_configure_window_logits_softmax(*input->info(), *max->info(), *output->info(), *tmp->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    switch(input->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = &logits_1d_softmax_qasymm8;
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &logits_1d_softmax_float<float16_t>;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
            _func = &logits_1d_softmax_float<float>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
            break;
    }

    _input  = input;
    _max    = max;
    _output = output;
    _beta   = beta;
    _tmp    = tmp;

    INEKernel::configure(win_config.second);
}

Status NELogits1DSoftmaxKernel::validate(const ITensorInfo *input, const ITensorInfo *max,
                                         const ITensorInfo *output, const float beta, const ITensorInfo *tmp)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, max, output, tmp);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_logits_softmax(*input, *max, *output, beta, *tmp));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_logits_softmax(*input->clone(), *max->clone(), *output->clone(), *tmp->clone()).first);

    return Status{};
}

void NELogits1DSoftmaxKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const unsigned int num_elems_processed_per_iteration = _input->info()->valid_region().shape.x();
    const unsigned int tmp_size_for_thread               = _tmp->info()->element_size() * num_elems_processed_per_iteration;

    ARM_COMPUTE_ERROR_ON(_tmp->info()->total_size() < (info.num_threads * tmp_size_for_thread));

    void *tmp_for_thread = _tmp->buffer() + (info.thread_id * tmp_size_for_thread);

    (*_func)(*_input, *_max, tmp_for_thread, *_output, _beta, window);
}

} // namespace arm_compute
