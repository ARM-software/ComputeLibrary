/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NESelectKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "utils/TypePrinter.h"

#include <arm_neon.h>
#include <map>
#include <string>

namespace arm_compute
{
namespace
{
template <typename ScalarType, typename VectorType>
void select_op(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
               const int window_step_x, const int window_start_x, const int window_end_x, const int limit, VectorType (*condition_conversion)(const uint8_t *))
{
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator condition(cond, win);
    Iterator input1(in1, win);
    Iterator input2(in2, win);
    Iterator output(out, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        auto       output_ptr    = reinterpret_cast<ScalarType *>(output.ptr());
        const auto condition_ptr = reinterpret_cast<const uint8_t *>(condition.ptr());
        const auto input1_ptr    = reinterpret_cast<const ScalarType *>(input1.ptr());
        const auto input2_ptr    = reinterpret_cast<const ScalarType *>(input2.ptr());

        int x = window_start_x;
        for(; x <= limit; x += window_step_x)
        {
            const auto c = (*condition_conversion)(condition_ptr + x);
            const auto a = wrapper::vloadq(input1_ptr + x);
            const auto b = wrapper::vloadq(input2_ptr + x);
            wrapper::vstore(output_ptr + x, wrapper::vbsl(c, a, b));
        }
        for(; x < window_end_x; ++x)
        {
            const auto c      = *(condition_ptr + x);
            const auto a      = *(input1_ptr + x);
            const auto b      = *(input2_ptr + x);
            *(output_ptr + x) = static_cast<bool>(c) ? a : b;
        }
    },
    condition, input1, input2, output);
}

template <typename ScalarType, typename VectorType>
void select_op_8(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    const auto window_step_x  = 16 / sizeof(ScalarType);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    select_op<ScalarType, VectorType>(cond, in1, in2, out, window, window_step_x, window_start_x, window_end_x, window_end_x - window_step_x, [](const uint8_t *condition_ptr)
    {
        static const auto zero = wrapper::vdup_n(static_cast<uint8_t>(0), arm_compute::wrapper::traits::vector_128_tag());
        return wrapper::vcgt(wrapper::vloadq(condition_ptr), zero);
    });
}

template <typename ScalarType, typename VectorType>
void select_op_16(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    const auto window_step_x  = 16 / sizeof(ScalarType);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    select_op<ScalarType, VectorType>(cond, in1, in2, out, window, window_step_x, window_start_x, window_end_x, window_end_x - window_step_x, [](const uint8_t *condition_ptr)
    {
        static const auto zero = wrapper::vdup_n(static_cast<uint16_t>(0), arm_compute::wrapper::traits::vector_128_tag());
        return wrapper::vcgt(wrapper::vmovl(wrapper::vload(condition_ptr)), zero);
    });
}

template <typename ScalarType, typename VectorType>
void select_op_32(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    const auto window_step_x  = 16 / sizeof(ScalarType);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    select_op<ScalarType, VectorType>(cond, in1, in2, out, window, window_step_x, window_start_x, window_end_x, window_end_x - window_step_x, [](const uint8_t *condition_ptr)
    {
        static const auto zero = wrapper::vdup_n(static_cast<uint32_t>(0), arm_compute::wrapper::traits::vector_128_tag());
        return wrapper::vcgt(wrapper::vmovl(wrapper::vgetlow(wrapper::vmovl(wrapper::vload(condition_ptr)))), zero);
    });
}

template <typename ScalarType>
void select_op_not_same_rank(const ITensor *cond, const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    ARM_COMPUTE_UNUSED(window);

    auto       output_ptr    = reinterpret_cast<ScalarType *>(out->buffer());
    const auto condition_ptr = reinterpret_cast<const uint8_t *>(cond->buffer());
    const auto input1_ptr    = reinterpret_cast<const ScalarType *>(in1->buffer());
    const auto input2_ptr    = reinterpret_cast<const ScalarType *>(in2->buffer());

    const int outer_size = cond->info()->total_size() / cond->info()->element_size();
    const int inner_size = (in1->info()->total_size() / in1->info()->element_size()) / outer_size;
    int       offset     = 0;
    const int step       = 16 / in1->info()->element_size();

    for(int i = 0; i < outer_size; ++i)
    {
        int        x         = offset;
        const auto input_ptr = static_cast<bool>(*(condition_ptr + i)) ? input1_ptr : input2_ptr;
        for(; x <= offset + inner_size - step; x += step)
        {
            wrapper::vstore(output_ptr + x, wrapper::vloadq(input_ptr + x));
        }
        if(x <= offset + inner_size - (step / 2))
        {
            wrapper::vstore(output_ptr + x, wrapper::vload(input_ptr + x));
            x += step / 2;
        }
        for(; x < offset + inner_size; ++x)
        {
            *(output_ptr + x) = *(input_ptr + x);
        }
        offset += inner_size;
    }
}
} // namespace

NESelectKernel::NESelectKernel()
    : _function(nullptr), _c(nullptr), _x(nullptr), _y(nullptr), _output(nullptr), _has_same_rank(false)
{
}

void NESelectKernel::configure(const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(c, x, y, output);

    // Auto initialize output if not initialized
    auto_init_if_empty(*output->info(), x->info()->tensor_shape(), 1, x->info()->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(validate(c->info(), x->info(), y->info(), output->info()));

    _c             = c;
    _x             = x;
    _y             = y;
    _output        = output;
    _has_same_rank = (c->info()->tensor_shape().num_dimensions() == x->info()->tensor_shape().num_dimensions());

    std::string function_to_call("op_");
    function_to_call += string_from_data_type(x->info()->data_type());

    static std::map<std::string, SelectFunction *> map_function;

    if(_has_same_rank)
    {
        map_function =
        {
            { "op_S8", &select_op_8<int8_t, uint8x16_t> },
            { "op_S16", &select_op_16<int16_t, uint16x8_t> },
            { "op_S32", &select_op_32<int32_t, uint32x4_t> },
            { "op_U8", &select_op_8<uint8_t, uint8x16_t> },
            { "op_U16", &select_op_16<uint16_t, uint16x8_t> },
            { "op_U32", &select_op_32<uint32_t, uint32x4_t> },
            { "op_F32", &select_op_32<float, uint32x4_t> }
        };
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        map_function["op_F16"] = &select_op_16<float16_t, uint16x8_t>;
#endif /* ARM_COMPUTE_AARCH64_V8_2 */
    }
    else
    {
        map_function =
        {
            { "op_S8", &select_op_not_same_rank<int8_t> },
            { "op_S16", &select_op_not_same_rank<int16_t> },
            { "op_S32", &select_op_not_same_rank<int32_t> },
            { "op_U8", &select_op_not_same_rank<uint8_t> },
            { "op_U16", &select_op_not_same_rank<uint16_t> },
            { "op_U32", &select_op_not_same_rank<uint32_t> },
            { "op_F32", &select_op_not_same_rank<float> }
        };
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        map_function["op_F16"] = &select_op_not_same_rank<float16_t>;
#endif /* ARM_COMPUTE_AARCH64_V8_2 */
    }

    auto it = map_function.find(function_to_call);

    if(it != map_function.end())
    {
        _function = it->second;
    }

    Window win = calculate_max_window(x->info()->valid_region());
    INEKernel::configure(win);
}

Status NESelectKernel::validate(const ITensorInfo *c, const ITensorInfo *x, const ITensorInfo *y, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(x);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(x,
                                                         1,
                                                         DataType::U8, DataType::S8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(x, y);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(x, y);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(c, 1, DataType::U8);

    const bool is_same_rank = (c->tensor_shape().num_dimensions() == x->tensor_shape().num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(is_same_rank && (x->tensor_shape() != c->tensor_shape()));
    ARM_COMPUTE_RETURN_ERROR_ON(!is_same_rank && ((c->tensor_shape().num_dimensions() > 1) || (c->tensor_shape().x() != x->tensor_shape()[x->tensor_shape().num_dimensions() - 1])));

    if(output != nullptr && output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(x, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(x, output);
    }

    return Status{};
}

void NESelectKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_function == nullptr);
    _function(_c, _x, _y, _output, window);
}
} // namespace arm_compute
