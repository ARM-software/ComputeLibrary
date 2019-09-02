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
#include "arm_compute/core/NEON/kernels/NEElementwiseUnaryKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <algorithm>
#include <arm_neon.h>
#include <cstdint>
#include <map>
#include <string>

namespace arm_compute
{
class Coordinates;

namespace
{
template <ElementWiseUnary op, typename ScalarType>
inline ScalarType elementwise_op_scalar(const ScalarType &a)
{
    switch(op)
    {
        case ElementWiseUnary::RSQRT:
            return 1 / sqrt(a);
        case ElementWiseUnary::EXP:
            return std::exp(a);
        case ElementWiseUnary::NEG:
            return -a;
        case ElementWiseUnary::LOG:
            return std::log(a);
        case ElementWiseUnary::ABS:
            return std::abs(a);
        case ElementWiseUnary::ROUND:
            return support::cpp11::nearbyint(a);
        case ElementWiseUnary::SIN:
            return std::sin(a);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
}

/* Elementwise operations that are supported for float */
template <ElementWiseUnary op, typename ScalarType, bool is_float, typename VectorType, typename std::enable_if<is_float, int>::type = 0>
inline VectorType elementwise_op(const VectorType &a)
{
    switch(op)
    {
        case ElementWiseUnary::RSQRT:
            return wrapper::vinvsqrt(a);
        case ElementWiseUnary::EXP:
            return wrapper::vexpq(a);
        case ElementWiseUnary::NEG:
            return wrapper::vneg(a);
        case ElementWiseUnary::LOG:
            return wrapper::vlog(a);
        case ElementWiseUnary::ABS:
            return wrapper::vabs(a);
        case ElementWiseUnary::ROUND:
            return wrapper::vround(a);
        case ElementWiseUnary::SIN:
            return wrapper::vsin(a);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
}

/* Elementwise operations that are supported for non floats */
template < ElementWiseUnary op, typename ScalarType, bool is_float, typename VectorType, typename std::enable_if < !is_float, int >::type = 0 >
inline VectorType elementwise_op(const VectorType &a)
{
    switch(op)
    {
        case ElementWiseUnary::NEG:
            return wrapper::vneg(a);
        case ElementWiseUnary::ABS:
            return wrapper::vabs(a);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
}

template <ElementWiseUnary op, typename ScalarType, bool is_float>
void elementwise_op(const ITensor *in, ITensor *out, const Window &window)
{
    const int  window_step_x  = 16 / sizeof(ScalarType);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(in, win);
    Iterator output(out, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        auto       output_ptr = reinterpret_cast<ScalarType *>(output.ptr());
        const auto input_ptr  = reinterpret_cast<const ScalarType *>(input.ptr());

        int x = window_start_x;
        for(; x <= window_end_x - window_step_x; x += window_step_x)
        {
            wrapper::vstore(output_ptr + x, elementwise_op<op, ScalarType, is_float>(wrapper::vloadq(input_ptr + x)));
        }
        for(; x < window_end_x; ++x)
        {
            *(output_ptr + x) = elementwise_op_scalar<op>(*(input_ptr + x));
        }
    },
    input, output);
}

template <ElementWiseUnary op>
std::function<void(const ITensor *input, ITensor *output, const Window &window)>
configure_func(const ITensor *input, ITensor *output)
{
    std::string function_to_call("op_");
    function_to_call += string_from_data_type(input->info()->data_type()) + "_";
    function_to_call += string_from_data_type(output->info()->data_type());

    static std::map<std::string, NEElementwiseUnaryKernel::ElementwiseUnaryFunction *> map_function =
    {
        { "op_F32_F32", &elementwise_op<op, float, true> },
        { "op_S32_S32", &elementwise_op<op, int32_t, false> },
    };
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    map_function["op_F16_F16"] = &elementwise_op<op, float16_t, true>;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

    auto it = map_function.find(function_to_call);

    if(it != map_function.end())
    {
        auto func = it->second;
        return [func](const ITensor * input, ITensor * output, const Window & window)
        {
            func(input, output, window);
        };
    }
    return nullptr;
}
} // namespace

NEElementwiseUnaryKernel::NEElementwiseUnaryKernel()
    : _function(nullptr), _input(nullptr), _output(nullptr)
{
}

void NEElementwiseUnaryKernel::configure(ElementWiseUnary op, const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(op, *input->info(), *output->info()));
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Configure kernel window
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*input->info());
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    auto_init_if_empty(*output->info(), out_shape, 1, input->info()->data_type());

    Window win = calculate_max_window(valid_region);

    _input  = input;
    _output = output;

    INEKernel::configure(win);

    switch(op)
    {
        case ElementWiseUnary::RSQRT:
            _function = configure_func<ElementWiseUnary::RSQRT>(input, output);
            break;
        case ElementWiseUnary::EXP:
            _function = configure_func<ElementWiseUnary::EXP>(input, output);
            break;
        case ElementWiseUnary::NEG:
            _function = configure_func<ElementWiseUnary::NEG>(input, output);
            break;
        case ElementWiseUnary::LOG:
            _function = configure_func<ElementWiseUnary::LOG>(input, output);
            break;
        case ElementWiseUnary::ABS:
            _function = configure_func<ElementWiseUnary::ABS>(input, output);
            break;
        case ElementWiseUnary::ROUND:
            _function = configure_func<ElementWiseUnary::ROUND>(input, output);
            break;
        case ElementWiseUnary::SIN:
            _function = configure_func<ElementWiseUnary::SIN>(input, output);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
}

Status NEElementwiseUnaryKernel::validate_arguments(ElementWiseUnary op, const ITensorInfo &input, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input);
    switch(op)
    {
        case ElementWiseUnary::EXP:
        case ElementWiseUnary::RSQRT:
        case ElementWiseUnary::LOG:
        case ElementWiseUnary::ROUND:
        case ElementWiseUnary::SIN:
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::F16, DataType::F32);
            break;
        case ElementWiseUnary::NEG:
        case ElementWiseUnary::ABS:
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::F16, DataType::F32, DataType::S32);
            break;
        default:
            ARM_COMPUTE_ERROR("ElementWiseUnary operation not supported");
    }
    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
    }

    return Status{};
}

Status NEElementwiseUnaryKernel::validate(ElementWiseUnary op, const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(op, *input, *output));
    return Status{};
}

void NEElementwiseUnaryKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_function == nullptr);
    _function(_input, _output, window);
}
} // namespace arm_compute
