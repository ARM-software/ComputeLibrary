/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_ELEMENTWISE_UNARY_LIST_H
#define SRC_CORE_NEON_KERNELS_ELEMENTWISE_UNARY_LIST_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"

namespace arm_compute
{
namespace cpu
{
template <typename ScalarType>
inline ScalarType elementwise_op_scalar_imp(ElementWiseUnary op, const ScalarType &a)
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

template <typename ScalarType, typename VectorType>
inline VectorType elementwise_op_imp(ElementWiseUnary op, const VectorType &a)
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

template <typename ScalarType>
void elementwise_op(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op)
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
            wrapper::vstore(output_ptr + x, elementwise_op_imp<ScalarType>(op, wrapper::vloadq(input_ptr + x)));
        }
        for(; x < window_end_x; ++x)
        {
            *(output_ptr + x) = elementwise_op_scalar_imp(op, *(input_ptr + x));
        }
    },
    input, output);
}

} // namespace cpu
} // namespace arm_compute

#endif // SRC_CORE_NEON_KERNELS_ELEMENTWISE_UNARY_LIST_H