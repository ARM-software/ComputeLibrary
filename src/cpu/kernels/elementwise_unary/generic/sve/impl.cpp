/*
 * Copyright (c) 2022 Arm Limited.
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

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"

namespace arm_compute
{
namespace cpu
{
template <typename ScalarType, typename VectorType>
inline typename std::enable_if<utils::traits::is_floating_point<ScalarType>::value, VectorType>::type elementwise_op_sve_imp(svbool_t pg, ElementWiseUnary op, const VectorType &a)
{
    switch(op)
    {
        case ElementWiseUnary::RSQRT:
            return svinvsqrt(pg, a);
        case ElementWiseUnary::EXP:
            return wrapper::svexp_z(pg, a);
        case ElementWiseUnary::NEG:
            return svneg_z(pg, a);
        case ElementWiseUnary::LOG:
            return wrapper::svlog_z(pg, a);
        case ElementWiseUnary::ABS:
            return svabs_z(pg, a);
        case ElementWiseUnary::ROUND:
            return svrintn_z(pg, a);
        case ElementWiseUnary::SIN:
            return wrapper::svsin_z(pg, a);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED");
    }
}

template <typename ScalarType, typename VectorType>
inline typename std::enable_if<std::is_integral<ScalarType>::value, VectorType>::type elementwise_op_sve_imp(svbool_t pg, ElementWiseUnary op, const VectorType &a)
{
    switch(op)
    {
        case ElementWiseUnary::NEG:
            return svneg_z(pg, a);
        case ElementWiseUnary::ABS:
            return svabs_z(pg, a);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED");
    }
}

template <typename ScalarType>
void elementwise_sve_op(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op)
{
    const auto all_true_pg    = wrapper::svptrue<ScalarType>();
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
        int        x          = window_start_x;

        svbool_t pg = wrapper::svwhilelt<ScalarType>(x, window_end_x);
        do
        {
            const auto vin = svld1(pg, input_ptr + x);
            svst1(pg, output_ptr + x, elementwise_op_sve_imp<ScalarType, decltype(vin)>(pg, op, vin));
            x += wrapper::svcnt<ScalarType>();
            pg = wrapper::svwhilelt<ScalarType>(x, window_end_x);
        }
        while(svptest_any(all_true_pg, pg));
    },
    input, output);
}

template void elementwise_sve_op<float16_t>(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op);
template void elementwise_sve_op<float32_t>(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op);
template void elementwise_sve_op<int32_t>(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op);

} // namespace cpu
} // namespace arm_compute
