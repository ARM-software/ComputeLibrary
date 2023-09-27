/*
 * Copyright (c) 2019-2023 Arm Limited.
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
#include "src/cpu/kernels/instancenorm/generic/neon/impl.h"

#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
class ITensor;
class Window;
namespace cpu
{
template <typename InputType, typename AccType>
void vector_float_sum(AccType &result, AccType &result_square, const InputType &inputs)
{
    result        = wrapper::vadd(result, inputs);
    result_square = wrapper::vadd(result_square, wrapper::vmul(inputs, inputs));
}

template <typename InputType, typename AccType>
InputType
vector_float_norm(const InputType &inputs, const AccType &vec_mean, const AccType &vec_multip, const AccType &vec_beta)
{
    return wrapper::vadd(wrapper::vmul(wrapper::vsub(inputs, vec_mean), vec_multip), vec_beta);
}

template <typename T, typename AccType>
void instance_normalization_nchw(
    ITensor *input, ITensor *output, float gamma, float beta, float epsilon, const Window &window)
{
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    // Clear X/Y dimensions on execution window as we handle the planes manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(Window::DimY, Window::Dimension(0, 1, 1));

    constexpr int      window_step_x  = 16 / sizeof(T);
    const unsigned int elements_plane = input->info()->dimension(0) * output->info()->dimension(1);

    Iterator input_it(input, win);
    execute_window_loop(
        win,
        [&](const Coordinates &id)
        {
            Window win_plane = window;
            win_plane.set(Window::DimX, Window::Dimension(0, 1, 1));
            win_plane.set(Window::DimZ, Window::Dimension(id[2], id[2] + 1, 1));
            win_plane.set(3, Window::Dimension(id[3], id[3] + 1, 1));

            Iterator input_plane_it(input, win_plane);
            Iterator output_plane_it(output, win_plane);

            auto sum_h_w         = static_cast<AccType>(0.f);
            auto sum_squares_h_w = static_cast<AccType>(0.f);

            execute_window_loop(
                win_plane,
                [&](const Coordinates &)
                {
                    const auto input_ptr = reinterpret_cast<const T *>(input_plane_it.ptr());

                    auto vec_sum_h_w         = wrapper::vdup_n(static_cast<AccType>(0.f), ExactTagType{});
                    auto vec_sum_squares_h_w = wrapper::vdup_n(static_cast<AccType>(0.f), ExactTagType{});

                    // Compute S elements per iteration
                    int x = window.x().start();
                    for (; x <= (window.x().end() - window_step_x); x += window_step_x)
                    {
                        auto vec_input_val = wrapper::vloadq(input_ptr + x);
                        vector_float_sum(vec_sum_h_w, vec_sum_squares_h_w, vec_input_val);
                    }

                    auto vec2_sum_h_w = wrapper::vpadd(wrapper::vgethigh(vec_sum_h_w), wrapper::vgetlow(vec_sum_h_w));
                    auto vec2_sum_squares_h_w =
                        wrapper::vpadd(wrapper::vgethigh(vec_sum_squares_h_w), wrapper::vgetlow(vec_sum_squares_h_w));

                    vec2_sum_h_w         = wrapper::vpadd(vec2_sum_h_w, vec2_sum_h_w);
                    vec2_sum_squares_h_w = wrapper::vpadd(vec2_sum_squares_h_w, vec2_sum_squares_h_w);

                    sum_h_w += wrapper::vgetlane(vec2_sum_h_w, 0);
                    sum_squares_h_w += wrapper::vgetlane(vec2_sum_squares_h_w, 0);

                    // Compute left-over elements
                    for (; x < window.x().end(); ++x)
                    {
                        const auto value = static_cast<AccType>(*(input_ptr + x));
                        sum_h_w += value;
                        sum_squares_h_w += value * value;
                    }
                },
                input_plane_it, output_plane_it);

            const auto mean_h_w = sum_h_w / elements_plane;
            const auto var_h_w  = sum_squares_h_w / elements_plane - mean_h_w * mean_h_w;

            const auto multip_h_w     = gamma / std::sqrt(var_h_w + epsilon);
            const auto vec_mean_h_w   = wrapper::vdup_n(static_cast<AccType>(mean_h_w), ExactTagType{});
            const auto vec_multip_h_w = wrapper::vdup_n(static_cast<AccType>(multip_h_w), ExactTagType{});
            const auto vec_beta       = wrapper::vdup_n(static_cast<AccType>(beta), ExactTagType{});

            execute_window_loop(
                win_plane,
                [&](const Coordinates &)
                {
                    auto input_ptr  = reinterpret_cast<T *>(input_plane_it.ptr());
                    auto output_ptr = reinterpret_cast<T *>(output_plane_it.ptr());

                    // Compute S elements per iteration
                    int x = window.x().start();
                    //auto vec_val = wrapper::vdup_n(static_cast<T>(0.0f), ExactTagType{});
                    for (; x <= (window.x().end() - window_step_x); x += window_step_x)
                    {
                        const auto vec_val        = wrapper::vloadq(input_ptr + x);
                        const auto normalized_vec = vector_float_norm(vec_val, vec_mean_h_w, vec_multip_h_w, vec_beta);
                        wrapper::vstore(output_ptr + x, normalized_vec);
                    }

                    // Compute left-over elements
                    for (; x < window.x().end(); ++x)
                    {
                        const auto val    = static_cast<AccType>(*(input_ptr + x));
                        *(output_ptr + x) = static_cast<T>((val - mean_h_w) * multip_h_w + beta);
                    }
                },
                input_plane_it, output_plane_it);
        },
        input_it);
}

template void instance_normalization_nchw<float>(
    ITensor *input, ITensor *output, float gamma, float beta, float epsilon, const Window &window);
} // namespace cpu
} // namespace arm_compute
