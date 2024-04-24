/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_NORM_LAYER_GENERIC_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_NORM_LAYER_GENERIC_NEON_IMPL_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/helpers/NormalizationHelpers.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
/** Function to perform normalization depending on the given template
 *  dimension. The second template parameter specifies whether the
 *  normalization has to be 1D or 2D.
 *
 * @note Only supported normalizations are:
 *  - 1D over X or Z
 *  - 2D over X and Y
 *
 * @param[in] window     Region on which to execute the kernel.
 * @param[in] in         Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
 *                       and an optional 4th dimension for batch of inputs. Data types supported: FP16/F32. Data layouts supported: NCHW/NHWC.
 * @param[in] in_squared Source with each element has been squared. 3 lower dims represent a single input with dimensions [width, height, IFM],
 *                       Data type and layout supported: same as @p input.
 * @param[in] out        Destination tensor. Output will have the same number of dimensions as input. Data type and layout supported: same as @p input.
 * @param[in] ninfo      Normalization layer information like the normalization type, normalization size and other parameters.
 */
template <typename T, unsigned int S, unsigned int dim, bool do_2D_norm>
void normalize_float(
    const Window &window, const ITensor *in, const ITensor *in_squared, ITensor *out, NormalizationLayerInfo ninfo)
{
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = S;

    Iterator input(in, win);
    Iterator input_squared(in_squared, win);
    Iterator output(out, win);

    const int dim_y                      = in->info()->data_layout() == DataLayout::NCHW ? 1 : 2;
    const int radius                     = ninfo.norm_size() / 2;
    const int input_squared_stride_x     = in_squared->info()->strides_in_bytes()[0];
    const int input_squared_stride_slice = in_squared->info()->strides_in_bytes()[dim];
    const int input_squared_stride_row   = in_squared->info()->strides_in_bytes()[dim_y];

    const int max_right  = in->info()->dimension(dim) - 1;
    const int max_bottom = in->info()->dimension(dim_y) - 1;

    const auto coeff_vec = wrapper::vdup_n(static_cast<T>(ninfo.scale_coeff()), ExactTagType{});
    const auto beta_vec  = wrapper::vdup_n(static_cast<T>(ninfo.beta()), ExactTagType{});
    const auto kappa_vec = wrapper::vdup_n(static_cast<T>(ninfo.kappa()), ExactTagType{});

    auto sequential_normalization = [&](const int x, const Coordinates &id, const int current_row, const int first_row,
                                        const int last_row, const T *input_ptr, const uint8_t *input_squared_start_ptr,
                                        T *output_ptr)
    {
        const int current_slice = dim == 0 ? x : id[dim];
        const int first_slice   = std::max(current_slice - radius, 0);
        const int last_slice    = std::min(current_slice + radius, max_right);

        const uint8_t *const input_squared_x_ptr = input_squared_start_ptr + x * input_squared_stride_x;
        // Accumulate 2D In-Map values
        auto accu = static_cast<T>(0.f);
        for (int j = first_row; j <= last_row; ++j)
        {
            // Compute row displacement
            const uint8_t *const input_squared_ptr = input_squared_x_ptr + (j - current_row) * input_squared_stride_row;
            for (int i = first_slice; i <= last_slice; ++i)
            {
                accu +=
                    *reinterpret_cast<const T *>(input_squared_ptr + (i - current_slice) * input_squared_stride_slice);
            }
        }

        // Normalize
        const auto normalized =
            std::pow(accu * static_cast<T>(ninfo.scale_coeff()) + static_cast<T>(ninfo.kappa()), ninfo.beta());
        const auto normalized_pixel = (*(input_ptr + x)) / normalized;
        *(output_ptr + x)           = normalized_pixel;
    };

    execute_window_loop(
        win,
        [&](const Coordinates &id)
        {
            const auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
            auto       output_ptr = reinterpret_cast<T *>(output.ptr());

            // Get range to normalize
            const int current_row = do_2D_norm ? id[dim_y] : 0;
            const int first_row   = do_2D_norm ? std::max(current_row - radius, 0) : 0;
            const int last_row    = do_2D_norm ? std::min(current_row + radius, max_bottom) : 0;

            int x = window_start_x;
            // Compute serially starting elements for the case x dimension is width
            for (; x < radius && x < window_end_x && dim == 0; ++x)
            {
                sequential_normalization(x, id, current_row, first_row, last_row, input_ptr, input_squared.ptr(),
                                         output_ptr);
            }

            // Compute vectorized
            for (; x <= window_end_x - window_step_x - radius; x += window_step_x)
            {
                const int current_slice = dim == 0 ? x : id[dim];
                const int first_slice   = std::max(current_slice - radius, 0);
                const int last_slice    = std::min(current_slice + radius, max_right);

                const uint8_t *const input_squared_x_ptr = input_squared.ptr() + x * input_squared_stride_x;
                // Accumulate 2D In-Map values
                auto accu = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
                for (int j = first_row; j <= last_row; ++j)
                {
                    // Compute row displacement
                    const uint8_t *const input_squared_ptr =
                        input_squared_x_ptr + (j - current_row) * input_squared_stride_row;
                    for (int i = first_slice; i <= last_slice; ++i)
                    {
                        accu = wrapper::vadd(
                            accu, wrapper::vloadq(reinterpret_cast<const T *>(
                                      input_squared_ptr + (i - current_slice) * input_squared_stride_slice)));
                    }
                }

                // Normalize
                const auto normalized       = wrapper::vpow(wrapper::vmla(kappa_vec, coeff_vec, accu), beta_vec);
                const auto normalized_pixel = wrapper::vmul(wrapper::vloadq(input_ptr + x), wrapper::vinv(normalized));
                wrapper::vstore(reinterpret_cast<T *>(output_ptr + x), normalized_pixel);
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                sequential_normalization(x, id, current_row, first_row, last_row, input_ptr, input_squared.ptr(),
                                         output_ptr);
            }
        },
        input, input_squared, output);
}

} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_NORM_LAYER_GENERIC_NEON_IMPL_H
