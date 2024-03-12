/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_DIRECTCONV2D_NCHW_IMPL_H
#define ACL_SRC_CPU_KERNELS_DIRECTCONV2D_NCHW_IMPL_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/kernels/detail/NEDirectConvolutionDetail.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include <algorithm>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
template <typename T>
void convolve_nchw(
    const Window &window, const ITensor *src, const ITensor *weights, ITensor *dst, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_UNUSED(conv_info);

    // Declare useful types
    using vtype       = wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>;
    using vector_type = typename vtype::type;
    using tag_type    = typename vtype::tag_type;

    // Scalar quantities
    const int element_size   = src->info()->element_size();
    const int input_stride_w = src->info()->strides_in_bytes()[0] / element_size;
    const int input_stride_h = src->info()->strides_in_bytes()[1] / element_size;
    const int input_stride_c = src->info()->strides_in_bytes()[2] / element_size;
    const int input_stride_n = src->info()->strides_in_bytes()[3] / element_size;

    const int input_dim_w = src->info()->dimension(0);
    const int input_dim_h = src->info()->dimension(1);

    const int output_stride_c = dst->info()->strides_in_bytes()[2];

    const unsigned int kernel_stride_w = weights->info()->strides_in_bytes().x() / element_size;
    const unsigned int kernel_stride_h = weights->info()->strides_in_bytes().y() / element_size;
    const unsigned int kernel_stride_c = weights->info()->strides_in_bytes().z() / element_size;

    const int kernel_dim_w = weights->info()->dimension(0);
    const int kernel_dim_h = weights->info()->dimension(1);

    const int conv_pad_top  = conv_info.pad_top();
    const int conv_pad_left = conv_info.pad_left();
    const int conv_stride_w = std::get<0>(conv_info.stride());
    const int conv_stride_h = std::get<1>(conv_info.stride());

    // Setup input window for the output iterator
    Window window_out = window;
    window_out.set(Window::DimZ, Window::Dimension(0, 1, 1));

    // Setup input window for the weights iterator
    Window window_w = calculate_max_window(*weights->info(), Steps());
    window_w.set(Window::DimX, Window::Dimension(0, 1, 1));
    window_w.set(Window::DimY, Window::Dimension(0, 1, 1));
    window_w.set(Window::DimZ, Window::Dimension(0, 1, 1));

    Iterator out(dst, window_out);
    Iterator wei(weights, window_w);

    constexpr int num_elems_read_per_iteration = 16 / sizeof(T);

    execute_window_loop(
        window_out,
        [&](const Coordinates &id)
        {
            // We are computing the theoretical starting input starting points
            const int in_w_start_t = static_cast<int>(id.x()) * conv_stride_w - conv_pad_left;
            const int in_h_start_t = static_cast<int>(id.y()) * conv_stride_h - conv_pad_top;
            const int in_w_end_t   = in_w_start_t + kernel_dim_w;
            const int in_h_end_t   = in_h_start_t + kernel_dim_h;

            // We are computing the valid initial and ending input points by checking the borders
            const int in_w_start = std::max(in_w_start_t, 0);
            const int in_h_start = std::max(in_h_start_t, 0);
            const int in_w_end   = std::min(in_w_end_t, input_dim_w);
            const int in_h_end   = std::min(in_h_end_t, input_dim_h);

            // We use the input points to select the valid weight points to use
            const int wei_w_start = in_w_start - in_w_start_t;
            const int wei_h_start = in_h_start - in_h_start_t;
            const int wei_h_end   = kernel_dim_h - (in_h_end_t - in_h_end);

            const int      index_c_end = weights->info()->dimension(2);
            const T *const in_ptr_start =
                reinterpret_cast<const T *>(src->buffer() + src->info()->offset_first_element_in_bytes()) +
                id[3] * input_stride_n;
            execute_window_loop(
                window_w,
                [&](const Coordinates &id_w)
                {
                    const T *const weights_ptr_start = reinterpret_cast<const T *>(wei.ptr());
                    uint8_t       *out_ptr           = out.ptr() + id_w[3] * output_stride_c;
                    T              out_temp          = static_cast<T>(0);

                    for (int index_wei_c = 0, index_in_c = 0; index_wei_c < index_c_end; ++index_wei_c, ++index_in_c)
                    {
                        const T *const in_ptr_row_0      = in_ptr_start + index_in_c * input_stride_c;
                        const T *const weights_ptr_row_0 = weights_ptr_start + index_wei_c * kernel_stride_c;
                        for (int index_wei_h = wei_h_start, index_in_h = in_h_start; index_wei_h < wei_h_end;
                             ++index_wei_h, ++index_in_h)
                        {
                            const T    *in_ptr_row      = in_ptr_row_0 + index_in_h * input_stride_h;
                            const T    *weights_ptr_row = weights_ptr_row_0 + index_wei_h * kernel_stride_h;
                            int         index_w         = in_w_start;
                            int         index_wei_w     = wei_w_start;
                            vector_type out_temp_vec    = wrapper::vdup_n(static_cast<T>(0), tag_type());
                            for (; index_w <= ((in_w_end - num_elems_read_per_iteration));
                                 index_w += num_elems_read_per_iteration, index_wei_w += num_elems_read_per_iteration)
                            {
                                const auto src_vec = wrapper::vloadq(in_ptr_row + index_w * input_stride_w);
                                const auto w_vec   = wrapper::vloadq(weights_ptr_row + index_wei_w * kernel_stride_w);
                                out_temp_vec       = wrapper::vmla(out_temp_vec, w_vec, src_vec);
                            }
                            out_temp += vreduce(out_temp_vec);
                            for (; index_w < in_w_end; ++index_w, ++index_wei_w)
                            {
                                const auto src_val = *(in_ptr_row + index_w * input_stride_w);
                                const auto w_val   = *(weights_ptr_row + index_wei_w * kernel_stride_w);
                                out_temp += src_val * w_val;
                            }
                        }
                    }
                    *(reinterpret_cast<T *>(out_ptr)) = out_temp;
                },
                wei);
        },
        out);
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_DIRECTCONV2D_NCHW_IMPL_H
