/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_CONV3D_QUANTIZED_H
#define SRC_CORE_NEON_KERNELS_CONV3D_QUANTIZED_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
void directconv3d_quantized_neon_ndhwc(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, const Conv3dInfo &conv_info, const Window &window)
{
    const ITensor *src     = src0;
    const ITensor *weights = src1;
    const ITensor *biases  = src2;

    using vtype                                = wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>;
    using vector_type                          = typename vtype::type;
    using tag_type                             = typename vtype::tag_type;
    constexpr int num_elems_read_per_iteration = 16 / sizeof(T);
    using q16_t                                = typename wrapper::traits::promote_t<T>;
    using q32_t                                = typename wrapper::traits::promote_t<q16_t>;
    using q32x4_t                              = typename wrapper::traits::neon_vector<q32_t, 4>::type;

    const int32_t input_offset   = -src->info()->quantization_info().uniform().offset;
    const float   input_scale    = src->info()->quantization_info().uniform().scale;
    const int32_t weights_offset = -weights->info()->quantization_info().uniform().offset;
    const float   weights_scale  = weights->info()->quantization_info().uniform().scale;
    const int32_t output_offset  = dst->info()->quantization_info().uniform().offset;
    const float   output_scale   = dst->info()->quantization_info().uniform().scale;

    int32_t     output_multiplier = 0;
    int32_t     output_shift      = 0;
    const float multiplier        = input_scale * weights_scale / output_scale;
    arm_compute::quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);

    // Scalar quantities (N D H W Cin)
    const int element_size   = src->info()->element_size();
    const int input_stride_w = src->info()->strides_in_bytes().y() / element_size;
    const int input_stride_h = src->info()->strides_in_bytes().z() / element_size;
    const int input_stride_d = src->info()->strides_in_bytes()[3] / element_size;
    const int input_stride_n = src->info()->strides_in_bytes()[4] / element_size;
    const int input_dim_w    = src->info()->dimension(1);
    const int input_dim_h    = src->info()->dimension(2);
    const int input_dim_d    = src->info()->dimension(3);

    // Kernel info (D H W Cin Cout)
    const unsigned int kernel_stride_w = weights->info()->strides_in_bytes()[2] / element_size;
    const unsigned int kernel_stride_h = weights->info()->strides_in_bytes()[3] / element_size;
    const unsigned int kernel_stride_d = weights->info()->strides_in_bytes()[4] / element_size;
    const int          kernel_dim_w    = weights->info()->dimension(2);
    const int          kernel_dim_h    = weights->info()->dimension(3);
    const int          kernel_dim_d    = weights->info()->dimension(4);

    // Convolution padding and stride
    const int conv_pad_top   = conv_info.padding.top;
    const int conv_pad_left  = conv_info.padding.left;
    const int conv_pad_front = conv_info.padding.front;
    const int conv_stride_w  = conv_info.stride.width;
    const int conv_stride_h  = conv_info.stride.height;
    const int conv_stride_d  = conv_info.stride.depth;

    // Setup input window for the output iterator
    Window window_out = window;
    window_out.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Setup input window for the weights iterator
    Window window_w = calculate_max_window(*weights->info(), Steps());
    window_w.set(Window::DimY, Window::Dimension(0, 1, 1));
    window_w.set(Window::DimZ, Window::Dimension(0, 1, 1));
    window_w.set(Window::DimW, Window::Dimension(0, 1, 1));
    window_w.set(4, Window::Dimension(0, 1, 1));

    Iterator out(dst, window_out);
    Iterator wei(weights, window_w);

    const int32_t *biases_ptr = nullptr;
    if(biases != nullptr)
    {
        biases_ptr = reinterpret_cast<int32_t *>(biases->buffer() + biases->info()->offset_first_element_in_bytes());
    }
    execute_window_loop(window_out, [&](const Coordinates & id)
    {
        // We are computing the theoretical input starting points
        const int in_w_start_t = static_cast<int>(id.y()) * conv_stride_w - conv_pad_left;
        const int in_h_start_t = static_cast<int>(id.z()) * conv_stride_h - conv_pad_top;
        const int in_d_start_t = static_cast<int>(id[3]) * conv_stride_d - conv_pad_front;
        const int in_w_end_t   = in_w_start_t + kernel_dim_w;
        const int in_h_end_t   = in_h_start_t + kernel_dim_h;
        const int in_d_end_t   = in_d_start_t + kernel_dim_d;

        // We are computing the valid initial and ending input points by checking the borders
        const int in_w_start = std::max(in_w_start_t, 0);
        const int in_h_start = std::max(in_h_start_t, 0);
        const int in_d_start = std::max(in_d_start_t, 0);
        const int in_w_end   = std::min(in_w_end_t, input_dim_w);
        const int in_h_end   = std::min(in_h_end_t, input_dim_h);
        const int in_d_end   = std::min(in_d_end_t, input_dim_d);

        // We use the input points to select the valid weight points to use
        const int wei_w_start = in_w_start - in_w_start_t;
        const int wei_h_start = in_h_start - in_h_start_t;
        const int wei_d_start = in_d_start - in_d_start_t;
        const int wei_w_end   = kernel_dim_w - (in_w_end_t - in_w_end);
        const int wei_h_end   = kernel_dim_h - (in_h_end_t - in_h_end);
        const int wei_d_end   = kernel_dim_d - (in_d_end_t - in_d_end);

        const int      index_c_out_end = weights->info()->dimension(0);
        const int      index_c_in_end  = weights->info()->dimension(1);
        const T *const in_ptr_start    = reinterpret_cast<const T *>(src->buffer() + src->info()->offset_first_element_in_bytes()) + id[4] * input_stride_n;

        execute_window_loop(window_w, [&](const Coordinates & id_w)
        {
            /*
            * This is the loop in the weights, and it goes along OFM (output feature map)
            */
            const auto weights_ptr_start = reinterpret_cast<const T *>(wei.ptr());
            int32_t    acc               = static_cast<int32_t>(0);
            T         *out_ptr           = reinterpret_cast<T *>(out.ptr());
            for(int index_wei_d = wei_d_start, index_in_d = in_d_start; index_wei_d < wei_d_end; ++index_wei_d, ++index_in_d)
            {
                const auto in_ptr_d      = in_ptr_start + index_in_d * input_stride_d;
                const auto weights_ptr_d = weights_ptr_start + index_wei_d * kernel_stride_d;
                for(int index_wei_h = wei_h_start, index_in_h = in_h_start; index_wei_h < wei_h_end; ++index_wei_h, ++index_in_h)
                {
                    const T *const in_ptr_row      = in_ptr_d + index_in_h * input_stride_h;
                    const T *const weights_ptr_row = weights_ptr_d + index_wei_h * kernel_stride_h;
                    for(int index_wei_w = wei_w_start, index_in_w = in_w_start; index_wei_w < wei_w_end; ++index_wei_w, ++index_in_w)
                    {
                        const T    *in_ptr_mover      = in_ptr_row + index_in_w * input_stride_w;
                        const T    *weights_ptr_mover = weights_ptr_row + index_wei_w * kernel_stride_w;
                        int         index_c_in        = 0;
                        vector_type w_vec             = wrapper::vdup_n(static_cast<T>(0), tag_type());

                        q32x4_t acc_q32_0 = wrapper::vdup_n(static_cast<q32_t>(0), tag_type());
                        q32x4_t acc_q32_1 = wrapper::vdup_n(static_cast<q32_t>(0), tag_type());
                        q32x4_t acc_q32_2 = wrapper::vdup_n(static_cast<q32_t>(0), tag_type());
                        q32x4_t acc_q32_3 = wrapper::vdup_n(static_cast<q32_t>(0), tag_type());

                        for(; index_c_in <= index_c_in_end - num_elems_read_per_iteration;
                            index_c_in += num_elems_read_per_iteration, in_ptr_mover += num_elems_read_per_iteration)
                        {
                            const auto src_vec = wrapper::vloadq(in_ptr_mover);
                            //Load Cin weights
                            for(int k = 0; k < num_elems_read_per_iteration; ++k, weights_ptr_mover += index_c_out_end)
                            {
                                w_vec = wrapper::vsetlane(*weights_ptr_mover, w_vec, k);
                            }
                            q32x4_t src_q32_0 = wrapper::vdup_n(static_cast<q32_t>(input_offset), tag_type());
                            q32x4_t src_q32_1 = wrapper::vdup_n(static_cast<q32_t>(input_offset), tag_type());
                            q32x4_t src_q32_2 = wrapper::vdup_n(static_cast<q32_t>(input_offset), tag_type());
                            q32x4_t src_q32_3 = wrapper::vdup_n(static_cast<q32_t>(input_offset), tag_type());

                            q32x4_t wei_q32_0 = wrapper::vdup_n(static_cast<q32_t>(weights_offset), tag_type());
                            q32x4_t wei_q32_1 = wrapper::vdup_n(static_cast<q32_t>(weights_offset), tag_type());
                            q32x4_t wei_q32_2 = wrapper::vdup_n(static_cast<q32_t>(weights_offset), tag_type());
                            q32x4_t wei_q32_3 = wrapper::vdup_n(static_cast<q32_t>(weights_offset), tag_type());

                            const auto src_q16_0 = wrapper::vmovl(wrapper::vgetlow(src_vec));
                            const auto src_q16_1 = wrapper::vmovl(wrapper::vgethigh(src_vec));
                            const auto wei_q16_0 = wrapper::vmovl(wrapper::vgetlow(w_vec));
                            const auto wei_q16_1 = wrapper::vmovl(wrapper::vgethigh(w_vec));

                            src_q32_0 = wrapper::vadd(src_q32_0, wrapper::vmovl(wrapper::vgetlow(src_q16_0)));
                            src_q32_1 = wrapper::vadd(src_q32_1, wrapper::vmovl(wrapper::vgethigh(src_q16_0)));
                            src_q32_2 = wrapper::vadd(src_q32_2, wrapper::vmovl(wrapper::vgetlow(src_q16_1)));
                            src_q32_3 = wrapper::vadd(src_q32_3, wrapper::vmovl(wrapper::vgethigh(src_q16_1)));

                            wei_q32_0 = wrapper::vadd(wei_q32_0, wrapper::vmovl(wrapper::vgetlow(wei_q16_0)));
                            wei_q32_1 = wrapper::vadd(wei_q32_1, wrapper::vmovl(wrapper::vgethigh(wei_q16_0)));
                            wei_q32_2 = wrapper::vadd(wei_q32_2, wrapper::vmovl(wrapper::vgetlow(wei_q16_1)));
                            wei_q32_3 = wrapper::vadd(wei_q32_3, wrapper::vmovl(wrapper::vgethigh(wei_q16_1)));

                            acc_q32_0 = wrapper::vmla(acc_q32_0, wei_q32_0, src_q32_0);
                            acc_q32_1 = wrapper::vmla(acc_q32_1, wei_q32_1, src_q32_1);
                            acc_q32_2 = wrapper::vmla(acc_q32_2, wei_q32_2, src_q32_2);
                            acc_q32_3 = wrapper::vmla(acc_q32_3, wei_q32_3, src_q32_3);
                        }
#if defined(__aarch64__)
                        acc += wrapper::vaddv(acc_q32_0);
                        acc += wrapper::vaddv(acc_q32_1);
                        acc += wrapper::vaddv(acc_q32_2);
                        acc += wrapper::vaddv(acc_q32_3);
#else // __aarch64__
                        auto temp = wrapper::vpadd(wrapper::vgethigh(acc_q32_0), wrapper::vgetlow(acc_q32_0));
                        temp      = wrapper::vpadd(temp, temp);
                        acc       += wrapper::vgetlane(temp, 0);

                        temp      = wrapper::vpadd(wrapper::vgethigh(acc_q32_1), wrapper::vgetlow(acc_q32_1));
                        temp      = wrapper::vpadd(temp, temp);
                        acc       += wrapper::vgetlane(temp, 0);

                        temp      = wrapper::vpadd(wrapper::vgethigh(acc_q32_2), wrapper::vgetlow(acc_q32_2));
                        temp      = wrapper::vpadd(temp, temp);
                        acc       += wrapper::vgetlane(temp, 0);

                        temp      = wrapper::vpadd(wrapper::vgethigh(acc_q32_3), wrapper::vgetlow(acc_q32_3));
                        temp      = wrapper::vpadd(temp, temp);
                        acc       += wrapper::vgetlane(temp, 0);

#endif // __aarch64__

                        for(; index_c_in < index_c_in_end; ++index_c_in, ++in_ptr_mover, weights_ptr_mover += index_c_out_end)
                        {
                            const auto src_val = *(in_ptr_mover) + input_offset;
                            const auto w_val   = *(weights_ptr_mover) + weights_offset;
                            acc += src_val * w_val;
                        }
                    }
                }
            }

            if(biases)
            {
                acc += *reinterpret_cast<const int32_t *>(biases_ptr + id_w[0]);
            }

            T out_val                                   = finalize_quantization(acc, output_multiplier, output_shift, output_offset, T(0), T(0), false);
            *(reinterpret_cast<T *>(out_ptr + id_w[0])) = out_val;
        },
        wei);
    },
    out);
}
} // namespace cpu
} // namespace arm_compute
#endif // SRC_CORE_NEON_KERNELS_CONV3D_QUANTIZED_H