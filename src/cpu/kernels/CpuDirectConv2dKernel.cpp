/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#include "src/cpu/kernels/CpuDirectConv2dKernel.h"

#include "src/core/NEON/kernels/detail/NEDirectConvolutionDetail.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <algorithm>

using namespace arm_compute::detail;

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);

    const DataLayout data_layout = src->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(channel_idx) != src->dimension(channel_idx));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(width_idx) != weights->dimension(height_idx));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(data_layout == DataLayout::NHWC && src->data_type() != DataType::F32);
    ARM_COMPUTE_UNUSED(width_idx);
    // Checks performed when output is configured
    if(dst->total_size() != 0)
    {
        TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info);

        DataType data_type = src->data_type();

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON(dst->data_type() != data_type);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON(src->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_UNUSED(src);

    Window win{};
    bool   window_changed = false;

    // Configure window without any padding
    win = calculate_max_window(*dst, Steps());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

bool have_zero_x_internal_padding(ITensorInfo *src, const ITensorInfo *weights)
{
    return (src->padding().left == 0 && weights->padding().left == 0 && src->padding().right == 0 && weights->padding().right == 0);
}

} // namespace

template <typename T>
void CpuDirectConv2dKernel::convolve_nhwc_optimized(const Window &window, const ITensor *src, const ITensor *weights, ITensor *dst)
{
    // This function assumes that input and weights have not padding in channel

    // Declare useful types
    using vtype       = wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>;
    using vector_type = typename vtype::type;
    using tag_type    = typename vtype::tag_type;

    // Scalar quantities
    const int element_size   = src->info()->element_size();
    const int input_stride_w = src->info()->strides_in_bytes().y() / element_size;
    const int input_stride_h = src->info()->strides_in_bytes().z() / element_size;
    const int input_stride_n = src->info()->strides_in_bytes()[3] / element_size;
    const int input_dim_w    = src->info()->dimension(1);
    const int input_dim_h    = src->info()->dimension(2);

    const int output_stride_c = dst->info()->strides_in_bytes().x();

    const unsigned int kernel_stride_w = weights->info()->strides_in_bytes().y() / element_size;
    const unsigned int kernel_stride_h = weights->info()->strides_in_bytes().z() / element_size;
    const int          kernel_dim_w    = weights->info()->dimension(1);
    const int          kernel_dim_h    = weights->info()->dimension(2);

    const int conv_pad_top  = _conv_info.pad_top();
    const int conv_pad_left = _conv_info.pad_left();
    const int conv_stride_w = std::get<0>(_conv_info.stride());
    const int conv_stride_h = std::get<1>(_conv_info.stride());

    // Setup input window for the output iterator
    Window window_out = window;
    window_out.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Setup input window for the weights iterator
    Window window_w = calculate_max_window(*weights->info(), Steps());
    window_w.set(Window::DimX, Window::Dimension(0, 1, 1));
    window_w.set(Window::DimY, Window::Dimension(0, 1, 1));
    window_w.set(Window::DimZ, Window::Dimension(0, 1, 1));

    Iterator out(dst, window_out);
    Iterator wei(weights, window_w);

    constexpr int num_elems_read_per_iteration = 16 / sizeof(T);
    /*
     * This implementation parallelize the full WC plane of input and weights by
     * treating them as series of elements. So for example, a 3x3 weights and
     * floating point vector operations of 4 elements per time, the first 3
     * channel elements of the first row would be taken and additionally the first
     * element of the second row. The 9 elements in each single WC weight plane
     * would require 2 4-element vector operations and a last single element operation.
     *
     * This works since when we create the input vector to multiply with the weights,
     * the exact required elements are loaded in the same order. Therefore the
     * multiplication works on the correct input/weight elements.
     */
    execute_window_loop(window_out, [&](const Coordinates & id)
    {
        /*
         * In here we create theoretical indexes which then we validate for both
         * inputs and weights.
         * As a reminder, this loop take each output point in NHW, C is treated
         * in the weights loop.
         */
        // We are computing the theoretical starting input starting points
        const int in_w_start_t = static_cast<int>(id.y()) * conv_stride_w - conv_pad_left;
        const int in_h_start_t = static_cast<int>(id.z()) * conv_stride_h - conv_pad_top;
        const int in_w_end_t   = in_w_start_t + kernel_dim_w;
        const int in_h_end_t   = in_h_start_t + kernel_dim_h;

        // We are computing the valid initial and ending input points by checking the borders
        const int in_w_start = std::max(in_w_start_t, 0);
        const int in_h_start = std::max(in_h_start_t, 0);
        const int in_w_end   = std::min(in_w_end_t, input_dim_w);
        const int in_h_end   = std::min(in_h_end_t, input_dim_h);

        // We use the input points to select the valid weight points to use
        const int index_wc_start = (in_w_start - in_w_start_t) * kernel_stride_w;
        const int index_h_start  = in_h_start - in_h_start_t;
        const int index_wc_end   = (kernel_dim_w - (in_w_end_t - in_w_end)) * kernel_stride_w;
        const int index_h_end    = kernel_dim_h - (in_h_end_t - in_h_end);

        execute_window_loop(window_w, [&](const Coordinates & id_w)
        {
            /*
             * This is the loop in the weights, and it goes along N (the batches)
             * As a reminder, the batches of the weights are translated into the
             * channels of the output
             */
            const T *in_ptr_row = reinterpret_cast<const T *>(src->buffer() + src->info()->offset_first_element_in_bytes())
                                  + id[3] * input_stride_n + in_w_start * input_stride_w + in_h_start * input_stride_h;
            const T *weights_ptr_row = reinterpret_cast<const T *>(wei.ptr()) + index_h_start * kernel_stride_h;
            uint8_t *out_ptr         = out.ptr() + id_w[3] * output_stride_c;

            T out_temp = static_cast<T>(0);
            for(int index_h = index_h_start; index_h < index_h_end; ++index_h, in_ptr_row += input_stride_h, weights_ptr_row += kernel_stride_h)
            {
                const T    *in_ptr_mover = in_ptr_row;
                int         index_wc     = index_wc_start;
                vector_type out_temp_vec = wrapper::vdup_n(static_cast<T>(0), tag_type());
                for(; index_wc <= index_wc_end - num_elems_read_per_iteration; index_wc += num_elems_read_per_iteration, in_ptr_mover += num_elems_read_per_iteration)
                {
                    const auto src_vec = wrapper::vloadq(in_ptr_mover);
                    const auto w_vec   = wrapper::vloadq(weights_ptr_row + index_wc);
                    out_temp_vec       = wrapper::vmla(out_temp_vec, w_vec, src_vec);
                }
                out_temp += vreduce(out_temp_vec);
                for(; index_wc < index_wc_end; ++index_wc, ++in_ptr_mover)
                {
                    const auto src_val = *(in_ptr_mover);
                    const auto w_val   = *(weights_ptr_row + index_wc);
                    out_temp += src_val * w_val;
                }
            }
            *(reinterpret_cast<T *>(out_ptr)) = out_temp;
        },
        wei);
    },
    out);
}

template <typename T>
void CpuDirectConv2dKernel::convolve_nhwc(const Window &window, const ITensor *src, const ITensor *weights, ITensor *dst)
{
    // Declare useful types
    using vtype       = wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>;
    using vector_type = typename vtype::type;
    using tag_type    = typename vtype::tag_type;

    // Scalar quantities
    const int element_size   = src->info()->element_size();
    const int input_stride_w = src->info()->strides_in_bytes().y() / element_size;
    const int input_stride_h = src->info()->strides_in_bytes().z() / element_size;
    const int input_stride_n = src->info()->strides_in_bytes()[3] / element_size;
    const int input_dim_w    = src->info()->dimension(1);
    const int input_dim_h    = src->info()->dimension(2);

    const int output_stride_c = dst->info()->strides_in_bytes().x();

    const unsigned int kernel_stride_w = weights->info()->strides_in_bytes().y() / element_size;
    const unsigned int kernel_stride_h = weights->info()->strides_in_bytes().z() / element_size;
    const int          kernel_dim_w    = weights->info()->dimension(1);
    const int          kernel_dim_h    = weights->info()->dimension(2);

    const int conv_pad_top  = _conv_info.pad_top();
    const int conv_pad_left = _conv_info.pad_left();
    const int conv_stride_w = std::get<0>(_conv_info.stride());
    const int conv_stride_h = std::get<1>(_conv_info.stride());

    // Setup input window for the output iterator
    Window window_out = window;
    window_out.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Setup input window for the weights iterator
    Window window_w = calculate_max_window(*weights->info(), Steps());
    window_w.set(Window::DimX, Window::Dimension(0, 1, 1));
    window_w.set(Window::DimY, Window::Dimension(0, 1, 1));
    window_w.set(Window::DimZ, Window::Dimension(0, 1, 1));

    Iterator out(dst, window_out);
    Iterator wei(weights, window_w);

    constexpr int num_elems_read_per_iteration = 16 / sizeof(T);

    execute_window_loop(window_out, [&](const Coordinates & id)
    {
        // We are computing the theoretical starting input starting points
        const int in_w_start_t = static_cast<int>(id.y()) * conv_stride_w - conv_pad_left;
        const int in_h_start_t = static_cast<int>(id.z()) * conv_stride_h - conv_pad_top;
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
        const int wei_w_end   = kernel_dim_w - (in_w_end_t - in_w_end);
        const int wei_h_end   = kernel_dim_h - (in_h_end_t - in_h_end);

        const int      index_c_end  = weights->info()->dimension(0);
        const T *const in_ptr_start = reinterpret_cast<const T *>(src->buffer() + src->info()->offset_first_element_in_bytes()) + id[3] * input_stride_n;

        execute_window_loop(window_w, [&](const Coordinates & id_w)
        {
            const T *const weights_ptr_start = reinterpret_cast<const T *>(wei.ptr());
            uint8_t       *out_ptr           = out.ptr() + id_w[3] * output_stride_c;

            T out_temp = static_cast<T>(0);
            for(int index_wei_h = wei_h_start, index_in_h = in_h_start; index_wei_h < wei_h_end; ++index_wei_h, ++index_in_h)
            {
                const T *const in_ptr_row      = in_ptr_start + index_in_h * input_stride_h;
                const T *const weights_ptr_row = weights_ptr_start + index_wei_h * kernel_stride_h;
                for(int index_wei_w = wei_w_start, index_in_w = in_w_start; index_wei_w < wei_w_end; ++index_wei_w, ++index_in_w)
                {
                    const T    *in_ptr_mover      = in_ptr_row + index_in_w * input_stride_w;
                    const T    *weights_ptr_mover = weights_ptr_row + index_wei_w * kernel_stride_w;
                    int         index_c           = 0;
                    vector_type out_temp_vec      = wrapper::vdup_n(static_cast<T>(0), tag_type());
                    for(; index_c <= index_c_end - num_elems_read_per_iteration; index_c += num_elems_read_per_iteration, in_ptr_mover += num_elems_read_per_iteration, weights_ptr_mover += num_elems_read_per_iteration)
                    {
                        const auto src_vec = wrapper::vloadq(in_ptr_mover);
                        const auto w_vec   = wrapper::vloadq(weights_ptr_mover);
                        out_temp_vec       = wrapper::vmla(out_temp_vec, w_vec, src_vec);
                    }
                    out_temp += vreduce(out_temp_vec);
                    for(; index_c < index_c_end; ++index_c, ++in_ptr_mover, ++weights_ptr_mover)
                    {
                        const auto src_val = *(in_ptr_mover);
                        const auto w_val   = *(weights_ptr_mover);
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

template <typename T>
void CpuDirectConv2dKernel::convolve_nchw(const Window &window, const ITensor *src, const ITensor *weights, ITensor *dst)
{
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

    const int conv_pad_top  = _conv_info.pad_top();
    const int conv_pad_left = _conv_info.pad_left();
    const int conv_stride_w = std::get<0>(_conv_info.stride());
    const int conv_stride_h = std::get<1>(_conv_info.stride());

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

    execute_window_loop(window_out, [&](const Coordinates & id)
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

        const int      index_c_end  = weights->info()->dimension(2);
        const T *const in_ptr_start = reinterpret_cast<const T *>(src->buffer() + src->info()->offset_first_element_in_bytes()) + id[3] * input_stride_n;
        execute_window_loop(window_w, [&](const Coordinates & id_w)
        {
            const T *const weights_ptr_start = reinterpret_cast<const T *>(wei.ptr());
            uint8_t       *out_ptr           = out.ptr() + id_w[3] * output_stride_c;
            T              out_temp          = static_cast<T>(0);

            for(int index_wei_c = 0, index_in_c = 0; index_wei_c < index_c_end; ++index_wei_c, ++index_in_c)
            {
                const T *const in_ptr_row_0      = in_ptr_start + index_in_c * input_stride_c;
                const T *const weights_ptr_row_0 = weights_ptr_start + index_wei_c * kernel_stride_c;
                for(int index_wei_h = wei_h_start, index_in_h = in_h_start; index_wei_h < wei_h_end; ++index_wei_h, ++index_in_h)
                {
                    const T    *in_ptr_row      = in_ptr_row_0 + index_in_h * input_stride_h;
                    const T    *weights_ptr_row = weights_ptr_row_0 + index_wei_h * kernel_stride_h;
                    int         index_w         = in_w_start;
                    int         index_wei_w     = wei_w_start;
                    vector_type out_temp_vec    = wrapper::vdup_n(static_cast<T>(0), tag_type());
                    for(; index_w <= ((in_w_end - num_elems_read_per_iteration)); index_w += num_elems_read_per_iteration, index_wei_w += num_elems_read_per_iteration)
                    {
                        const auto src_vec = wrapper::vloadq(in_ptr_row + index_w * input_stride_w);
                        const auto w_vec   = wrapper::vloadq(weights_ptr_row + index_wei_w * kernel_stride_w);
                        out_temp_vec       = wrapper::vmla(out_temp_vec, w_vec, src_vec);
                    }
                    out_temp += vreduce(out_temp_vec);
                    for(; index_w < in_w_end; ++index_w, ++index_wei_w)
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

void CpuDirectConv2dKernel::configure(ITensorInfo *src, ITensorInfo *weights, ITensorInfo *dst, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);

    _conv_info   = conv_info;
    _data_layout = src->data_layout();
    _kernel_size = weights->dimension(get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH));

    // Get convolved dimensions
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info);

    DataType data_type = src->data_type();

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, output_shape, 1, data_type);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, dst, conv_info));

    // Configure kernel window
    auto win_config = validate_and_configure_window(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuDirectConv2dKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, dst, conv_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(),
                                                              dst->clone().get())
                                .first);

    return Status{};
}

void CpuDirectConv2dKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src     = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto dst     = tensors.get_tensor(TensorType::ACL_DST);

    if(_data_layout == DataLayout::NCHW)
    {
        switch(src->info()->data_type())
        {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
            {
                convolve_nchw<float16_t>(window, src, weights, dst);
                break;
            }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::F32:
            {
                convolve_nchw<float>(window, src, weights, dst);
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
        }
    }
    else
    {
        switch(src->info()->data_type())
        {
            case DataType::F32:
            {
                if(have_zero_x_internal_padding(src->info(), weights->info()))
                {
                    convolve_nhwc_optimized<float>(window, src, weights, dst);
                }
                else
                {
                    convolve_nhwc<float>(window, src, weights, dst);
                }
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
        }
    }
}
const char *CpuDirectConv2dKernel::name() const
{
    return "CpuDirectConvolutionLayerKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
