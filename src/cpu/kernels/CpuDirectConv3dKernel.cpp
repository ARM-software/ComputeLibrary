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
#include "src/cpu/kernels/CpuDirectConv3dKernel.h"

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
#include "src/core/CPP/Validate.h"
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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const Conv3dInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_layout() != DataLayout::NDHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);

    const DataLayout data_layout = src->data_layout();
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    // Weight layout is D, H, W, Cin, Cout
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 5);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(1) != src->dimension(channel_idx));

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->dimension(0) != weights->dimension(0),
                                        "biases size and number of output feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->num_dimensions() > 1, "biases should be one dimensional");
    }

    // Checks performed when output is configured
    if(dst->total_size() != 0)
    {
        TensorShape output_shape = misc::shape_calculator::compute_conv3d_shape(src->tensor_shape(), weights->tensor_shape(), conv_info);

        DataType data_type = src->data_type();

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON(dst->data_type() != data_type);
    }

    return Status{};
}

/** Reduce a vector to be a scalar by accumulating all lanes in the vector
 *
 * @param[in] v Vector to be reduced.
 *
 * @return the wrapped-around number.
 */
auto vreduce(const float32x4_t &v)
{
    auto v0    = wrapper::vgethigh(v);
    auto v1    = wrapper::vgetlow(v);
    auto v_out = wrapper::vadd(v0, v1);

    float a = wrapper::vgetlane(v_out, 0);
    float b = wrapper::vgetlane(v_out, 1);
    return a + b;
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
auto vreduce(const float16x8_t &v)
{
    auto v0    = wrapper::vgethigh(v);
    auto v1    = wrapper::vgetlow(v);
    auto v_out = wrapper::vadd(v0, v1);

    float16_t a = wrapper::vgetlane(v_out, 0);
    float16_t b = wrapper::vgetlane(v_out, 1);
    float16_t c = wrapper::vgetlane(v_out, 2);
    float16_t d = wrapper::vgetlane(v_out, 3);
    return a + b + c + d;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
}

template <typename T>
void CpuDirectConv3dKernel::convolve_ndhwc(const Window &window, const ITensor *src, const ITensor *weights, const ITensor *biases, ITensor *dst)
{
    using vtype                                = wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>;
    using vector_type                          = typename vtype::type;
    using tag_type                             = typename vtype::tag_type;
    constexpr int num_elems_read_per_iteration = 16 / sizeof(T);

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
    const int conv_pad_top   = _conv_info.padding.top;
    const int conv_pad_left  = _conv_info.padding.left;
    const int conv_pad_front = _conv_info.padding.front;
    const int conv_stride_w  = _conv_info.stride.width;
    const int conv_stride_h  = _conv_info.stride.height;
    const int conv_stride_d  = _conv_info.stride.depth;

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

    const T *biases_ptr = nullptr;
    if(biases)
    {
        biases_ptr = reinterpret_cast<T *>(biases->buffer() + biases->info()->offset_first_element_in_bytes());
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
            T          out_temp          = static_cast<T>(0);
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
                        vector_type out_temp_vec      = wrapper::vdup_n(static_cast<T>(0), tag_type());
                        vector_type w_vec             = wrapper::vdup_n(static_cast<T>(0), tag_type());
                        for(; index_c_in <= index_c_in_end - num_elems_read_per_iteration;
                            index_c_in += num_elems_read_per_iteration, in_ptr_mover += num_elems_read_per_iteration)
                        {
                            const auto src_vec = wrapper::vloadq(in_ptr_mover);
                            //Load Cin weights
                            for(unsigned int k = 0; k < num_elems_read_per_iteration; ++k, weights_ptr_mover += index_c_out_end)
                            {
                                w_vec = wrapper::vsetlane(*weights_ptr_mover, w_vec, k);
                            }
                            out_temp_vec = wrapper::vmla(out_temp_vec, w_vec, src_vec);
                        }
                        out_temp += vreduce(out_temp_vec);
                        for(; index_c_in < index_c_in_end; ++index_c_in, ++in_ptr_mover, weights_ptr_mover += index_c_out_end)
                        {
                            const auto src_val = *(in_ptr_mover);
                            const auto w_val   = *(weights_ptr_mover);
                            out_temp += src_val * w_val;
                        }
                    }
                }
            }
            *(reinterpret_cast<T *>(out_ptr + id_w[0])) = (biases) ? out_temp + biases_ptr[id_w[0]] : out_temp;
        },
        wei);
    },
    out);
}

void CpuDirectConv3dKernel::configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const Conv3dInfo &conv_info)
{
    ARM_COMPUTE_UNUSED(biases);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);

    _conv_info = conv_info;

    // Get convolved dimensions
    TensorShape output_shape = misc::shape_calculator::compute_conv3d_shape(src->tensor_shape(), weights->tensor_shape(), conv_info);

    DataType data_type = src->data_type();

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, output_shape, 1, data_type);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, biases, dst, conv_info));

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps());
    ICpuKernel::configure(win);
}

Status CpuDirectConv3dKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const Conv3dInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, biases, dst, conv_info));

    return Status{};
}

void CpuDirectConv3dKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src     = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto biases  = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto dst     = tensors.get_tensor(TensorType::ACL_DST);

    switch(src->info()->data_type())
    {
        case DataType::F32:
        {
            convolve_ndhwc<float>(window, src, weights, biases, dst);
            break;
        }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            convolve_ndhwc<float16_t>(window, src, weights, biases, dst);
            break;
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            break;
    }
}

const char *CpuDirectConv3dKernel::name() const
{
    return "CpuDirectConv3dKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute