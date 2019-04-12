/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEDepthwiseConvolutionLayer3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/detail/NEDirectConvolutionDetail.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace
{
template <typename T1, typename T2, unsigned int stridex>
class convolver_3x3
{
public:
    static void convolve(const Window &window, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation)
    {
        const int input_offset   = -input->info()->quantization_info().offset;
        const int weights_offset = -weights->info()->quantization_info().offset;

        const int          input_stride_x  = input->info()->strides_in_bytes().x();
        const int          input_stride_y  = input->info()->strides_in_bytes().y();
        const int          input_stride_z  = input->info()->strides_in_bytes().z();
        const int          input_stride_w  = input->info()->strides_in_bytes()[3];
        const int          output_stride_y = output->info()->strides_in_bytes().y();
        const int          kernel_stride_y = weights->info()->strides_in_bytes().y();
        const int          kernel_stride_z = weights->info()->strides_in_bytes().z();
        const int          output_w        = output->info()->dimension(0);
        const int          output_h        = output->info()->dimension(1);
        const int          delta_input     = detail::get_input_num_elems_processed<stridex>(num_elems_written_per_iteration);
        const unsigned int conv_stride_y   = std::get<1>(conv_info.stride());
        const unsigned int conv_pad_x      = conv_info.pad_left();
        const unsigned int conv_pad_y      = conv_info.pad_top();

        // setup output window for the iterator
        Window window_out = window;
        window_out.set(Window::DimX, Window::Dimension(0, output->info()->dimension(Window::DimX), output->info()->dimension(Window::DimX)));
        window_out.set(Window::DimY, Window::Dimension(0, output->info()->dimension(Window::DimY), output->info()->dimension(Window::DimY)));

        // setup input window for the iterator
        Window window_in = window;
        // Iteration of input is taken care of in execute_window_loop
        window_in.set(Window::DimX, Window::Dimension(0, 0, 0));
        window_in.set(Window::DimY, Window::Dimension(0, 0, 0));
        window_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Window window_k = calculate_max_window(*weights->info(), Steps(1u));

        Iterator in(input, window_in);
        Iterator out(output, window_out);
        Iterator w(weights, window_k);

        const uint8_t *weights_ptr = w.ptr();

        execute_window_loop(window_out, [&](const Coordinates & id)
        {
            int ih = 0;
            int oh = 0;

            const uint8_t *input_ptr        = in.ptr() - conv_pad_x * input_stride_x - conv_pad_y * input_stride_y + (id.z() / depth_multiplier) * input_stride_z + input_stride_w * id[3];
            const uint8_t *ptr_weights_base = weights_ptr + id.z() * kernel_stride_z;

            const auto ptr_weights_r0 = reinterpret_cast<const T1 *>(ptr_weights_base);
            const auto ptr_weights_r1 = reinterpret_cast<const T1 *>(ptr_weights_base + kernel_stride_y);
            const auto ptr_weights_r2 = reinterpret_cast<const T1 *>(ptr_weights_base + kernel_stride_y * 2);
            const auto vw_r0          = detail::load_matrix_row(ptr_weights_r0, weights_offset);
            const auto vw_r1          = detail::load_matrix_row(ptr_weights_r1, weights_offset);
            const auto vw_r2          = detail::load_matrix_row(ptr_weights_r2, weights_offset);

            for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
            {
                auto in_top = reinterpret_cast<const T1 *>(input_ptr + (ih + 0) * input_stride_y);
                auto in_mid = reinterpret_cast<const T1 *>(input_ptr + (ih + dilation.y()) * input_stride_y);
                auto in_low = reinterpret_cast<const T1 *>(input_ptr + (ih + 2 * dilation.y()) * input_stride_y); //uint8
                auto p_out  = reinterpret_cast<T2 *>(out.ptr() + oh * output_stride_y);                           //int32

                for(int ow = 0; ow < output_w; ow += num_elems_written_per_iteration,
                    in_top += delta_input, in_mid += delta_input, in_low += delta_input,
                    p_out += num_elems_written_per_iteration)
                {
                    if(dilation == Size2D(1U, 1U))
                    {
                        auto vres = detail::convolve_3x3<stridex>(in_top, in_mid, in_low, vw_r0, vw_r1, vw_r2, input_offset);
                        detail::store_results<stridex>(p_out, vres);
                    }
                    else
                    {
                        auto vres = detail::convolve_3x3_dilation<stridex>(in_top, in_mid, in_low, vw_r0, vw_r1, vw_r2, dilation.x(), input_offset);
                        detail::store_results<stridex>(p_out, vres);
                    }
                }
            }
        },
        out);
    }
};

template <typename T1, typename T2>
inline void convolve_3x3(const Window &window, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output,
                         const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation)
{
    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    switch(conv_stride_x)
    {
        case 1:
            convolver_3x3<T1, T2, 1>::convolve(window, num_elems_written_per_iteration, input, weights, output, conv_info, depth_multiplier, dilation);
            break;
        case 2:
            convolver_3x3<T1, T2, 2>::convolve(window, num_elems_written_per_iteration, input, weights, output, conv_info, depth_multiplier, dilation);
            break;
        case 3:
            convolver_3x3<T1, T2, 3>::convolve(window, num_elems_written_per_iteration, input, weights, output, conv_info, depth_multiplier, dilation);
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    const DataLayout   data_layout = input->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(width_idx) != 3 || weights->dimension(height_idx) != 3);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.stride().first < 1 || conv_info.stride().first > 3);

    if(output->total_size() != 0)
    {
        const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);

        if(is_data_type_quantized_asymmetric(input->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON(output->data_type() != DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        }
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                                                        const Size2D &dilation)
{
    Window win;
    bool   window_changed = false;

    // Get convolved dimensions
    const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);
    const DataType    output_dt    = (input->data_type() == DataType::QASYMM8) ? DataType::S32 : input->data_type();

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape).set_data_type(output_dt));

    // Configure kernel window (generic)
    const unsigned int conv_stride_x = conv_info.stride().first;
    const unsigned int conv_stride_y = conv_info.stride().second;
    const unsigned int conv_pad_top  = conv_info.pad_top();
    const unsigned int conv_pad_left = conv_info.pad_left();

    unsigned int num_elems_written_per_iteration = 16 >> conv_stride_x;
    unsigned int num_elems_read_per_iteration    = 0;

    switch(input->data_type())
    {
        case DataType::QASYMM8:
            num_elems_read_per_iteration = 16 + 15 * (dilation.x() - 1);
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            num_elems_written_per_iteration = 32 >> conv_stride_x;
            num_elems_read_per_iteration    = 24 + 23 * (dilation.x() - 1);
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            num_elems_read_per_iteration = 12 + 11 * (dilation.x() - 1);
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not supported.");
    }

    // Configure kernel window
    win = calculate_max_window(*output, Steps(num_elems_written_per_iteration));

    AccessWindowRectangle  input_access(input, -conv_pad_left, -conv_pad_top, num_elems_read_per_iteration, 3 + 2 * (dilation.y() - 1), conv_stride_x, conv_stride_y);
    AccessWindowStatic     weights_access(weights, 0, 0, 3, 3);
    AccessWindowHorizontal output_access(output, 0, num_elems_written_per_iteration);

    window_changed = update_window_and_padding(win, input_access, weights_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEDepthwiseConvolutionLayer3x3Kernel::NEDepthwiseConvolutionLayer3x3Kernel()
    : _border_size(0), _input(), _output(), _weights(), _conv_info(), _num_elems_written_per_iteration(0), _depth_multiplier(1), _dilation()
{
}

BorderSize NEDepthwiseConvolutionLayer3x3Kernel::border_size() const
{
    return _border_size;
}

void NEDepthwiseConvolutionLayer3x3Kernel::configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                                                     const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), output->info(), conv_info, depth_multiplier, dilation));

    _input            = input;
    _output           = output;
    _weights          = weights;
    _conv_info        = conv_info;
    _depth_multiplier = depth_multiplier;
    switch(input->info()->data_type())
    {
        case DataType::QASYMM8:
        case DataType::F32:
            _num_elems_written_per_iteration = 16 >> _conv_info.stride().first;
            break;
        case DataType::F16:
            _num_elems_written_per_iteration = 32 >> _conv_info.stride().first;
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not supported.");
    }
    _border_size    = BorderSize(_conv_info.pad_top(), _conv_info.pad_right(), _conv_info.pad_bottom(), _conv_info.pad_left());
    _dilation       = dilation;
    auto win_config = validate_and_configure_window(_input->info(), _weights->info(), _output->info(), _conv_info, _depth_multiplier, dilation);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEDepthwiseConvolutionLayer3x3Kernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                                                      const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, output, conv_info, depth_multiplier, dilation));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(), output->clone().get(), conv_info, depth_multiplier, dilation).first);
    return Status{};
}

void NEDepthwiseConvolutionLayer3x3Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_UNUSED(info);

    ARM_COMPUTE_UNUSED(info);

    switch(_input->info()->data_type())
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            convolve_3x3<float16_t, float16_t>(window, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info, _depth_multiplier, _dilation);
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            convolve_3x3<float, float>(window, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info, _depth_multiplier, _dilation);
            break;
        case DataType::QASYMM8:
            convolve_3x3<uint8_t, int32_t>(window, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info, _depth_multiplier, _dilation);
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}
} // namespace arm_compute
