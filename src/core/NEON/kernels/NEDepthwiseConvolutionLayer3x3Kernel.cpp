/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/AccessWindowTranspose.h"
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

using namespace arm_compute;
using namespace arm_compute::detail;
using namespace arm_compute::misc::shape_calculator;
using namespace depthwise;

namespace
{
template <typename T1, typename T2, unsigned int stridex>
class convolver_3x3
{
public:
    static void convolve(const Window &window, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
    {
        const int input_offset   = -input->info()->quantization_info().offset;
        const int weights_offset = -weights->info()->quantization_info().offset;

        const int          input_stride_x  = input->info()->strides_in_bytes().x();
        const int          input_stride_y  = input->info()->strides_in_bytes().y();
        const int          output_stride_y = output->info()->strides_in_bytes().y();
        const int          kernel_stride_y = weights->info()->strides_in_bytes().y();
        const int          kernel_stride_z = weights->info()->strides_in_bytes().z();
        const int          output_w        = output->info()->dimension(0);
        const int          output_h        = output->info()->dimension(1);
        const int          delta_input     = get_input_num_elems_processed<stridex>(num_elems_written_per_iteration);
        const unsigned int conv_stride_y   = std::get<1>(conv_info.stride());
        const unsigned int conv_pad_x      = conv_info.pad_left();
        const unsigned int conv_pad_y      = conv_info.pad_top();

        // setup output window for the iterator
        Window window_out = window;
        window_out.set(Window::DimX, Window::Dimension(0, output->info()->dimension(Window::DimX), output->info()->dimension(Window::DimX)));
        window_out.set(Window::DimY, Window::Dimension(0, output->info()->dimension(Window::DimY), output->info()->dimension(Window::DimY)));

        // setup input window for the iterator
        Window window_in = window;
        // we just want execute_window_loop to iterate over the dimensions > 2, so we set the first 2 dimensions to 0
        window_in.set(Window::DimX, Window::Dimension(0, 0, 0));
        window_in.set(Window::DimY, Window::Dimension(0, 0, 0));

        Window window_k = calculate_max_window(*weights->info(), Steps(1u));

        Iterator in(input, window_in);
        Iterator out(output, window_out);
        Iterator w(weights, window_k);

        const uint8_t *weights_ptr = w.ptr();

        execute_window_loop(window_out, [&](const Coordinates & id)
        {
            int ih = 0;
            int oh = 0;

            const uint8_t *input_ptr        = in.ptr() - conv_pad_x * input_stride_x - conv_pad_y * input_stride_y;
            const uint8_t *ptr_weights_base = weights_ptr + id.z() * kernel_stride_z;

            const auto ptr_weights_r0 = reinterpret_cast<const T1 *>(ptr_weights_base);
            const auto ptr_weights_r1 = reinterpret_cast<const T1 *>(ptr_weights_base + kernel_stride_y);
            const auto ptr_weights_r2 = reinterpret_cast<const T1 *>(ptr_weights_base + kernel_stride_y * 2);
            const auto vw_r0          = load_matrix_row(ptr_weights_r0, weights_offset);
            const auto vw_r1          = load_matrix_row(ptr_weights_r1, weights_offset);
            const auto vw_r2          = load_matrix_row(ptr_weights_r2, weights_offset);

            for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
            {
                auto in_top = reinterpret_cast<const T1 *>(input_ptr + (ih + 0) * input_stride_y);
                auto in_mid = reinterpret_cast<const T1 *>(input_ptr + (ih + 1) * input_stride_y);
                auto in_low = reinterpret_cast<const T1 *>(input_ptr + (ih + 2) * input_stride_y);
                auto p_out  = reinterpret_cast<T2 *>(out.ptr() + oh * output_stride_y);

                for(int ow = 0; ow < output_w; ow += num_elems_written_per_iteration,
                    in_top += delta_input, in_mid += delta_input, in_low += delta_input,
                    p_out += num_elems_written_per_iteration)
                {
                    auto vres = convolve_3x3<stridex>(in_top, in_mid, in_low, vw_r0, vw_r1, vw_r2, 0, input_offset);
                    store_results<stridex>(p_out, vres);
                }
            }
        },
        in, out);
    }
};

template <typename T1, typename T2>
inline void convolve_3x3(const Window &window, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
{
    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    switch(conv_stride_x)
    {
        case 1:
            convolver_3x3<T1, T2, 1>::convolve(window, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        case 2:
            convolver_3x3<T1, T2, 2>::convolve(window, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        case 3:
            convolver_3x3<T1, T2, 3>::convolve(window, num_elems_written_per_iteration, input, weights, output, conv_info);
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}
} // namespace

NEDepthwiseConvolutionLayer3x3Kernel::NEDepthwiseConvolutionLayer3x3Kernel()
    : _border_size(0), _input(), _output(), _weights(), _conv_info(), _convolver(nullptr), _num_elems_written_per_iteration(0), _run_optimized(false)
{
}

BorderSize NEDepthwiseConvolutionLayer3x3Kernel::border_size() const
{
    return _border_size;
}

void NEDepthwiseConvolutionLayer3x3Kernel::configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info, DataLayout data_layout)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    _input     = input;
    _output    = output;
    _weights   = weights;
    _conv_info = conv_info;
    _convolver = nullptr;

    _run_optimized = NEDepthwiseConvolutionLayer3x3Kernel::is_optimized_execution_possible(input->info()->tensor_shape(),
                                                                                           conv_info,
                                                                                           input->info()->data_type(),
                                                                                           data_layout);

    (_run_optimized) ? configure_optimized() : configure_generic();
}

void NEDepthwiseConvolutionLayer3x3Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_UNUSED(info);

    (_run_optimized) ? run_optimized(window, info) : run_generic(window, info);
}

bool NEDepthwiseConvolutionLayer3x3Kernel::is_optimized_execution_possible(TensorShape input_shape, PadStrideInfo conv_info, DataType dt, DataLayout data_layout)
{
    // Reshape input shape if in NHWC format
    TensorShape in_shape{ input_shape };
    if(data_layout == DataLayout::NHWC)
    {
        in_shape.set(Window::DimX, input_shape.y());
        in_shape.set(Window::DimY, input_shape.z());
        in_shape.set(Window::DimZ, input_shape.x());
    }

    // Check supported data type
    bool supported_datatype = (dt == DataType::F32);

    // Check for supported strides
    const auto &strides           = conv_info.stride();
    bool        supported_strides = (strides.first == strides.second) && ((strides.first == 1) || (strides.first == 2));

    // Check for supported padding
    const auto    pad_top           = conv_info.pad_top();
    const auto    pad_right         = conv_info.pad_right();
    const auto    pad_bottom        = conv_info.pad_bottom();
    const auto    pad_left          = conv_info.pad_left();
    PadStrideInfo same_pad          = calculate_same_pad(in_shape, TensorShape(3U, 3U), conv_info);
    bool          is_same_padding   = (pad_top == same_pad.pad_top()) && (pad_right == same_pad.pad_right()) && (pad_bottom == same_pad.pad_bottom()) && (pad_left == same_pad.pad_left());
    bool          is_valid_padding  = (pad_top == 0) && (pad_right == 0) && (pad_bottom == 0) && (pad_left == 0);
    bool          supported_padding = is_same_padding || is_valid_padding;

    return supported_datatype && supported_strides && supported_padding;
}

void NEDepthwiseConvolutionLayer3x3Kernel::generate_convolver()
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(_input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(_input, _weights);
    ARM_COMPUTE_ERROR_ON(_weights->info()->dimension(1) != 3 || _weights->info()->dimension(2) != 3);

    _convolver = create_convolver_object(_input->info()->tensor_shape(), _conv_info,
                                         _weights->buffer(), _input->buffer(), _output->buffer());
}

void NEDepthwiseConvolutionLayer3x3Kernel::configure_generic()
{
    ARM_COMPUTE_ERROR_ON(_weights->info()->dimension(0) != 3 || _weights->info()->dimension(1) != 3);

    // Get convolved dimensions
    const TensorShape output_shape = compute_depthwise_convolution_shape(*_input->info(), *_weights->info(), _conv_info);
    const DataType    output_dt    = (_input->info()->data_type() == DataType::QASYMM8) ? DataType::S32 : _input->info()->data_type();

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*_output->info(),
                       _input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape).set_data_type(output_dt));

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(_output->info()->tensor_shape(), output_shape);

    const unsigned int conv_stride_x   = _conv_info.stride().first;
    const unsigned int conv_pad_top    = _conv_info.pad_top();
    const unsigned int conv_pad_right  = _conv_info.pad_right();
    const unsigned int conv_pad_bottom = _conv_info.pad_bottom();
    const unsigned int conv_pad_left   = _conv_info.pad_left();

    ARM_COMPUTE_ERROR_ON(conv_stride_x < 1 || conv_stride_x > 3);

    unsigned int num_elems_read_per_iteration = 0;
    switch(_input->info()->data_type())
    {
        case DataType::QASYMM8:
            num_elems_read_per_iteration     = 16;
            _num_elems_written_per_iteration = 16 >> conv_stride_x;
            break;
        case DataType::F32:
            num_elems_read_per_iteration     = 12;
            _num_elems_written_per_iteration = 16 >> conv_stride_x;
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not supported.");
    }
    _border_size = BorderSize(conv_pad_top, conv_pad_right, conv_pad_bottom, conv_pad_left);

    // Configure kernel window
    Window win = calculate_max_window(*_output->info(), Steps(_num_elems_written_per_iteration));

    const unsigned int num_x_steps               = (output_shape.x() + _num_elems_written_per_iteration - 1) / _num_elems_written_per_iteration;
    const int          input_num_elems_processed = get_input_num_elems_processed(_num_elems_written_per_iteration, conv_stride_x);

    AccessWindowStatic input_access(_input->info(),
                                    -conv_pad_left,
                                    -conv_pad_top,
                                    (num_x_steps - 1) * input_num_elems_processed + num_elems_read_per_iteration,
                                    _input->info()->tensor_shape().y() + conv_pad_bottom);
    AccessWindowStatic weights_access(_weights->info(), 0, 0, _weights->info()->dimension(0), _weights->info()->dimension(1));
    AccessWindowStatic output_access(_output->info(), 0, 0, num_x_steps * _num_elems_written_per_iteration, output_shape.y());

    update_window_and_padding(win, input_access, weights_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), _output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEDepthwiseConvolutionLayer3x3Kernel::configure_optimized()
{
    ARM_COMPUTE_ERROR_ON(_weights->info()->dimension(1) != 3 || _weights->info()->dimension(2) != 3);

    _border_size = BorderSize(0, 0);
    _convolver   = create_convolver_object(_input->info()->tensor_shape(), _conv_info,
                                           _weights->buffer(), _input->buffer(), _output->buffer());

    // Auto-configure output
    bool        same_padding = _conv_info.has_padding();
    TensorShape output_shape{ _input->info()->tensor_shape() };

    output_shape.set(1, _convolver->output_size(output_shape.y(), same_padding)); // Set width
    output_shape.set(2, _convolver->output_size(output_shape.z(), same_padding)); // Set height

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*_output->info(),
                       _input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape));

    // Configure window
    Window win;
    auto   win_last = _convolver->get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

void NEDepthwiseConvolutionLayer3x3Kernel::run_generic(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    switch(_input->info()->data_type())
    {
        case DataType::F32:
            convolve_3x3<float, float>(window, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
            break;
        case DataType::QASYMM8:
            convolve_3x3<uint8_t, int32_t>(window, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}

void NEDepthwiseConvolutionLayer3x3Kernel::run_optimized(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON(!_convolver);

    const size_t start = window.x().start();
    const size_t end   = window.x().end();
    _convolver->run(start, end);
}

std::unique_ptr<depthwise::IDepthwiseConvolution> NEDepthwiseConvolutionLayer3x3Kernel::create_convolver_object(TensorShape    shape,
                                                                                                                PadStrideInfo  conv_info,
                                                                                                                const uint8_t *w_ptr,
                                                                                                                uint8_t       *in_ptr,
                                                                                                                uint8_t       *out_ptr)
{
    const int  in_rows      = shape.z();
    const int  in_cols      = shape.y();
    const int  n_batches    = shape[3];
    const int  n_channels   = shape.x();
    const bool padding_same = conv_info.has_padding();

    const auto stride_x = conv_info.stride().first;
    switch(stride_x)
    {
        case 1:
            return arm_compute::support::cpp14::make_unique<DepthwiseConvolution<2, 2, 3, 3, 1, 1, float, float>>(
                       n_batches,
                       in_rows,
                       in_cols,
                       n_channels,
                       padding_same,
                       reinterpret_cast<const float *>(w_ptr),
                       reinterpret_cast<float *>(in_ptr),
                       reinterpret_cast<float *>(out_ptr));
        case 2:
            return arm_compute::support::cpp14::make_unique<DepthwiseConvolution<2, 2, 3, 3, 2, 2, float, float>>(
                       n_batches,
                       in_rows,
                       in_cols,
                       n_channels,
                       padding_same,
                       reinterpret_cast<const float *>(w_ptr),
                       reinterpret_cast<float *>(in_ptr),
                       reinterpret_cast<float *>(out_ptr));
        default:
            return nullptr;
    }
}