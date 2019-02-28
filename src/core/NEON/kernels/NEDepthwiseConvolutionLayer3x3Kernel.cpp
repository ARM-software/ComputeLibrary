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
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier)
    {
        const int input_offset   = -input->info()->quantization_info().offset;
        const int weights_offset = -weights->info()->quantization_info().offset;

        const int          input_stride_x  = input->info()->strides_in_bytes().x();
        const int          input_stride_y  = input->info()->strides_in_bytes().y();
        const int          input_stride_z  = input->info()->strides_in_bytes().z();
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

            const uint8_t *input_ptr        = in.ptr() - conv_pad_x * input_stride_x - conv_pad_y * input_stride_y - (id.z() - id.z() / depth_multiplier) * input_stride_z;
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
                    auto vres = convolve_3x3<stridex>(in_top, in_mid, in_low, vw_r0, vw_r1, vw_r2, input_offset);
                    store_results<stridex>(p_out, vres);
                }
            }
        },
        in, out);
    }
};

template <typename T1, typename T2>
inline void convolve_3x3(const Window &window, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier)
{
    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    switch(conv_stride_x)
    {
        case 1:
            convolver_3x3<T1, T2, 1>::convolve(window, num_elems_written_per_iteration, input, weights, output, conv_info, depth_multiplier);
            break;
        case 2:
            convolver_3x3<T1, T2, 2>::convolve(window, num_elems_written_per_iteration, input, weights, output, conv_info, depth_multiplier);
            break;
        case 3:
            convolver_3x3<T1, T2, 3>::convolve(window, num_elems_written_per_iteration, input, weights, output, conv_info, depth_multiplier);
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier, bool is_optimized)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    const DataLayout   data_layout = input->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(width_idx) != 3 || weights->dimension(height_idx) != 3);

    if(!is_optimized)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(conv_info.stride().first < 1 || conv_info.stride().first > 3);
    }

    if(output->total_size() != 0)
    {
        const TensorShape output_shape = compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier);
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

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier, bool is_optimized,
                                                        IDepthwiseConvolution *convolver = nullptr)
{
    Window win;
    bool   window_changed = false;

    if(is_optimized)
    {
        if(convolver != nullptr)
        {
            auto win_last = convolver->get_window();
            win.set(Window::DimX, Window::Dimension(0, win_last, 1));

            // Auto-configure output
            bool        same_padding = conv_info.has_padding();
            TensorShape output_shape{ input->tensor_shape() };

            output_shape.set(1, convolver->output_size(output_shape.y(), same_padding)); // Set width
            output_shape.set(2, convolver->output_size(output_shape.z(), same_padding)); // Set height

            const DataType output_dt = (input->data_type() == DataType::QASYMM8) ? DataType::S32 : input->data_type();

            // Output auto inizialitation if not yet initialized
            auto_init_if_empty(*output, input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape).set_data_type(output_dt));

            // Configure window (optimised)
            // Set padding in channels
            const int num_channels = weights->dimension(0);
            if((num_channels >= 128) && (num_channels % 16 == 0))
            {
                input->extend_padding(PaddingSize(0, 4, 0, 0));
                weights->extend_padding(PaddingSize(0, 4, 0, 0));
                output->extend_padding(PaddingSize(0, 4, 0, 0));
            }
        }
    }
    else
    {
        // Get convolved dimensions
        const TensorShape output_shape = compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier);
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
                num_elems_read_per_iteration = 16;
                break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
                num_elems_read_per_iteration = 24;
                break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F32:
                num_elems_read_per_iteration = 12;
                break;
            default:
                ARM_COMPUTE_ERROR("Data type not supported.");
        }

        // Configure kernel window
        win = calculate_max_window(*output, Steps(num_elems_written_per_iteration));

        AccessWindowRectangle  input_access(input, -conv_pad_left, -conv_pad_top, num_elems_read_per_iteration, 3, conv_stride_x, conv_stride_y);
        AccessWindowStatic     weights_access(weights, 0, 0, 3, 3);
        AccessWindowHorizontal output_access(output, 0, num_elems_written_per_iteration);

        window_changed = update_window_and_padding(win, input_access, weights_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEDepthwiseConvolutionLayer3x3Kernel::NEDepthwiseConvolutionLayer3x3Kernel()
    : _border_size(0), _input(), _output(), _weights(), _conv_info(), _convolver(nullptr), _num_elems_written_per_iteration(0), _run_optimized(false), _depth_multiplier(1)
{
}

BorderSize NEDepthwiseConvolutionLayer3x3Kernel::border_size() const
{
    return _border_size;
}

void NEDepthwiseConvolutionLayer3x3Kernel::configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                                                     DataLayout data_layout)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    _input            = input;
    _output           = output;
    _weights          = weights;
    _conv_info        = conv_info;
    _depth_multiplier = depth_multiplier;
    _convolver        = nullptr;

    _run_optimized = NEDepthwiseConvolutionLayer3x3Kernel::is_optimized_execution_possible(input->info()->tensor_shape(),
                                                                                           conv_info,
                                                                                           input->info()->data_type(), depth_multiplier,
                                                                                           data_layout);

    (_run_optimized) ? configure_optimized() : configure_generic();
}

Status NEDepthwiseConvolutionLayer3x3Kernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);

    bool is_optimized = NEDepthwiseConvolutionLayer3x3Kernel::is_optimized_execution_possible(input->tensor_shape(), conv_info, input->data_type(), depth_multiplier, input->data_layout());

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, output, conv_info, depth_multiplier, is_optimized));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(), output->clone().get(), conv_info, depth_multiplier, is_optimized).first);
    return Status{};
}

void NEDepthwiseConvolutionLayer3x3Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_UNUSED(info);

    (_run_optimized) ? run_optimized(window, info) : run_generic(window, info);
}

bool NEDepthwiseConvolutionLayer3x3Kernel::is_optimized_execution_possible(TensorShape input_shape, PadStrideInfo conv_info, DataType dt, unsigned int depth_multiplier, DataLayout data_layout)
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
    bool supported_datatype = is_data_type_float(dt) || is_data_type_quantized(dt);

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

    return supported_datatype && supported_strides && supported_padding && (depth_multiplier == 1);
}

void NEDepthwiseConvolutionLayer3x3Kernel::generate_convolver()
{
    ARM_COMPUTE_ERROR_ON_CPU_F16_UNSUPPORTED(_input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(_input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(_input, _weights);
    ARM_COMPUTE_ERROR_ON(_weights->info()->dimension(1) != 3 || _weights->info()->dimension(2) != 3);

    _convolver = create_convolver_object(_conv_info, _weights, _input, _output, true);
    if(_convolver)
    {
        _convolver->set_offsets(-_input->info()->quantization_info().offset, -_weights->info()->quantization_info().offset);
    }
}

void NEDepthwiseConvolutionLayer3x3Kernel::configure_generic()
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(_input->info(), _weights->info(), _output->info(), _conv_info, _depth_multiplier, _run_optimized));

    _num_elems_written_per_iteration = 16 >> _conv_info.stride().first;
    _border_size                     = BorderSize(_conv_info.pad_top(), _conv_info.pad_right(), _conv_info.pad_bottom(), _conv_info.pad_left());

    auto win_config = validate_and_configure_window(_input->info(), _weights->info(), _output->info(), _conv_info, _depth_multiplier, false);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

void NEDepthwiseConvolutionLayer3x3Kernel::configure_optimized()
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(_input->info(), _weights->info(), _output->info(), _conv_info, _depth_multiplier, _run_optimized));

    _border_size = BorderSize(0, 0);
    _convolver   = create_convolver_object(_conv_info, _weights, _input, _output);

    auto win_config = validate_and_configure_window(_input->info(), _weights->info(), _output->info(), _conv_info, _depth_multiplier, true, _convolver.get());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

void NEDepthwiseConvolutionLayer3x3Kernel::run_generic(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    switch(_input->info()->data_type())
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            convolve_3x3<float16_t, float16_t>(window, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info, _depth_multiplier);
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            convolve_3x3<float, float>(window, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info, _depth_multiplier);
            break;
        case DataType::QASYMM8:
            convolve_3x3<uint8_t, int32_t>(window, _num_elems_written_per_iteration, _input, _weights, _output, _conv_info, _depth_multiplier);
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

std::unique_ptr<depthwise::IDepthwiseConvolution> NEDepthwiseConvolutionLayer3x3Kernel::create_convolver_object(PadStrideInfo  conv_info,
                                                                                                                const ITensor *w,
                                                                                                                const ITensor *in,
                                                                                                                ITensor       *out,
                                                                                                                bool           setup_strides)
{
    const DataType    dt                  = in->info()->data_type();
    const TensorShape shape               = in->info()->tensor_shape();
    const int         in_rows             = shape.z();
    const int         in_cols             = shape.y();
    const int         n_batches           = shape[3];
    const int         n_channels          = shape.x();
    const bool        padding_same        = conv_info.has_padding();
    const int         weight_col_stride   = (setup_strides) ? w->info()->strides_in_bytes().y() / w->info()->element_size() : 0;
    const int         weight_row_stride   = (setup_strides) ? w->info()->strides_in_bytes().z() / w->info()->element_size() : 0;
    const int         input_col_stride    = (setup_strides) ? in->info()->strides_in_bytes().y() / in->info()->element_size() : 0;
    const int         input_row_stride    = (setup_strides) ? in->info()->strides_in_bytes().z() / in->info()->element_size() : 0;
    const int         input_batch_stride  = (setup_strides) ? in->info()->strides_in_bytes()[3] / in->info()->element_size() : 0;
    const int         output_col_stride   = (setup_strides) ? out->info()->strides_in_bytes().y() / out->info()->element_size() : 0;
    const int         output_row_stride   = (setup_strides) ? out->info()->strides_in_bytes().z() / out->info()->element_size() : 0;
    const int         output_batch_stride = (setup_strides) ? out->info()->strides_in_bytes()[3] / out->info()->element_size() : 0;

    const auto stride_x = conv_info.stride().first;
    switch(dt)
    {
        case DataType::QASYMM8:
        {
            switch(stride_x)
            {
                case 1:
                    return arm_compute::support::cpp14::make_unique<DepthwiseConvolution<4, 4, 3, 3, 1, 1, uint8_t, int32_t>>(
                               n_batches, in_rows, in_cols, n_channels, padding_same,
                               reinterpret_cast<const uint8_t *>(w->ptr_to_element(Coordinates())),
                               in->ptr_to_element(Coordinates()),
                               reinterpret_cast<int32_t *>(out->ptr_to_element(Coordinates())), weight_col_stride,
                               weight_row_stride, input_col_stride, input_row_stride, input_batch_stride,
                               output_col_stride, output_row_stride, output_batch_stride);
                case 2:
                    return arm_compute::support::cpp14::make_unique<DepthwiseConvolution<4, 4, 3, 3, 2, 2, uint8_t, int32_t>>(
                               n_batches, in_rows, in_cols, n_channels, padding_same,
                               reinterpret_cast<const uint8_t *>(w->ptr_to_element(Coordinates())),
                               in->ptr_to_element(Coordinates()),
                               reinterpret_cast<int32_t *>(out->ptr_to_element(Coordinates())), weight_col_stride,
                               weight_row_stride, input_col_stride, input_row_stride, input_batch_stride,
                               output_col_stride, output_row_stride, output_batch_stride);
                default:
                    return nullptr;
            }
            break;
        }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            switch(stride_x)
            {
                case 1:
                    return arm_compute::support::cpp14::make_unique<DepthwiseConvolution<4, 4, 3, 3, 1, 1, float16_t, float16_t>>(
                               n_batches, in_rows, in_cols, n_channels, padding_same,
                               reinterpret_cast<const float16_t *>(w->ptr_to_element(Coordinates())),
                               reinterpret_cast<float16_t *>(in->ptr_to_element(Coordinates())),
                               reinterpret_cast<float16_t *>(out->ptr_to_element(Coordinates())), weight_col_stride,
                               weight_row_stride, input_col_stride, input_row_stride, input_batch_stride,
                               output_col_stride, output_row_stride, output_batch_stride);
                case 2:
                    return arm_compute::support::cpp14::make_unique<DepthwiseConvolution<4, 4, 3, 3, 2, 2, float16_t, float16_t>>(
                               n_batches, in_rows, in_cols, n_channels, padding_same,
                               reinterpret_cast<const float16_t *>(w->ptr_to_element(Coordinates())),
                               reinterpret_cast<float16_t *>(in->ptr_to_element(Coordinates())),
                               reinterpret_cast<float16_t *>(out->ptr_to_element(Coordinates())), weight_col_stride,
                               weight_row_stride, input_col_stride, input_row_stride, input_batch_stride,
                               output_col_stride, output_row_stride, output_batch_stride);
                default:
                    return nullptr;
            }
            break;
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
        {
            switch(stride_x)
            {
                case 1:
                    return arm_compute::support::cpp14::make_unique<DepthwiseConvolution<4, 4, 3, 3, 1, 1, float, float>>(
                               n_batches, in_rows, in_cols, n_channels, padding_same,
                               reinterpret_cast<const float *>(w->ptr_to_element(Coordinates())),
                               reinterpret_cast<float *>(in->ptr_to_element(Coordinates())),
                               reinterpret_cast<float *>(out->ptr_to_element(Coordinates())), weight_col_stride,
                               weight_row_stride, input_col_stride, input_row_stride, input_batch_stride,
                               output_col_stride, output_row_stride, output_batch_stride);
                case 2:
                    return arm_compute::support::cpp14::make_unique<DepthwiseConvolution<3, 3, 3, 3, 2, 2, float, float>>(
                               n_batches, in_rows, in_cols, n_channels, padding_same,
                               reinterpret_cast<const float *>(w->ptr_to_element(Coordinates())),
                               reinterpret_cast<float *>(in->ptr_to_element(Coordinates())),
                               reinterpret_cast<float *>(out->ptr_to_element(Coordinates())), weight_col_stride,
                               weight_row_stride, input_col_stride, input_row_stride, input_batch_stride,
                               output_col_stride, output_row_stride, output_batch_stride);
                default:
                    return nullptr;
            }
            break;
        }
        default:
            return nullptr;
    }
}