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
#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>

using namespace arm_compute;
using namespace misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                          bool has_bias, const Size2D &dilation, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::QASYMM8 && has_bias);
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups > 1, "Number of groups greater than one are not supported on NEON");

    if(output->total_size() > 0)
    {
        TensorInfo expected_output = output->clone()->set_tensor_shape(compute_im2col_conv_shape(input, kernel_dims, conv_info, has_bias, dilation, false));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&expected_output, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                                                        bool has_bias, const Size2D &dilation)
{
    const unsigned int width_idx   = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);

    std::pair<unsigned int, unsigned int> convolved_dims = scaled_dimensions(input->dimension(width_idx), input->dimension(height_idx),
                                                                             kernel_dims.width, kernel_dims.height,
                                                                             conv_info, dilation);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_im2col_conv_shape(input, kernel_dims, conv_info, has_bias, dilation, false)));

    Window win = calculate_max_window(*input, Steps());
    win.set(width_idx, Window::Dimension(0, convolved_dims.first, 1));
    win.set(height_idx, Window::Dimension(0, convolved_dims.second, 1));
    win.set(channel_idx, Window::Dimension(0, 1, 1));

    // The NEIm2ColKernel doesn't need padding so update_window_and_padding() can be skipped
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    return std::make_pair(Status{}, win);
}

template <typename T, bool has_pads>
inline void linearize_volume_nchw(const uint8_t *const in_ptr,
                                  T                   *out_ptr,
                                  bool                 has_bias,
                                  int                  top_left_x,
                                  int                  top_left_y,
                                  int                  kernel_width,
                                  int                  kernel_height,
                                  int                  kernel_depth,
                                  int                  input_w,
                                  int                  input_h,
                                  int                  input_stride_x,
                                  int                  input_stride_y,
                                  int                  input_stride_z,
                                  int                  pad_value,
                                  int                  dilation_x,
                                  int                  dilation_y)
{
    const int kernel_size2 = kernel_width * kernel_height;
    const int x_e          = top_left_x + kernel_width * dilation_x;
    const int y_e          = top_left_y + kernel_height * dilation_y;

    // Linearize volume
    int d = 0;
    // This for loop linearize a volume with 3 slices. This allows:
    // 1) to reduce the iterations of the outer for loop "d"
    // 2) to have an optimized im2col for the first convolution layer where usually we have 3 IFMs
    for(; d <= (kernel_depth - 3); d += 3)
    {
        for(int y = top_left_y; y < y_e; y += dilation_y)
        {
            if((y < 0 || y >= input_h) && has_pads)
            {
                // All the values will be the offset (will be zeros when not quantized)
                for(int x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                {
                    *(out_ptr + 0 * kernel_size2) = pad_value;
                    *(out_ptr + 1 * kernel_size2) = pad_value;
                    *(out_ptr + 2 * kernel_size2) = pad_value;
                }
            }
            else
            {
                for(int x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                {
                    if((x < 0 || x >= input_w) && has_pads)
                    {
                        *(out_ptr + 0 * kernel_size2) = pad_value;
                        *(out_ptr + 1 * kernel_size2) = pad_value;
                        *(out_ptr + 2 * kernel_size2) = pad_value;
                    }
                    else
                    {
                        *(out_ptr + 0 * kernel_size2) = *(reinterpret_cast<const T *>(in_ptr + ((d + 0) * input_stride_z + y * input_stride_y + x * input_stride_x)));
                        *(out_ptr + 1 * kernel_size2) = *(reinterpret_cast<const T *>(in_ptr + ((d + 1) * input_stride_z + y * input_stride_y + x * input_stride_x)));
                        *(out_ptr + 2 * kernel_size2) = *(reinterpret_cast<const T *>(in_ptr + ((d + 2) * input_stride_z + y * input_stride_y + x * input_stride_x)));
                    }
                }
            }
        }
        out_ptr += 2 * kernel_size2;
    }

    // Left over
    for(; d < kernel_depth; d++)
    {
        for(int y = top_left_y; y < y_e; y += dilation_y)
        {
            if((y < 0 || y >= input_h) && has_pads)
            {
                // All the values will be the offset (will be zeros when not quantized)
                memset(out_ptr, pad_value, kernel_width * sizeof(T));
                out_ptr += kernel_width;
            }
            else
            {
                for(int x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                {
                    if((x < 0 || x >= input_w) && has_pads)
                    {
                        *out_ptr = pad_value;
                    }
                    else
                    {
                        *out_ptr = *(reinterpret_cast<const T *>(in_ptr + (d * input_stride_z + y * input_stride_y + x * input_stride_x)));
                    }
                }
            }
        }
    }

    // Append 1 if the convolution layer has biases
    if(has_bias)
    {
        *out_ptr = static_cast<T>(1);
    }
}

template <typename T, bool has_pads>
inline void linearize_volume_nhwc(const uint8_t *const in_ptr,
                                  T                   *out_ptr,
                                  bool                 has_bias,
                                  int                  start_x,
                                  int                  start_y,
                                  int                  kernel_width,
                                  int                  kernel_height,
                                  int                  input_w,
                                  int                  input_h,
                                  int                  input_c,
                                  int                  input_stride_y,
                                  int                  input_stride_z,
                                  int                  pad_value,
                                  int                  dilation_x,
                                  int                  dilation_y)
{
    const int end_x     = start_x + kernel_width * dilation_x;
    const int end_y     = start_y + kernel_height * dilation_y;
    const int pad_quant = kernel_width * input_c;
    if((start_y >= 0) && (end_y < input_h) && (start_x >= 0) && (end_x < input_w) && (dilation_x == 1))
    {
        for(int y = start_y; y < end_y; y += dilation_y)
        {
            //optimized for no dilation and no boundary pixels
            memcpy(out_ptr, reinterpret_cast<const T *>(in_ptr + (y * input_stride_z + start_x * input_stride_y)), input_c * kernel_width * sizeof(T));
            out_ptr += input_c * kernel_width;
        }
    }
    else
    {
        for(int y = start_y; y < end_y; y += dilation_y)
        {
            if(y < 0 || y >= input_h)
            {
                memset(out_ptr, pad_value, pad_quant * sizeof(T));
                out_ptr += pad_quant;
            }
            else if(dilation_x > 1 || start_x < 0 || end_x >= input_w)
            {
                for(int x = start_x; x < end_x; x += dilation_x)
                {
                    if(x < 0 || x >= input_w)
                    {
                        memset(out_ptr, pad_value, input_c * sizeof(T));
                        out_ptr += input_c;
                    }
                    else
                    {
                        memcpy(out_ptr, reinterpret_cast<const T *>(in_ptr + (y * input_stride_z + x * input_stride_y)), input_c * sizeof(T));
                        out_ptr += input_c;
                    }
                }
            }
            else
            {
                //optimized for no dilation and no boundary pixels
                memcpy(out_ptr, reinterpret_cast<const T *>(in_ptr + (y * input_stride_z + start_x * input_stride_y)), input_c * kernel_width * sizeof(T));
                out_ptr += input_c * kernel_width;
            }
        }
    }
    // Append 1 if the convolution layer has biases
    if(has_bias)
    {
        *out_ptr = static_cast<T>(1);
    }
}
} // namespace

template <typename T, bool has_pads, bool is_nchw>
void NEIm2ColKernel::run_im2col(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const DataLayout   data_layout = _input->info()->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const int input_w        = _input->info()->dimension(width_idx);
    const int input_h        = _input->info()->dimension(height_idx);
    const int input_c        = _input->info()->dimension(channel_idx);
    const int input_stride_x = _input->info()->strides_in_bytes().x();
    const int input_stride_y = _input->info()->strides_in_bytes().y();
    const int input_stride_z = _input->info()->strides_in_bytes().z();
    const int pad_left       = _conv_info.pad_left();
    const int pad_top        = _conv_info.pad_top();
    const int stride_x       = _conv_info.stride().first;
    const int stride_y       = _conv_info.stride().second;
    const int pad_value      = is_data_type_quantized(_input->info()->data_type()) ? _input->info()->quantization_info().offset : 0;

    Window window_in_out(window);
    // The first three dimensions of the input and output are increased by the inner loops
    window_in_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_in_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_in_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Create iterators
    Iterator in(_input, window_in_out);
    Iterator out(_output, window_in_out);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int start_w = id[width_idx] * stride_x - pad_left;
        const int start_h = id[height_idx] * stride_y - pad_top;

        // Get pointers
        const uint8_t *const input_ptr  = in.ptr();
        auto                 output_ptr = reinterpret_cast<T *>(out.ptr() + (id[width_idx] + id[height_idx] * _convolved_dims.first) * _output->info()->strides_in_bytes().y());

        // Linearize volume
        if(is_nchw)
        {
            linearize_volume_nchw<T, has_pads>(input_ptr,
                                               output_ptr,
                                               _has_bias,
                                               start_w,
                                               start_h,
                                               _kernel_width,
                                               _kernel_height,
                                               input_c,
                                               input_w,
                                               input_h,
                                               input_stride_x,
                                               input_stride_y,
                                               input_stride_z,
                                               pad_value,
                                               _dilation.x(),
                                               _dilation.y());
        }
        else
        {
            linearize_volume_nhwc<T, has_pads>(input_ptr,
                                               output_ptr,
                                               _has_bias,
                                               start_w,
                                               start_h,
                                               _kernel_width,
                                               _kernel_height,
                                               input_w,
                                               input_h,
                                               input_c,
                                               input_stride_y,
                                               input_stride_z,
                                               pad_value,
                                               _dilation.x(),
                                               _dilation.y());
        }
    },
    in, out);
}

NEIm2ColKernel::NEIm2ColKernel()
    : _func(), _input(nullptr), _output(nullptr), _convolved_dims(), _conv_info(), _kernel_width(0), _kernel_height(0), _has_bias(false), _dilation(1U, 1U)
{
}

void NEIm2ColKernel::configure(const ITensor *input, ITensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                               bool has_bias, const Size2D &dilation, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), kernel_dims, conv_info, has_bias, dilation, num_groups));
    ARM_COMPUTE_UNUSED(num_groups);

    const DataLayout   data_layout = input->info()->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    _input          = input;
    _output         = output;
    _conv_info      = conv_info;
    _kernel_width   = kernel_dims.width;
    _kernel_height  = kernel_dims.height;
    _dilation       = dilation;
    _convolved_dims = scaled_dimensions(input->info()->dimension(width_idx), input->info()->dimension(height_idx),
                                        _kernel_width, _kernel_height,
                                        _conv_info, _dilation);
    _has_bias = has_bias;

    if(data_layout == DataLayout::NCHW)
    {
        switch(_input->info()->data_type())
        {
            case DataType::F32:
                _func = (!conv_info.has_padding()) ? &NEIm2ColKernel::run_im2col<float, false, true> : &NEIm2ColKernel::run_im2col<float, true, true>;
                break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
                _func = (!conv_info.has_padding()) ? &NEIm2ColKernel::run_im2col<float16_t, false, true> : &NEIm2ColKernel::run_im2col<float16_t, true, true>;
                break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::QASYMM8:
                _func = (!conv_info.has_padding()) ? &NEIm2ColKernel::run_im2col<qasymm8_t, false, true> : &NEIm2ColKernel::run_im2col<qasymm8_t, true, true>;
                break;
            default:
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
        }
    }
    else
    {
        switch(_input->info()->data_type())
        {
            case DataType::F32:
                _func = (!conv_info.has_padding()) ? &NEIm2ColKernel::run_im2col<float, false, false> : &NEIm2ColKernel::run_im2col<float, true, false>;
                break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
                _func = (!conv_info.has_padding()) ? &NEIm2ColKernel::run_im2col<float16_t, false, false> : &NEIm2ColKernel::run_im2col<float16_t, true, false>;
                break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::QASYMM8:
                _func = (!conv_info.has_padding()) ? &NEIm2ColKernel::run_im2col<qasymm8_t, false, false> : &NEIm2ColKernel::run_im2col<qasymm8_t, true, false>;
                break;
            default:
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
        }
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), kernel_dims, conv_info, has_bias, dilation);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEIm2ColKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                                bool has_bias, const Size2D &dilation, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, kernel_dims, conv_info, has_bias, dilation, num_groups));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), kernel_dims, conv_info, has_bias, dilation).first);
    return Status{};
}

void NEIm2ColKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (this->*_func)(window);
}
