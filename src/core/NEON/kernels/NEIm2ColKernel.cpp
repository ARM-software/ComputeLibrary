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

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                          bool has_bias, const Size2D &dilation, unsigned int num_groups, bool is_fully_connected, bool is_flatten)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::QASYMM8 && has_bias);
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups > 1, "Number of groups greater than one are not supported on NEON");

    if(output->total_size() > 0)
    {
        TensorShape expected_output_shape;

        if(is_flatten || is_fully_connected)
        {
            expected_output_shape = misc::shape_calculator::compute_flatten_shape(input);
        }
        else
        {
            expected_output_shape = misc::shape_calculator::compute_im2col_conv_shape(input, kernel_dims, conv_info, has_bias, dilation, false);
        }

        TensorInfo expected_output = output->clone()->set_tensor_shape(expected_output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&expected_output, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

template <typename T, bool has_pads>
inline void linearize_volume(const uint8_t *const in_ptr,
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
} // namespace

template <typename T, bool has_pads>
void NEIm2ColKernel::run_generic(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const DataLayout   data_layout = _input->info()->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const int kernel_depth   = _input->info()->dimension(channel_idx);
    const int input_w        = _input->info()->dimension(width_idx);
    const int input_h        = _input->info()->dimension(height_idx);
    const int input_stride_x = _input->info()->strides_in_bytes()[width_idx];
    const int input_stride_y = _input->info()->strides_in_bytes()[height_idx];
    const int input_stride_z = _input->info()->strides_in_bytes()[channel_idx];
    const int offset         = is_data_type_quantized(_input->info()->data_type()) ? _input->info()->quantization_info().offset : 0;

    int pad_left = 0;
    int pad_top  = 0;
    int stride_x = 0;
    int stride_y = 0;
    pad_left     = _conv_info.pad_left();
    pad_top      = _conv_info.pad_top();
    std::tie(stride_x, stride_y) = _conv_info.stride();

    // Setup input window
    const int start_x = -pad_left;
    const int start_y = -pad_top;

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
        const int top_left_x = id[width_idx] * stride_x + start_x;
        const int top_left_y = id[height_idx] * stride_y + start_y;

        // Get pointers
        const uint8_t *const input_ptr  = in.ptr();
        auto                 output_ptr = reinterpret_cast<T *>(out.ptr() + (id[width_idx] + id[height_idx] * _convolved_dims.first) * _output->info()->strides_in_bytes().y());

        // Linearize volume
        linearize_volume<T, has_pads>(input_ptr,
                                      output_ptr,
                                      _has_bias,
                                      top_left_x,
                                      top_left_y,
                                      static_cast<int>(_kernel_width),
                                      static_cast<int>(_kernel_height),
                                      kernel_depth,
                                      input_w,
                                      input_h,
                                      input_stride_x,
                                      input_stride_y,
                                      input_stride_z,
                                      offset,
                                      _dilation.x(),
                                      _dilation.y());
    },
    in, out);
}

template <typename T>
void NEIm2ColKernel::run_reduced(const Window &window)
{
    const size_t in_width   = _input->info()->dimension(0);
    const size_t in_height  = _input->info()->dimension(1);
    const size_t out_step_x = in_width * _input->info()->element_size();
    const size_t out_step_y = out_step_x * in_height;
    const size_t out_width  = _output->info()->dimension(0);

    Window in_window(window);
    in_window.set(Window::DimX, Window::Dimension(0, 1, 1));

    Window out_window;
    out_window.use_tensor_dimensions(_output->info()->tensor_shape());
    out_window.set(Window::DimX, Window::Dimension(out_window.x().start(), out_window.x().end(), in_width));

    Window in_slice  = in_window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_1D();

    do
    {
        Iterator in(_input, in_slice);
        Iterator out(_output, out_slice);

        uint8_t *out_ptr = out.ptr();

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            memcpy(out_ptr + id.y() * out_step_x + id.z() * out_step_y, in.ptr(), out_step_x);
        },
        in);

        // Add bias
        if(_has_bias)
        {
            *(reinterpret_cast<T *>(out_ptr) + out_width - 1) = static_cast<T>(1);
        }
    }
    while(in_window.slide_window_slice_3D(in_slice) && out_window.slide_window_slice_1D(out_slice));
}

NEIm2ColKernel::NEIm2ColKernel()
    : _func(), _input(nullptr), _output(nullptr), _convolved_dims(), _conv_info(), _kernel_width(0), _kernel_height(0), _has_bias(false), _dilation(1U, 1U)
{
}

void NEIm2ColKernel::configure(const ITensor *input, ITensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                               bool has_bias, const Size2D &dilation, unsigned int num_groups, bool is_fully_connected, bool is_flatten)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Perform validation step
    ARM_COMPUTE_UNUSED(is_fully_connected, is_flatten);
    ARM_COMPUTE_UNUSED(num_groups);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), kernel_dims, conv_info, has_bias, dilation, num_groups, is_fully_connected, is_flatten));

    const DataLayout   data_layout = input->info()->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

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

    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();

    bool run_img2col_reduced = (output->info()->dimension(0) == (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))) && (TensorShape::num_max_dimensions >= 4)
                               && (std::equal(input->info()->tensor_shape().cbegin() + 3,
                                              input->info()->tensor_shape().cend(),
                                              output->info()->tensor_shape().cbegin() + 1))
                               && ((stride_x == 1) && (stride_y == 1) && !conv_info.has_padding())
                               && ((dilation.x() == 1) && (dilation.y() == 1));

    Window window = calculate_max_window(*input->info(), Steps());

    if(run_img2col_reduced)
    {
        switch(_input->info()->data_type())
        {
            case DataType::F32:
                _func = &NEIm2ColKernel::run_reduced<float>;
                break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
                _func = &NEIm2ColKernel::run_reduced<float16_t>;
                break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::QASYMM8:
                _func = &NEIm2ColKernel::run_reduced<qasymm8_t>;
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
                _func = (!conv_info.has_padding()) ? &NEIm2ColKernel::run_generic<float, false> : &NEIm2ColKernel::run_generic<float, true>;
                break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
                _func = (!conv_info.has_padding()) ? &NEIm2ColKernel::run_generic<float16_t, false> : &NEIm2ColKernel::run_generic<float16_t, true>;
                break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::QASYMM8:
                _func = (!conv_info.has_padding()) ? &NEIm2ColKernel::run_generic<qasymm8_t, false> : &NEIm2ColKernel::run_generic<qasymm8_t, true>;
                break;
            default:
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
        }
        window.set(width_idx, Window::Dimension(0, _convolved_dims.first, 1));
        window.set(height_idx, Window::Dimension(0, _convolved_dims.second, 1));
        window.set(channel_idx, Window::Dimension(0, 1, 1));
    }

    // The NEIm2ColKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    IKernel::configure(window);
}

Status NEIm2ColKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                                bool has_bias, const Size2D &dilation, unsigned int num_groups, bool is_fully_connected, bool is_flatten)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, kernel_dims, conv_info, has_bias, dilation, num_groups, is_fully_connected, is_flatten));
    return Status{};
}

void NEIm2ColKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (this->*_func)(window);
}
