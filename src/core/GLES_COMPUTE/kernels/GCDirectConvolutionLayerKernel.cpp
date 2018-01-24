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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDirectConvolutionLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

template <unsigned int kernel_size>
GCDirectConvolutionLayerKernel<kernel_size>::GCDirectConvolutionLayerKernel()
    : _input(nullptr), _bias(nullptr), _weights(nullptr), _output(nullptr), _border_size(0), _conv_stride_x(0), _conv_stride_y(0), _conv_pad_x(0), _conv_pad_y(0), _lws(gles::NDRange(1U, 1U, 1U))
{
}

template <unsigned int kernel_size>
BorderSize             GCDirectConvolutionLayerKernel<kernel_size>::border_size() const
{
    return _border_size;
}

template <unsigned int kernel_size>
void GCDirectConvolutionLayerKernel<kernel_size>::configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *bias, IGCTensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(2) != input->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != weights->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON_MSG((kernel_size == 3 && std::get<0>(conv_info.stride()) > 2), "Strides larger than 2 not supported in 3x3 direct convolution!");
    ARM_COMPUTE_ERROR_ON(kernel_size != weights->info()->dimension(0));

    if(bias != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, bias);
        // FIXME: Bug in framework, workaround it in tests currently.
        //ARM_COMPUTE_ERROR_ON(bias->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(bias->info()->num_dimensions() > 1);
    }

    // Get convolved dimensions
    unsigned int owidth  = 0;
    unsigned int oheight = 0;
    std::tie(owidth, oheight) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1), kernel_size, kernel_size, conv_info);

    TensorShape output_shape = input->info()->tensor_shape();
    output_shape.set(0, owidth);
    output_shape.set(1, oheight);
    output_shape.set(2, weights->info()->dimension(3));

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    _conv_stride_x = std::get<0>(conv_info.stride());
    _conv_stride_y = std::get<1>(conv_info.stride());
    _conv_pad_x    = std::get<0>(conv_info.pad());
    _conv_pad_y    = std::get<1>(conv_info.pad());

    _input       = input;
    _weights     = weights;
    _output      = output;
    _bias        = bias;
    _border_size = BorderSize(_conv_pad_y, _conv_pad_x);

    std::set<std::string> options;

    options.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(_lws[0]));
    options.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(_lws[1]));
    options.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(_lws[2]));
    options.emplace("#define STRIDE_X " + support::cpp11::to_string(_conv_stride_x));
    options.emplace("#define STRIDE_Y " + support::cpp11::to_string(_conv_stride_y));

    std::string dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    options.emplace(("#define " + dt_name));

    unsigned int num_elems_read_per_iteration_x    = kernel_size * _conv_stride_x;
    unsigned int num_elems_read_per_iteration_y    = 1;
    unsigned int num_elems_written_per_iteration_x = 1;
    unsigned int num_elems_written_per_iteration_y = 1;
    unsigned int num_elems_written_per_iteration_z = 1;

    if(kernel_size == 3)
    {
        if((_conv_stride_x == 1) && (_conv_stride_y == 1))
        {
            switch(input->info()->data_type())
            {
                case DataType::F16:
#define PROCESS_4X_3Y_1Z

#if defined(PROCESS_8X_3Y_1Z)
                    options.emplace("#define PROCESS_8X_3Y_1Z");
                    num_elems_read_per_iteration_x    = 16;
                    num_elems_read_per_iteration_y    = 5;
                    num_elems_written_per_iteration_x = 8;
                    num_elems_written_per_iteration_y = 3;
#elif defined(PROCESS_4X_3Y_1Z)
                    options.emplace("#define PROCESS_4X_3Y_1Z");
                    num_elems_read_per_iteration_x    = 8;
                    num_elems_read_per_iteration_y    = 5;
                    num_elems_written_per_iteration_x = 4;
                    num_elems_written_per_iteration_y = 3;
#elif defined(PROCESS_4X_4Y_1Z)
                    options.emplace("#define PROCESS_4X_4Y_1Z");
                    num_elems_read_per_iteration_x    = 8;
                    num_elems_read_per_iteration_y    = 6;
                    num_elems_written_per_iteration_x = 4;
                    num_elems_written_per_iteration_y = 4;
#elif defined(PROCESS_4X_3Y_2Z)
                    options.emplace("#define PROCESS_4X_3Y_2Z");
                    num_elems_read_per_iteration_x    = 8;
                    num_elems_read_per_iteration_y    = 5;
                    num_elems_written_per_iteration_x = 4;
                    num_elems_written_per_iteration_y = 3;
                    num_elems_written_per_iteration_z = 2;
#endif /* PROCESS_nX_nY_nZ */
#undef PROCESS_8X_3Y_1Z
#undef PROCESS_4X_3Y_1Z
#undef PROCESS_4X_4Y_1Z
#undef PROCESS_4X_3Y_2Z
                    break;

                case DataType::F32:
                    options.emplace("#define PROCESS_4X_3Y_1Z");
                    num_elems_read_per_iteration_x    = 8;
                    num_elems_read_per_iteration_y    = 5;
                    num_elems_written_per_iteration_x = 4;
                    num_elems_written_per_iteration_y = 3;
                    break;

                default:
                    ARM_COMPUTE_ERROR("Current data type is not supported");
                    break;
            }
        }
        else
        {
            switch(input->info()->data_type())
            {
                case DataType::F16:
                    options.emplace("#define PROCESS_4X_1Y_1Z");
                    num_elems_read_per_iteration_x    = 8;
                    num_elems_written_per_iteration_x = 4;
                    break;

                case DataType::F32:
#define PROCESS_4X_1Y_1Z

#if defined(PROCESS_1X_1Y_1Z)
                    options.emplace("#define PROCESS_1X_1Y_1Z");
                    num_elems_read_per_iteration_x    = 3;
                    num_elems_written_per_iteration_x = 1;
#elif defined(PROCESS_4X_1Y_1Z)
                    options.emplace("#define PROCESS_4X_1Y_1Z");
                    num_elems_read_per_iteration_x    = 8;
                    num_elems_written_per_iteration_x = 4;
#elif defined(PROCESS_8X_1Y_1Z)
                    options.emplace("#define PROCESS_8X_1Y_1Z");
                    num_elems_read_per_iteration_x    = 12;
                    num_elems_written_per_iteration_x = 8;
#else /* PROCESS_nX_nY_nZ */
#error Have to declare how many elements to process in one thread.
#endif /* PROCESS_nX_nY_nZ */
#undef PROCESS_1X_1Y_1Z
#undef PROCESS_4X_1Y_1Z
#undef PROCESS_8X_1Y_1Z
                    break;

                default:
                    ARM_COMPUTE_ERROR("Current data type is not supported");
                    break;
            }
        }
    }
    else if(kernel_size == 1)
    {
        if(weights->info()->dimension(2) % 2 == 0)
        {
            options.emplace("#define WEIGHTS_OPTIMIZATION");
        }
        switch(input->info()->data_type())
        {
            case DataType::F16:
#define PROCESS_8X_2Y_1Z

#if defined(PROCESS_4X_1Y_1Z)
                options.emplace("#define PROCESS_4X_1Y_1Z");
                num_elems_read_per_iteration_x    = 4;
                num_elems_written_per_iteration_x = 4;
#elif defined(PROCESS_4X_2Y_1Z)
                options.emplace("#define PROCESS_4X_2Y_1Z");
                num_elems_read_per_iteration_x    = 4;
                num_elems_read_per_iteration_y    = 2;
                num_elems_written_per_iteration_x = 4;
                num_elems_written_per_iteration_y = 2;
#elif defined(PROCESS_4X_3Y_1Z)
                options.emplace("#define PROCESS_4X_3Y_1Z");
                num_elems_read_per_iteration_x    = 4;
                num_elems_read_per_iteration_y    = 3;
                num_elems_written_per_iteration_x = 4;
                num_elems_written_per_iteration_y = 3;
#elif defined(PROCESS_4X_4Y_1Z)
                options.emplace("#define PROCESS_4X_4Y_1Z");
                num_elems_read_per_iteration_x    = 4;
                num_elems_read_per_iteration_y    = 4;
                num_elems_written_per_iteration_x = 4;
                num_elems_written_per_iteration_y = 4;
#elif defined(PROCESS_4X_2Y_2Z)
                ARM_COMPUTE_ERROR_ON_MSG((weights->info()->dimension(4) % 2) == 1, "Current 'weights->info()->dimension(4) % 2) == 1' is not supported");
                options.emplace("#define PROCESS_4X_2Y_2Z");
                num_elems_read_per_iteration_x    = 4;
                num_elems_read_per_iteration_y    = 2;
                num_elems_written_per_iteration_x = 4;
                num_elems_written_per_iteration_y = 2;
                num_elems_written_per_iteration_z = 2;
#elif defined(PROCESS_8X_1Y_1Z)
                options.emplace("#define PROCESS_8X_1Y_1Z");
                num_elems_read_per_iteration_x    = 8;
                num_elems_written_per_iteration_x = 8;
#elif defined(PROCESS_8X_2Y_1Z)
                options.emplace("#define PROCESS_8X_2Y_1Z");
                num_elems_read_per_iteration_x    = 8;
                num_elems_read_per_iteration_y    = 2;
                num_elems_written_per_iteration_x = 8;
                num_elems_written_per_iteration_y = 2;
#else /* PROCESS_4X_1Y_1Z */
#error Have to declare how many elements to process in one thread.
#endif /* PROCESS_4X_1Y_1Z */
#undef PROCESS_4X_1Y_1Z
#undef PROCESS_4X_2Y_1Z
#undef PROCESS_4X_3Y_1Z
#undef PROCESS_4X_4Y_1Z
#undef PROCESS_4X_2Y_2Z
#undef PROCESS_8X_1Y_1Z
#undef PROCESS_8X_2Y_1Z
                break;

            case DataType::F32:
                num_elems_read_per_iteration_x    = 1;
                num_elems_written_per_iteration_x = 1;
                break;

            default:
                break;
        }
    }
    else if(kernel_size == 5)
    {
        switch(input->info()->data_type())
        {
            case DataType::F16:
                options.emplace("#define PROCESS_4X_1Y_1Z");
                num_elems_read_per_iteration_x    = 8;
                num_elems_written_per_iteration_x = 4;

            default:
                break;
        }
    }
    else
    {
    }

    if(_bias != nullptr)
    {
        options.emplace("#define BIAS");
    }

    std::stringstream kernel_name;
    kernel_name << "direct_convolution" << kernel_size << "x" << kernel_size;

    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel(kernel_name.str(), options));

    unsigned int idx = (_bias == nullptr) ? 3 * num_arguments_per_3D_tensor() : (num_arguments_per_1D_tensor() + 3 * num_arguments_per_3D_tensor());

    // Calculate output right and bottom border
    const int output_width          = output->info()->dimension(0);
    const int output_height         = output->info()->dimension(1);
    const int output_padding_right  = ceil_to_multiple(output_width, num_elems_written_per_iteration_x * _lws[0]) - output_width;
    const int output_padding_bottom = ceil_to_multiple(output_height, num_elems_written_per_iteration_y * _lws[1]) - output_height;

    // Calculate input right and bottom border
    const int input_width    = input->info()->dimension(0);
    const int input_height   = input->info()->dimension(1);
    const int upper_bound_w  = ceil_to_multiple(((output_width + output_padding_right) * _conv_stride_x + (kernel_size - 1)), num_elems_read_per_iteration_x * _lws[0]) - _conv_pad_x - input_width;
    const int upper_bound_h  = ceil_to_multiple(((output_height + output_padding_bottom) * _conv_stride_y + (kernel_size - 1)), num_elems_read_per_iteration_y * _lws[1]) - _conv_pad_y - input_height;
    const int padding_right  = std::max(upper_bound_w, _conv_pad_x);
    const int padding_bottom = std::max(upper_bound_h, _conv_pad_y);

    BorderSize border = BorderSize(0, output_padding_right, output_padding_bottom, 0);

    Window win = calculate_max_enlarged_window(*output->info(), Steps(num_elems_written_per_iteration_x, num_elems_written_per_iteration_y, num_elems_written_per_iteration_z), border);

    AccessWindowStatic input_access(input->info(), -_conv_pad_x, -_conv_pad_y, input_width + padding_right, input_height + padding_bottom);
    AccessWindowStatic weights_access = AccessWindowStatic(nullptr, 0, 0, 0, 0);
    AccessWindowStatic bias_access    = AccessWindowStatic(nullptr, 0, 0, 0, 1);

    switch(weights->info()->data_type())
    {
        case DataType::F16:
            if((weights->info()->dimension(2) % 2 != 0) || (kernel_size != 1))
            {
                weights_access = AccessWindowStatic(weights->info(), 0, 0, kernel_size + 1, kernel_size);
            }
            if(_bias != nullptr)
            {
                bias_access = AccessWindowStatic(_bias->info(), 0, 0, _bias->info()->dimension(0) + 1, 1);
            }
            break;

        case DataType::F32:
            weights_access = AccessWindowStatic(weights->info(), 0, 0, kernel_size, kernel_size);
            if(_bias != nullptr)
            {
                bias_access = AccessWindowStatic(_bias->info(), 0, 0, _bias->info()->dimension(0), 1);
            }
            break;

        default:
            ARM_COMPUTE_ERROR("Current data type is not supported");
            break;
    }

    AccessWindowStatic output_access(output->info(), 0, 0, output_width + output_padding_right, output_height + output_padding_bottom);

    if(_bias != nullptr)
    {
        update_window_and_padding(win, input_access, weights_access, bias_access, output_access);
    }
    else
    {
        update_window_and_padding(win, input_access, weights_access, output_access);
    }

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    _kernel.set_argument(idx++, _weights->info()->strides_in_bytes()[3]); // weights_stride_w
    _kernel.set_argument(idx++, _weights->info()->dimension(2));          // weights_depth

    IGCKernel::configure(win);
}

template <unsigned int kernel_size>
void GCDirectConvolutionLayerKernel<kernel_size>::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    _kernel.use();

    // Get initial windows
    Window slice  = window.first_slice_window_3D();
    Window win_in = window;

    win_in.adjust(Window::DimX, -_conv_pad_x, true);
    win_in.adjust(Window::DimY, -_conv_pad_y, true);
    win_in.set_dimension_step(Window::DimX, window.x().step() * _conv_stride_x);
    win_in.set_dimension_step(Window::DimY, window.y().step() * _conv_stride_y);

    Window slice_in = win_in.first_slice_window_3D();

    unsigned int idx1 = 2 * num_arguments_per_3D_tensor();
    add_3D_tensor_argument(idx1, _weights, 3, slice);

    if(_bias != nullptr)
    {
        Window slice_bias;
        slice_bias.use_tensor_dimensions(_bias->info()->tensor_shape());
        add_1D_tensor_argument(idx1, _bias, 4, slice_bias);
    }

    do
    {
        unsigned int idx = 0;

        add_3D_tensor_argument(idx, _input, 1, slice_in);
        add_3D_tensor_argument(idx, _output, 2, slice);

        _kernel.update_shader_params();
        enqueue(*this, slice, _lws);
    }
    while(window.slide_window_slice_3D(slice) && win_in.slide_window_slice_3D(slice_in));
}

template class arm_compute::GCDirectConvolutionLayerKernel<1>;
template class arm_compute::GCDirectConvolutionLayerKernel<3>;
template class arm_compute::GCDirectConvolutionLayerKernel<5>;
