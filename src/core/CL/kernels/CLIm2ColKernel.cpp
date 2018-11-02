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
#include "arm_compute/core/CL/kernels/CLIm2ColKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <tuple>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, bool has_bias, const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::QASYMM8 && has_bias);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    }

    return Status{};
}
} // namespace

CLIm2ColKernel::CLIm2ColKernel()
    : _input(nullptr), _output(nullptr), _convolved_dims(), _num_elems_processed_per_iteration(1), _run_func(nullptr), _kernel_dims()
{
}

void CLIm2ColKernel::configure(const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), has_bias, dilation));

    _input       = input;
    _output      = output;
    _kernel_dims = kernel_dims;

    const DataType  data_type  = input->info()->data_type();
    const GPUTarget gpu_target = get_target();

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option(("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type)));
    build_opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(input->info()->element_size()));
    build_opts.add_option_if(has_bias, "-DHAS_BIAS");
    build_opts.add_option_if(is_data_type_fixed_point(data_type), "-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));

    int stride_x = 0;
    int stride_y = 0;

    std::tie(stride_x, stride_y) = conv_info.stride();

    const bool run_img2col_reduced = (output->info()->dimension(0) == (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))) && (TensorShape::num_max_dimensions >= 4)
                                     && (std::equal(input->info()->tensor_shape().cbegin() + 3,
                                                    input->info()->tensor_shape().cend(),
                                                    output->info()->tensor_shape().cbegin() + 1))
                                     && ((stride_x == 1) && (stride_y == 1) && !conv_info.has_padding());

    bool is_optimized_path = false;

    _num_elems_processed_per_iteration = 1;

    std::string kernel_name;
    if(!run_img2col_reduced)
    {
        // Default kernel name
        kernel_name = "im2col_generic_dchw";

        _convolved_dims = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1),
                                            kernel_dims.width, kernel_dims.height,
                                            conv_info, dilation);

        build_opts.add_option("-DKERNEL_WIDTH=" + support::cpp11::to_string(kernel_dims.width));
        build_opts.add_option("-DKERNEL_HEIGHT=" + support::cpp11::to_string(kernel_dims.height));
        build_opts.add_option("-DKERNEL_DEPTH=" + support::cpp11::to_string(input->info()->dimension(2)));
        build_opts.add_option("-DCONVOLVED_WIDTH=" + support::cpp11::to_string(_convolved_dims.first));
        build_opts.add_option("-DCONVOLVED_HEIGHT=" + support::cpp11::to_string(_convolved_dims.second));
        build_opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_info.stride().first));
        build_opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_info.stride().second));
        build_opts.add_option("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
        build_opts.add_option("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
        build_opts.add_option("-DPAD_RIGHT=" + support::cpp11::to_string(conv_info.pad_right()));
        build_opts.add_option("-DPAD_BOTTOM=" + support::cpp11::to_string(conv_info.pad_bottom()));
        build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
        build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(1)));
        build_opts.add_option("-DDILATION_X=" + support::cpp11::to_string(dilation.x()));
        build_opts.add_option("-DDILATION_Y=" + support::cpp11::to_string(dilation.y()));
        build_opts.add_option_if_else(is_data_type_quantized(data_type), "-DPAD_VALUE=" + support::cpp11::to_string(input->info()->quantization_info().offset), "-DPAD_VALUE=0");

        const bool squared_im2col = kernel_dims.width == kernel_dims.height;

        if(dilation == Size2D(1U, 1U))
        {
            if(squared_im2col && !is_data_type_fixed_point(data_type))
            {
                // Check if we can run an optimized im2col
                switch(kernel_dims.width)
                {
                    case 1:
                        // Optimized im2col1x1 if stride_x = 1 and conv_info.has_padding() = false
                        if(conv_info.stride().first == 1 && !conv_info.has_padding())
                        {
                            // Set hint for LWS
                            _lws_hint                          = cl::NDRange(1, 1, 8);
                            _num_elems_processed_per_iteration = 4;
                            is_optimized_path                  = true;
                            kernel_name                        = "im2col1x1_stridex1_dchw";
                        }
                        break;
                    case 3:
                        _lws_hint                          = cl::NDRange(1, 1, 8);
                        _num_elems_processed_per_iteration = 1;
                        is_optimized_path                  = true;
                        kernel_name                        = "im2col3x3_dchw";
                        break;
                    case 5:
                        _num_elems_processed_per_iteration = 1;
                        is_optimized_path                  = true;
                        kernel_name                        = "im2col5x5_dchw";
                        break;
                    case 11:
                        // Optimized im2col11x11 if pad_x = pad_y = 0
                        if(!conv_info.has_padding())
                        {
                            _num_elems_processed_per_iteration = 1;
                            is_optimized_path                  = true;
                            kernel_name                        = "im2col11x11_padx0_pady0_dchw";
                        }
                        break;
                    default:
                        is_optimized_path = false;
                        break;
                }
            }
            else if(kernel_dims.width > 1 && !conv_info.has_padding())
            {
                _num_elems_processed_per_iteration = 1;
                kernel_name                        = "im2col_generic_padx0_pady0_dchw";

                // Optimized im2col is performed using one or more vector operations with the specified vector size
                // and a remainder. For example, for 5x5 convolutions, im2col is performed using vectors of size 4
                // and scalars; for 7x7 convolutions, using vectors of size 4 and vectors of size 3.
                // Using the vector size of 4 is always safe since OpenCL supports vectors of size 2 and 3.
                // Using the vector size of 8, however, may be faster.
                size_t vector_size = 4;
                // For 2x2 convolutions, use vectors of size 2. (For 3x3 convolutions, im2col_kernel3x3_padx0_pady0
                // is used instead.)
                if(kernel_dims.width < vector_size)
                {
                    vector_size = kernel_dims.width;
                }
                // Local work size and vector size optimized for the 11x11 AlexNet convolution on Bifrost.
                if(gpu_target_is_in(gpu_target, GPUTarget::G71, GPUTarget::G72, GPUTarget::G51, GPUTarget::G51BIG, GPUTarget::G51LIT, GPUTarget::TNOX) && kernel_dims.width == 11)
                {
                    _lws_hint   = cl::NDRange(1, 1, 1);
                    vector_size = 8;
                }
                const size_t width_mod_vector_size = kernel_dims.width % vector_size;
                build_opts.add_option("-DVECTOR_SIZE=" + support::cpp11::to_string(vector_size));
                build_opts.add_option("-DWIDTH_MOD_VECTOR_SIZE=" + support::cpp11::to_string(width_mod_vector_size));
            }
        }
        _run_func = &CLIm2ColKernel::run_generic;
    }
    else
    {
        _num_elems_processed_per_iteration = 1;
        kernel_name                        = "im2col_reduced_dchw";
        _run_func                          = &CLIm2ColKernel::run_reduced;
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Configure kernel window
    Window win;
    if(is_optimized_path)
    {
        win = calculate_max_window(*input->info(),
                                   Steps(_num_elems_processed_per_iteration),
                                   false,
                                   BorderSize(conv_info.pad_top(), conv_info.pad_right(), conv_info.pad_bottom(), conv_info.pad_left()));

        const int x = -conv_info.pad_left();
        const int y = -conv_info.pad_top();
        const int w = kernel_dims.width * _num_elems_processed_per_iteration;
        const int h = kernel_dims.height;

        AccessWindowRectangle input_access(input->info(), x, y, w, h);

        update_window_and_padding(win, input_access);
    }
    else
    {
        // For the generic case, CLIm2ColKernel doesn't need padding (we do not read out-of-bounds elements) so
        // update_window_and_padding() can be skipped
        win = calculate_max_window(*input->info(), Steps());
    }

    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));
    if(!run_img2col_reduced)
    {
        // set the Z dimension's step same size as the whole dimension so that one can't split across the Z dimension
        win.set_dimension_step(Window::DimZ, win[Window::DimZ].end() - win[Window::DimZ].start());
    }

    ICLKernel::configure(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

Status CLIm2ColKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation)
{
    ARM_COMPUTE_UNUSED(kernel_dims);
    ARM_COMPUTE_UNUSED(conv_info);
    ARM_COMPUTE_UNUSED(has_bias);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, has_bias, dilation));
    return Status{};
}

void CLIm2ColKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON(_run_func == nullptr);
    (this->*_run_func)(window, queue);
}

void CLIm2ColKernel::run_generic(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    // Get initial windows
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    // Change the Z dimension's step back to 1
    window_collapsed.set_dimension_step(Window::DimZ, 1);

    Window slice     = window_collapsed.first_slice_window_3D();
    Window slice_in  = window_collapsed.first_slice_window_3D();
    Window slice_out = window_collapsed.first_slice_window_3D();

    // Setup slice if stride_x != 0 or stride_y != 0
    if(_convolved_dims.first != _input->info()->dimension(0) || _convolved_dims.second != _input->info()->dimension(1))
    {
        // If the stride_x or stride_y are not 1, the output tensor of matrix multiply (Convolved tensor) will not
        // have the same shape of the im2col input tensor
        // In this case we need to re-compute the window using the shape of the tensor after matrix multiply (convolved_dims)
        slice.set(Window::DimX, Window::Dimension(0, static_cast<int>(_convolved_dims.first), 1));
        slice.set(Window::DimY, Window::Dimension(0, static_cast<int>(_convolved_dims.second), 1));
    }

    // Setup input slice
    // The first three dimensions of the input are increased by the inner loops
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Setup output slice
    slice_out.set(Window::DimX, Window::Dimension(0, _output->info()->dimension(0), _kernel_dims.area()));
    slice_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    slice_out.set(Window::DimZ, Window::Dimension(0, 1, 1));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input->info()->strides_in_bytes()[3]));
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_output->info()->strides_in_bytes()[3]));
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window_collapsed.slide_window_slice_3D(slice) && window_collapsed.slide_window_slice_3D(slice_out) && window_collapsed.slide_window_slice_3D(slice_in));
}

void CLIm2ColKernel::run_reduced(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window out_window;
    out_window.use_tensor_dimensions(_output->info()->tensor_shape());

    Window out_slice = out_window.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_3D();

    // Run kernel
    do
    {
        // Set arguments
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, in_slice);
        add_1D_tensor_argument(idx, _output, out_slice);

        _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(0));
        _kernel.setArg<cl_uint>(idx++, _input->info()->dimension(1));
        enqueue(queue, *this, in_slice, _lws_hint);
    }
    while(window.slide_window_slice_3D(in_slice) && out_window.slide_window_slice_1D(out_slice));
}
