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
#include "arm_compute/core/CL/kernels/CLIm2ColKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <tuple>
#include <utility>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
struct Im2ColConfiguration
{
    std::string           kernel_name{};
    std::set<std::string> build_options{};
    unsigned int          num_elems_processed_per_iteration{};
    bool                  is_padding_required_nchw{};
};

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation,
                          unsigned int num_groups)
{
    const unsigned int channel_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::QASYMM8 && has_bias);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(num_groups == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::NHWC && num_groups > 1);
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(channel_idx) % num_groups) != 0);

    if(output->total_size() > 0)
    {
        const TensorInfo tensor_info_output = output->clone()->set_tensor_shape(compute_im2col_conv_shape(input, kernel_dims, conv_info, has_bias, dilation, num_groups == 1, num_groups));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation,
                                                        unsigned int num_elems_processed_per_iteration, bool is_padding_required_nchw, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto initialization if not yet initialized
    TensorShape expected_output_shape = compute_im2col_conv_shape(input, kernel_dims, conv_info, has_bias, dilation, num_groups == 1, num_groups);

    auto_init_if_empty(*output, input->clone()->set_tensor_shape(expected_output_shape));

    const DataLayout   data_layout  = input->data_layout();
    const unsigned int width_idx    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int input_width  = input->dimension(width_idx);
    const unsigned int input_height = input->dimension(height_idx);

    // Configure the execute window based on the selected optimal OpenCL kernel
    bool   window_changed = false;
    Window win;

    if(data_layout == DataLayout::NHWC)
    {
        win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

        const int xin_start = 0;
        const int xin_end   = input->dimension(0) < num_elems_processed_per_iteration ? ceil_to_multiple(input->dimension(0), num_elems_processed_per_iteration) : input->dimension(0);
        const int yin_start = 0;
        const int yin_end   = input->dimension(1);

        const int xout_start = 0;
        const int xout_end   = input->dimension(0) < num_elems_processed_per_iteration ? output->dimension(0) + (num_elems_processed_per_iteration - input->dimension(0)) : output->dimension(0);
        const int yout_start = 0;
        const int yout_end   = output->dimension(1);

        AccessWindowStatic input_access(input, xin_start, yin_start, xin_end, yin_end);
        AccessWindowStatic output_access(output, xout_start, yout_start, xout_end, yout_end);
        window_changed = window_changed || update_window_and_padding(win, input_access, output_access);
    }
    else
    {
        if(is_padding_required_nchw)
        {
            const BorderSize border(conv_info.pad_top(), conv_info.pad_right(), conv_info.pad_bottom(), conv_info.pad_left());
            win = calculate_max_window(*input,
                                       Steps(num_elems_processed_per_iteration * conv_info.stride().first, conv_info.stride().second));
            AccessWindowStatic input_access(input,
                                            -border.left,
                                            -border.top,
                                            ceil_to_multiple(input_width + border.right, kernel_dims.width * num_elems_processed_per_iteration),
                                            input_height + border.bottom);
            window_changed = window_changed || update_window_and_padding(win, input_access);
        }
        else
        {
            // For the generic case, CLIm2ColKernel doesn't need padding (we do not read out-of-bounds elements) so
            // update_window_and_padding() can be skipped
            win = calculate_max_window(*input, Steps());
        }
    }

    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));
    // set the Z dimension's step same size as the whole dimension so that one can't split across the Z dimension
    win.set_dimension_step(Window::DimZ, win[Window::DimZ].end() - win[Window::DimZ].start());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

Im2ColConfiguration configure_opencl_kernel(const ITensorInfo *input, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation, unsigned int num_groups)
{
    const DataLayout   data_layout   = input->data_layout();
    const DataType     data_type     = input->data_type();
    const unsigned int width_idx     = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    const unsigned int input_width   = input->dimension(width_idx);
    const unsigned int input_height  = input->dimension(height_idx);
    const unsigned int input_channel = input->dimension(channel_idx);

    const std::pair<unsigned int, unsigned int> convolved_dims = scaled_dimensions(input_width, input_height, kernel_dims.width, kernel_dims.height, conv_info, dilation);

    // Im2Col configuration
    std::string                   kernel_name = "im2col_generic_";
    CLBuildOptions                build_opts;
    unsigned int                  num_elems_processed_per_iteration = 1;
    bool                          is_padding_required_nchw          = false;
    const UniformQuantizationInfo qinfo                             = input->quantization_info().uniform();

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(input->element_size()));
    build_opts.add_option("-DKERNEL_WIDTH=" + support::cpp11::to_string(kernel_dims.width));
    build_opts.add_option("-DKERNEL_HEIGHT=" + support::cpp11::to_string(kernel_dims.height));
    build_opts.add_option("-DCONVOLVED_WIDTH=" + support::cpp11::to_string(convolved_dims.first));
    build_opts.add_option("-DCONVOLVED_HEIGHT=" + support::cpp11::to_string(convolved_dims.second));
    build_opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_info.stride().first));
    build_opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_info.stride().second));
    build_opts.add_option("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
    build_opts.add_option("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
    build_opts.add_option("-DPAD_RIGHT=" + support::cpp11::to_string(conv_info.pad_right()));
    build_opts.add_option("-DPAD_BOTTOM=" + support::cpp11::to_string(conv_info.pad_bottom()));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input_width));
    build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input_height));
    build_opts.add_option("-DSRC_DEPTH=" + support::cpp11::to_string(input_channel));
    build_opts.add_option("-DDILATION_X=" + support::cpp11::to_string(dilation.x()));
    build_opts.add_option("-DDILATION_Y=" + support::cpp11::to_string(dilation.y()));
    build_opts.add_option_if(num_groups > 1, "-DNUM_GROUPS=" + support::cpp11::to_string(num_groups));
    build_opts.add_option_if_else(is_data_type_quantized(data_type), "-DPAD_VALUE=" + support::cpp11::to_string(qinfo.offset), "-DPAD_VALUE=0");
    build_opts.add_option_if(has_bias, "-DHAS_BIAS");

    if(data_layout == DataLayout::NHWC)
    {
        num_elems_processed_per_iteration = 2;
        is_padding_required_nchw          = false;

        // Only the 3x3 and 9x9 cases are optimized for NHWC
        if(kernel_dims == Size2D(3U, 3U))
        {
            kernel_name = "im2col3x3_";
        }
        else if(kernel_dims == Size2D(9U, 9U))
        {
            kernel_name = "im2col9x9_";
        }

        build_opts.add_option("-DVECTOR_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
        build_opts.add_option("-DLAST_ACCESSED=" + support::cpp11::to_string(std::max(static_cast<int>(input_channel - num_elems_processed_per_iteration), 0)));
    }
    else
    {
        if(dilation == Size2D(1U, 1U))
        {
            const bool squared_im2col = kernel_dims.width == kernel_dims.height;
            if(squared_im2col)
            {
                // Check if we can run an optimized im2col for NCHW
                switch(kernel_dims.width)
                {
                    case 1:
                        // Optimized im2col1x1 if stride_x = 1 and conv_info.has_padding() = false
                        if(conv_info.stride().first == 1 && !conv_info.has_padding())
                        {
                            kernel_name                       = "im2col1x1_stridex1_";
                            num_elems_processed_per_iteration = 4;
                            is_padding_required_nchw          = true;
                        }
                        break;
                    case 3:
                        kernel_name                       = "im2col3x3_";
                        num_elems_processed_per_iteration = 1;
                        is_padding_required_nchw          = true;
                        break;
                    case 5:
                        kernel_name                       = "im2col5x5_";
                        num_elems_processed_per_iteration = 1;
                        is_padding_required_nchw          = true;
                        break;
                    case 11:
                        // Optimized im2col11x11 if pad_x = pad_y = 0
                        if(!conv_info.has_padding())
                        {
                            kernel_name                       = "im2col11x11_padx0_pady0_";
                            num_elems_processed_per_iteration = 1;
                            is_padding_required_nchw          = true;
                        }
                        break;
                    default:
                        kernel_name                       = "im2col_generic_";
                        num_elems_processed_per_iteration = 1;
                        is_padding_required_nchw          = false;
                        break;
                }
            }
            else if(kernel_dims.width > 1 && !conv_info.has_padding())
            {
                kernel_name                       = "im2col_generic_padx0_pady0_";
                num_elems_processed_per_iteration = 1;
                is_padding_required_nchw          = false;

                // Optimized im2col is performed using one or more vector operations with the specified vector size
                // and a remainder. For example, for 5x5 convolutions, im2col is performed using vectors of size 4
                // and scalars; for 7x7 convolutions, using vectors of size 4 and vectors of size 3.
                // Using the vector size of 4 is always safe since OpenCL supports vectors of size 2 and 3.
                // Using the vector size of 8, however, may be faster.
                // For 2x2 convolutions, use vectors of size 2. (For 3x3 convolutions, im2col_kernel3x3_padx0_pady0
                // is used instead.)
                const size_t vector_size           = std::min(static_cast<size_t>(4), kernel_dims.width);
                const size_t width_mod_vector_size = kernel_dims.width % vector_size;
                build_opts.add_option("-DVECTOR_SIZE=" + support::cpp11::to_string(vector_size));
                build_opts.add_option("-DWIDTH_MOD_VECTOR_SIZE=" + support::cpp11::to_string(width_mod_vector_size));
            }
        }
    }

    // Append the data layout to the kernel_name
    kernel_name += lower_string(string_from_data_layout(data_layout));

    Im2ColConfiguration im2col_config;
    im2col_config.kernel_name                       = kernel_name;
    im2col_config.build_options                     = build_opts.options();
    im2col_config.num_elems_processed_per_iteration = num_elems_processed_per_iteration;
    im2col_config.is_padding_required_nchw          = is_padding_required_nchw;

    return im2col_config;
}
} // namespace

CLIm2ColKernel::CLIm2ColKernel()
    : _input(nullptr), _output(nullptr), _data_layout(DataLayout::UNKNOWN), _convolved_dims(), _num_elems_processed_per_iteration(1), _kernel_dims(), _conv_info(), _num_groups()
{
}

void CLIm2ColKernel::configure(const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation,
                               unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), kernel_dims, conv_info, has_bias, dilation, num_groups));

    _data_layout = input->info()->data_layout();

    const unsigned int width_idx    = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx   = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int input_width  = input->info()->dimension(width_idx);
    const unsigned int input_height = input->info()->dimension(height_idx);

    // Select and configure the optimal OpenCL kernel to run.
    // This function returns the OpenCL kernel's name, the arguments to pass at compile time, the number of elements processed per iteration
    // and the padding requirement flag
    Im2ColConfiguration im2col_config = configure_opencl_kernel(input->info(), kernel_dims, conv_info, has_bias, dilation, num_groups);

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(im2col_config.kernel_name, im2col_config.build_options));

    _input                             = input;
    _output                            = output;
    _convolved_dims                    = scaled_dimensions(input_width, input_height, kernel_dims.width, kernel_dims.height, conv_info, dilation);
    _num_elems_processed_per_iteration = im2col_config.num_elems_processed_per_iteration;
    _kernel_dims                       = kernel_dims; // Only needed by the Tuner
    _conv_info                         = conv_info;   // Only needed by the Tuner
    _num_groups                        = num_groups;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), kernel_dims, conv_info, has_bias, dilation, im2col_config.num_elems_processed_per_iteration,
                                                    im2col_config.is_padding_required_nchw, num_groups);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = im2col_config.kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(num_groups);
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(_data_layout));
}

Status CLIm2ColKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation,
                                unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, kernel_dims, conv_info, has_bias, dilation, num_groups));
    Im2ColConfiguration im2col_config = configure_opencl_kernel(input, kernel_dims, conv_info, has_bias, dilation, num_groups);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), kernel_dims, conv_info, has_bias, dilation, im2col_config.num_elems_processed_per_iteration,
                                                              im2col_config.is_padding_required_nchw, num_groups)
                                .first);
    return Status{};
}

void CLIm2ColKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    // Get initial windows
    // Collapse in order to have (SRC_DEPTH * BATCH_SIZE) on the 3rd dimension
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    window_collapsed.set_dimension_step(Window::DimZ, 1);

    Window window_output;
    window_output.use_tensor_dimensions(_output->info()->tensor_shape());

    const Window first_slice_3d = window_collapsed.first_slice_window_3D();

    Window slice     = first_slice_3d;
    Window slice_in  = first_slice_3d;
    Window slice_out = window_output.first_slice_window_2D();

    if(_data_layout == DataLayout::NHWC)
    {
        const Window tmp_win     = window.collapse_if_possible(ICLKernel::window(), 3);
        const int    num_batches = tmp_win[3].end();

        slice.set(1, Window::Dimension(0, static_cast<int>(_output->info()->tensor_shape()[1]), 1));
        slice.set(2, Window::Dimension(0, static_cast<int>(num_batches), 1));
    }
    else
    {
        slice.set(0, Window::Dimension(0, static_cast<int>(ceil_to_multiple(_convolved_dims.first, _num_elems_processed_per_iteration)), _num_elems_processed_per_iteration));
        slice.set(1, Window::Dimension(0, static_cast<int>(_convolved_dims.second), 1));
        // Note: In case of NCHW the 3rd dimension is already set collapsing the input window
    }

    // Setup input slice
    // The dimensions of the input are increased within the OpenCL kernel
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Setup output slice
    // The dimensions of the output are increased within the OpenCL kernel
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    unsigned int idx = num_arguments_per_3D_tensor() + (_num_groups == 1 ? num_arguments_per_2D_tensor() : num_arguments_per_3D_tensor());
    _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input->info()->strides_in_bytes()[3]));
    _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_output->info()->strides_in_bytes()[((_num_groups == 1) ? 2 : 3)]));
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        if(_num_groups == 1)
        {
            add_2D_tensor_argument(idx, _output, slice_out);
        }
        else
        {
            add_3D_tensor_argument(idx, _output, slice_out);
        }
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window_collapsed.slide_window_slice_3D(slice) && window_output.slide_window_slice_2D(slice_out) && window_collapsed.slide_window_slice_3D(slice_in));
}
