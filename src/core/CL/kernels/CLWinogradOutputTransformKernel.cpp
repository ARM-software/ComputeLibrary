/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLWinogradOutputTransformKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

#include <cmath>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    if(act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QASYMM8, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((input->data_type() == DataType::QASYMM8) && (act_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
                                        && (act_info.activation() != ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
                                        && (act_info.activation() != ActivationLayerInfo::ActivationFunction::RELU)
                                        && (act_info.activation() != ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                        "For QASYMM8 only logistic, relu, lower bounded relu and lower-upper bounded relu are supported");
    }
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);

    ARM_COMPUTE_RETURN_ERROR_ON(output->data_layout() != winograd_info.output_data_layout);

    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const Size2D        kernel_size      = winograd_info.kernel_size;
    const Size2D        input_dimensions = winograd_info.input_dimensions;
    const unsigned int  num_channels     = (winograd_info.kernel_size.width + winograd_info.output_tile_size.width - 1) * (winograd_info.kernel_size.height + winograd_info.output_tile_size.height - 1);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!cl_winograd_convolution_layer_supported(output_tile_size, kernel_size, winograd_info.output_data_layout), "Winograd output transform not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->dimension(2) != num_channels, "Wrong number of channels");

    // Compute number of elements to process in the X and Y direction
    // Compute the number of output tiles along the x and y direction of size "output_tile_size"
    const Size2D num_tiles = compute_winograd_convolution_tiles(input_dimensions,
                                                                kernel_size,
                                                                output_tile_size,
                                                                conv_info);

    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(1) != static_cast<unsigned int>((num_tiles.area())));

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != bias->dimension(0));
    }

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info_output = input->clone()->set_tensor_shape(compute_winograd_output_transform_shape(*input, winograd_info));

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *bias, ITensorInfo *output, const Size2D &output_tile_size)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool   window_changed = false;

    int output_static_window_end_x = 0;
    int output_static_window_end_y = 0;

    if(output->data_layout() == DataLayout::NCHW)
    {
        output_static_window_end_x = ceil_to_multiple(output->dimension(0), output_tile_size.width);
        output_static_window_end_y = ceil_to_multiple(output->dimension(1), output_tile_size.height);
    }
    else
    {
        output_static_window_end_x = output->dimension(0);
        output_static_window_end_y = std::max(ceil_to_multiple(output->dimension(1), output_tile_size.width), output->dimension(1) + 1 /* For out of bound reads towards the z axis */);
    }

    AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration, num_elems_processed_per_iteration);
    AccessWindowStatic    output_access(output, 0, 0, output_static_window_end_x, output_static_window_end_y);
    window_changed = update_window_and_padding(win, input_access, output_access);
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    if(bias != nullptr)
    {
        AccessWindowStatic bias_access(bias, 0, 0, bias->dimension(0), bias->dimension(1));
        window_changed = window_changed || update_window_and_padding(win, bias_access);
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLWinogradOutputTransformKernel::CLWinogradOutputTransformKernel()
    : _input(nullptr), _bias(nullptr), _output(nullptr), _is_nhwc(false)
{
}

void CLWinogradOutputTransformKernel::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(compute_winograd_output_transform_shape(*input->info(), winograd_info)));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (bias != nullptr ? bias->info() : nullptr), output->info(), winograd_info, act_info));

    _input   = input;
    _bias    = bias;
    _output  = output;
    _is_nhwc = winograd_info.output_data_layout == DataLayout::NHWC;

    // Compute num_tiles_x
    const Size2D        input_dimensions = winograd_info.input_dimensions;
    const Size2D        kernel_size      = winograd_info.kernel_size;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const PadStrideInfo conv_info        = winograd_info.convolution_info;

    // Compute the number of output tiles along the x and y direction of size "output_tile_size"
    const Size2D num_tiles = compute_winograd_convolution_tiles(input_dimensions,
                                                                kernel_size,
                                                                output_tile_size,
                                                                conv_info);
    const size_t total_batches = output->info()->tensor_shape().total_size_upper(3);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option_if(act_info.enabled(), "-DFUSED_ACTIVATION=" + lower_string(string_from_activation_func(act_info.activation())));
    build_opts.add_option_if(act_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(act_info.a()));
    build_opts.add_option_if(act_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(act_info.b()));

    if((output_tile_size.x() == 2) || (output_tile_size.x() == 1 && output_tile_size.y() == 2))
    {
        build_opts.add_option("-DVEC_SIZE=2");
    }
    else if((output_tile_size.x() == 4) || (output_tile_size.x() == 1 && output_tile_size.y() == 4))
    {
        build_opts.add_option("-DVEC_SIZE=4");
    }

    build_opts.add_option_if(act_info.enabled(), "-DSELECT_DATA_TYPE=" + get_cl_select_type_from_data_type(input->info()->data_type()));

    build_opts.add_option_if(_bias != nullptr, std::string("-DHAS_BIAS"));
    build_opts.add_option("-DNUM_TILES_X=" + support::cpp11::to_string(num_tiles.width));
    build_opts.add_option("-DOUTPUT_TILE_W=" + support::cpp11::to_string(output_tile_size.width));
    build_opts.add_option("-DOUTPUT_TILE_H=" + support::cpp11::to_string(output_tile_size.height));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option_if(total_batches > 1, "-DSRC_DEPTH=" + support::cpp11::to_string(_input->info()->dimension(2)));
    build_opts.add_option_if(winograd_info.kernel_size.height == 1, "-DWINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL");
    build_opts.add_option_if(winograd_info.kernel_size.width == 1, "-DWINOGRAD_OUTPUT_TRANSFORM_VERTICAL");

    // Create kernel
    std::string kernel_name = "winograd_output_transform_" + output_tile_size.to_string() + "_" + kernel_size.to_string() + "_" + lower_string(string_from_data_layout(winograd_info.output_data_layout));
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (bias != nullptr ? bias->info() : nullptr), output->info(), winograd_info.output_tile_size);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(winograd_info.output_data_layout));
}

Status CLWinogradOutputTransformKernel::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, (bias != nullptr ? bias->clone().get() : nullptr), output, winograd_info, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (bias != nullptr ? bias->clone().get() : nullptr), output->clone().get(), winograd_info.output_tile_size).first);

    return Status{};
}

void CLWinogradOutputTransformKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    // Collapse window
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

    // Get initial windows
    Window slice = window_collapsed.first_slice_window_4D();
    slice.set(Window::DimZ, Window::Dimension(0, 1, 1));

    // Setup output slice
    Window slice_out(slice);
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    if(_bias != nullptr)
    {
        unsigned int idx1 = 2 * num_arguments_per_4D_tensor();
        Window       slice_biases;
        slice_biases.use_tensor_dimensions(_bias->info()->tensor_shape());
        add_1D_tensor_argument(idx1, _bias, slice_biases);
    }

    if(_is_nhwc)
    {
        unsigned int idx2 = 2 * num_arguments_per_4D_tensor() + ((_bias != nullptr) ? num_arguments_per_1D_tensor() : 0);
        _kernel.setArg(idx2, static_cast<int>(_output->info()->total_size() - _output->info()->strides_in_bytes().y()));
    }

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, _input, slice);
        add_4D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice) && window.slide_window_slice_3D(slice_out));
}
