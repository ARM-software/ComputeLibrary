/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#include "src/gpu/cl/kernels/ClWinogradOutputTransformKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

#include <cmath>

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
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
    ARM_COMPUTE_UNUSED(bias);

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool   window_changed = false;

    if(output->data_layout() == DataLayout::NCHW)
    {
        const int output_static_window_end_x = ceil_to_multiple(output->dimension(0), output_tile_size.width);
        const int output_static_window_end_y = ceil_to_multiple(output->dimension(1), output_tile_size.height);

        AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration, num_elems_processed_per_iteration);
        AccessWindowStatic    output_access(output, 0, 0, output_static_window_end_x, output_static_window_end_y);
        window_changed = update_window_and_padding(win, input_access, output_access);
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

ClWinogradOutputTransformKernel::ClWinogradOutputTransformKernel()
{
    _type = CLKernelType::WINOGRAD;
}

void ClWinogradOutputTransformKernel::configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *bias, ITensorInfo *dst, const WinogradInfo &winograd_info,
                                                const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_winograd_output_transform_shape(*src, winograd_info)));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, bias, dst, winograd_info, act_info));

    // Configure kernel window
    auto win_config = validate_and_configure_window(src, bias, dst, winograd_info.output_tile_size);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    IClKernel::configure_internal(win_config.second);

    auto padding_info = get_padding_info({ src, bias, dst });

    _is_nhwc = winograd_info.output_data_layout == DataLayout::NHWC;

    // Compute num_tiles_x
    const Size2D        input_dimensions = winograd_info.input_dimensions;
    const Size2D        kernel_size      = winograd_info.kernel_size;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const int           idx_width        = get_data_layout_dimension_index(winograd_info.output_data_layout, DataLayoutDimension::WIDTH);
    const int           idx_height       = get_data_layout_dimension_index(winograd_info.output_data_layout, DataLayoutDimension::HEIGHT);

    // Compute the number of output tiles along the x and y direction of size "output_tile_size"
    const Size2D num_tiles = compute_winograd_convolution_tiles(input_dimensions,
                                                                kernel_size,
                                                                output_tile_size,
                                                                conv_info);
    const size_t total_batches = dst->tensor_shape().total_size_upper(3);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(act_info.activation())));
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

    if(_is_nhwc)
    {
        build_opts.add_option_if(bias != nullptr, std::string("-DHAS_BIAS"));
        build_opts.add_option("-cl-fast-relaxed-math");
        build_opts.add_option("-DN0=" + support::cpp11::to_string(win_config.second.x().step()));
        build_opts.add_option("-DOUTPUT_TILE_W=" + support::cpp11::to_string(output_tile_size.width));
        build_opts.add_option("-DOUTPUT_TILE_H=" + support::cpp11::to_string(output_tile_size.height));
        build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
        build_opts.add_option_if(total_batches > 1, "-DSRC_DEPTH=" + support::cpp11::to_string(src->dimension(2)));
        build_opts.add_option_if(winograd_info.kernel_size.height == 1, "-DWINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL");
        build_opts.add_option_if(winograd_info.kernel_size.width == 1, "-DWINOGRAD_OUTPUT_TRANSFORM_VERTICAL");
        build_opts.add_option("-DNUM_TILES_X=" + support::cpp11::to_string(_num_tiles_x));
    }
    else
    {
        build_opts.add_option_if(bias != nullptr, std::string("-DHAS_BIAS"));
        build_opts.add_option("-cl-fast-relaxed-math");
        build_opts.add_option("-DN0=" + support::cpp11::to_string(win_config.second.x().step()));
        build_opts.add_option("-DNUM_TILES_X=" + support::cpp11::to_string(num_tiles.width));
        build_opts.add_option("-DOUTPUT_TILE_W=" + support::cpp11::to_string(output_tile_size.width));
        build_opts.add_option("-DOUTPUT_TILE_H=" + support::cpp11::to_string(output_tile_size.height));
        build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
        build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(src->dimension(1)));
        build_opts.add_option("-DDST_WIDTH=" + support::cpp11::to_string(dst->dimension(idx_width)));
        build_opts.add_option("-DDST_HEIGHT=" + support::cpp11::to_string(dst->dimension(idx_height)));
        build_opts.add_option_if(total_batches > 1, "-DSRC_DEPTH=" + support::cpp11::to_string(src->dimension(2)));
        build_opts.add_option_if(winograd_info.kernel_size.height == 1, "-DWINOGRAD_OUTPUT_TRANSFORM_HORIZONTAL");
        build_opts.add_option_if(winograd_info.kernel_size.width == 1, "-DWINOGRAD_OUTPUT_TRANSFORM_VERTICAL");
    }

    // Storing tensor dimensions to be sent later as kernel arguments
    _src_height  = src->dimension(1);
    _dst_width   = dst->dimension(idx_width);
    _dst_height  = dst->dimension(idx_height);
    _num_tiles_x = num_tiles.width;

    // Create kernel
    std::string kernel_name = "winograd_output_transform_" + output_tile_size.to_string() + "_" + kernel_size.to_string() + "_" + lower_string(string_from_data_layout(winograd_info.output_data_layout));

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(src->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(1));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(winograd_info.output_data_layout));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info) && _is_nhwc);
}

Status ClWinogradOutputTransformKernel::validate(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, (bias != nullptr ? bias->clone().get() : nullptr), dst, winograd_info, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), (bias != nullptr ? bias->clone().get() : nullptr), dst->clone().get(), winograd_info.output_tile_size).first);
    return Status{};
}

void ClWinogradOutputTransformKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IClKernel::window(), window);

    auto src  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    auto bias = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto dst  = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    // Collapse window
    Window window_collapsed = window.collapse_if_possible(IClKernel::window(), Window::DimZ);

    // Get initial windows
    Window slice = window_collapsed.first_slice_window_4D();
    slice.set(Window::DimZ, Window::Dimension(0, 1, 1));

    // Setup output slice
    Window slice_out(slice);
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    if(bias != nullptr)
    {
        unsigned int idx1 = 2 * num_arguments_per_4D_tensor();
        Window       slice_biases;
        slice_biases.use_tensor_dimensions(bias->info()->tensor_shape());
        add_1D_tensor_argument(idx1, bias, slice_biases);
    }

    if(_is_nhwc)
    {
        unsigned int idx2 = 2 * num_arguments_per_4D_tensor() + ((bias != nullptr) ? num_arguments_per_1D_tensor() : 0);
        _kernel.setArg(idx2++, static_cast<int>(dst->info()->total_size() - dst->info()->strides_in_bytes().y()));
        _kernel.setArg<cl_int>(idx2++, _src_height);
        _kernel.setArg<cl_int>(idx2++, _dst_width);
        _kernel.setArg<cl_int>(idx2++, _dst_height);
    }

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, src, slice);
        add_4D_tensor_argument(idx, dst, slice_out);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice) && window.slide_window_slice_3D(slice_out));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
