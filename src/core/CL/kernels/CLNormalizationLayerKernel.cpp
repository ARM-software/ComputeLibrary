/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "src/core/CL/kernels/CLNormalizationLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Window.h"

#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/NormalizationHelpers.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, NormalizationLayerInfo norm_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NCHW, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!(norm_info.norm_size() % 2), "Normalization size should be odd");

    // Checks performed when output is configured
    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window>
validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, NormalizationLayerInfo norm_info)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, *input->clone());

    bool             window_changed = false;
    Window           win;
    const DataLayout data_layout = input->data_layout();
    if (data_layout == DataLayout::NCHW)
    {
        const unsigned int vec_size_x =
            adjust_vec_size(max_cl_vector_width / input->element_size(), input->dimension(0));
        const unsigned int norm_idx             = get_normalization_dimension_index(input->data_layout(), norm_info);
        const bool         is_norm_across_width = norm_idx == 0;

        const unsigned int norm_radius = norm_info.norm_size() / 2;
        // Border / padding calculation:
        // For NCHW no border handling is impelmeneted in the kernel in the x axis.
        // This means the x axis is fully-padded depending on vec_size_x and norm_size
        // E.G. for input x dimension = 3, norm_size = 3 (radius = 1), vec_size_x = 2 ('#' is element 'p' is padding):
        // In : |p|#|#|#|p|p|
        // Out:   |#|#|#|p|
        // The output has 1 right padding because of the vec_size_x.
        // The input has 1 left padding because radius = 1.
        // The input has 2 right padding because of radius = 1 AND because of the extra output padding
        const unsigned int border_width_left = is_norm_across_width ? norm_radius : 0;
        const unsigned int border_width_right =
            is_norm_across_width ? norm_radius + (vec_size_x - input->dimension(0) % vec_size_x) : 0;
        const BorderSize border_size = BorderSize(0, border_width_right, 0, border_width_left);

        win = calculate_max_window(*input, Steps(vec_size_x));

        // We do not use a Rectangle window for IN_MAP_2D as we clamp the top and bottom accesses inside the kernel, avoiding padding
        // Reads can occur within the valid region of the input
        if (is_norm_across_width)
        {
            AccessWindowStatic input_access(input, -border_size.left, 0, input->dimension(0) + border_size.right, 0);
            window_changed = window_changed || update_window_and_padding(win, input_access);
        }
        else
        {
            AccessWindowHorizontal input_access(input, -border_size.left, vec_size_x);
            window_changed = window_changed || update_window_and_padding(win, input_access);
        }

        AccessWindowHorizontal output_access(output, 0, vec_size_x);
        window_changed = window_changed || update_window_and_padding(win, output_access);
    }
    else
    {
        unsigned int vec_size_x = adjust_vec_size(max_cl_vector_width / input->element_size(), input->dimension(0));
        if (norm_info.is_cross_map())
        {
            vec_size_x = 1;
        }
        win = calculate_max_window(*input, Steps(vec_size_x));
    }
    Status err =
        (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLNormalizationLayerKernel::CLNormalizationLayerKernel()
    : _input(nullptr), _output(nullptr), _border_size(0), _is_norm_across_width(false)
{
    _type = CLKernelType::ELEMENTWISE;
}

BorderSize CLNormalizationLayerKernel::border_size() const
{
    return _border_size;
}

void CLNormalizationLayerKernel::configure(const ICLTensor *input, ICLTensor *output, NormalizationLayerInfo norm_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, norm_info);
}

void CLNormalizationLayerKernel::configure(const CLCompileContext &compile_context,
                                           const ICLTensor        *input,
                                           ICLTensor              *output,
                                           NormalizationLayerInfo  norm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    auto padding_info = get_padding_info({input, output});

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), norm_info));
    auto win_config = validate_and_configure_window(input->info(), output->info(), norm_info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    _input  = input;
    _output = output;

    const DataLayout data_layout = input->info()->data_layout();
    unsigned int     vec_size_x =
        adjust_vec_size(max_cl_vector_width / input->info()->element_size(), input->info()->dimension(0));
    int vec_size_x_leftovers = input->info()->dimension(0) % vec_size_x;
    if (norm_info.is_cross_map() && data_layout == DataLayout::NHWC)
    {
        vec_size_x           = 1;
        vec_size_x_leftovers = 0;
    }

    if (data_layout == DataLayout::NCHW)
    {
        const unsigned int norm_idx    = get_normalization_dimension_index(data_layout, norm_info);
        _is_norm_across_width          = norm_idx == 0;
        const unsigned int norm_radius = norm_info.norm_size() / 2;
        // Border / padding calculation:
        // For NCHW no border handling is impelmeneted in the kernel in the x axis.
        // This means the x axis is fully-padded depending on vec_size_x and norm_size
        // E.G. for input x dimension = 3, norm_size = 3 (radius = 1), vec_size_x = 2 ('#' is element 'p' is padding):
        // In : |p|#|#|#|p|p|
        // Out:   |#|#|#|p|
        // The output has 1 right padding because of the vec_size_x.
        // The input has 1 left padding because radius = 1.
        // The input has 2 right padding because of radius = 1 AND the extra output padding
        const unsigned int border_width_left = _is_norm_across_width ? norm_radius : 0;
        const unsigned int border_width_right =
            _is_norm_across_width ? norm_radius + (vec_size_x - input->info()->dimension(0) % vec_size_x) : 0;
        _border_size = BorderSize(0, border_width_right, 0, border_width_left);
    }

    const bool is_in_map_2D = (norm_info.type() == NormType::IN_MAP_2D);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.add_option(("-DCOEFF=" + float_to_string_with_full_precision(norm_info.scale_coeff())));
    build_opts.add_option(("-DBETA=" + float_to_string_with_full_precision(norm_info.beta())));
    build_opts.add_option(("-DKAPPA=" + float_to_string_with_full_precision(norm_info.kappa())));
    build_opts.add_option(("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x)));
    build_opts.add_option(("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_x_leftovers)));
    build_opts.add_option(("-DRADIUS=" + support::cpp11::to_string(norm_info.norm_size() / 2)));
    build_opts.add_option(("-DNUM_SLICES=" + support::cpp11::to_string(input->info()->dimension(2))));
    build_opts.add_option_if(is_in_map_2D, "-DIN_MAP_2D");
    build_opts.add_option_if(norm_info.is_in_map() || (data_layout == DataLayout::NHWC && norm_info.is_cross_map()),
                             "-DWIDTH_SIZE=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.add_option_if(norm_info.is_in_map() && data_layout == DataLayout::NHWC,
                             "-DDIM1_SIZE=" + support::cpp11::to_string(input->info()->dimension(1)));

    // Create kernel
    std::string kernel_name;
    if (norm_info.is_in_map())
    {
        kernel_name = "normalization_layer_in_map_" + lower_string(string_from_data_layout(data_layout));
    }
    else
    {
        kernel_name = "normalization_layer_cross_map_" + lower_string(string_from_data_layout(data_layout));
    }
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "normalization_layer_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(static_cast<std::underlying_type<NormType>::type>(norm_info.type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(norm_info.norm_size());
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    if (data_layout == DataLayout::NHWC)
    {
        ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
    }
}

Status CLNormalizationLayerKernel::validate(const ITensorInfo     *input,
                                            const ITensorInfo     *output,
                                            NormalizationLayerInfo norm_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, norm_info));
    ARM_COMPUTE_RETURN_ON_ERROR(
        validate_and_configure_window(input->clone().get(), output->clone().get(), norm_info).first);

    return Status{};
}

void CLNormalizationLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    const int collapsed_dimension = _is_norm_across_width ? Window::DimZ : 4;
    Window    window_collapsed    = window.collapse_if_possible(ICLKernel::window(), collapsed_dimension);
    Window    slice               = window_collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    } while (window_collapsed.slide_window_slice_3D(slice));
}
} // namespace arm_compute
