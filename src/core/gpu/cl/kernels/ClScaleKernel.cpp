/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "src/core/gpu/cl/kernels/ClScaleKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/ScaleUtils.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
inline std::pair<float, float> calculate_scale_factors(const ITensorInfo *src, const ITensorInfo *dst, DataLayout data_layout, bool align_corners)
{
    const int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Compute the ratio between source width/height and destination width/height
    const unsigned int src_width  = src->dimension(idx_width);
    const unsigned int src_height = src->dimension(idx_height);
    const unsigned int dst_width  = dst->dimension(idx_width);
    const unsigned int dst_height = dst->dimension(idx_height);

    float wr = arm_compute::scale_utils::calculate_resize_ratio(src_width, dst_width, align_corners);
    float hr = arm_compute::scale_utils::calculate_resize_ratio(src_height, dst_height, align_corners);

    return std::make_pair(wr, hr);
}

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::U8, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(dst == src);
    ARM_COMPUTE_RETURN_ERROR_ON(info.align_corners && !arm_compute::scale_utils::is_align_corners_allowed_sampling_policy(info.sampling_policy));

    float            wr          = 0.f;
    float            hr          = 0.f;
    const DataLayout data_layout = info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : info.data_layout;
    std::tie(wr, hr) = calculate_scale_factors(src, dst, data_layout, info.align_corners);

    ARM_COMPUTE_RETURN_ERROR_ON(info.interpolation_policy == InterpolationPolicy::AREA && (wr > 1.f || hr > 1.f));

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src, ITensorInfo *dst, const ScaleKernelInfo &info, BorderSize &border)
{
    Window           win{};
    bool             window_changed{};
    unsigned int     num_elems_processed_per_iteration = 0;
    const DataLayout data_layout                       = info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : info.data_layout;

    switch(data_layout)
    {
        case DataLayout::NCHW:
        {
            if(info.border_mode == BorderMode::UNDEFINED)
            {
                border = BorderSize(0);
            }

            num_elems_processed_per_iteration = 4;
            // Configure kernel window
            win = calculate_max_window(*dst, Steps(num_elems_processed_per_iteration));
            AccessWindowStatic input_access(src,
                                            -border.left, -border.top,
                                            src->dimension(0) + border.right,
                                            src->dimension(1) + border.bottom);
            AccessWindowHorizontal output_access(dst, 0, num_elems_processed_per_iteration);

            output_access.set_valid_region(win, calculate_valid_region_scale(*src,
                                                                             dst->tensor_shape(),
                                                                             info.interpolation_policy,
                                                                             info.sampling_policy,
                                                                             info.border_mode == BorderMode::UNDEFINED));

            window_changed = update_window_and_padding(win, input_access, output_access);
        }
        break;
        case DataLayout::NHWC:
        {
            // Configure kernel window
            win = calculate_max_window(*dst, Steps());
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Data layout not supported");
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

BorderSize ClScaleKernel::border_size() const
{
    return BorderSize(static_cast<size_t>(_data_layout == DataLayout::NCHW));
}

Status ClScaleKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, info));
    const DataLayout data_layout = info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : info.data_layout;
    BorderSize       border      = BorderSize(static_cast<size_t>(data_layout == DataLayout::NCHW));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), dst->clone().get(), info, border).first);

    return Status{};
}

void ClScaleKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, info));
    auto padding_info = get_padding_info({ src, dst });

    // Info required for the static tuning
    _info        = info;
    _data_type   = src->data_type();
    _data_layout = _info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : _info.data_layout;

    float wr = 0.f;
    float hr = 0.f;
    std::tie(wr, hr) = calculate_scale_factors(src, dst, _data_layout, _info.align_corners);
    const bool call_quantized_kernel = is_data_type_quantized_asymmetric(src->data_type()) && _info.interpolation_policy == InterpolationPolicy::BILINEAR;

    // Compute actual border size
    BorderSize border  = border_size();
    const bool is_nhwc = _data_layout == DataLayout::NHWC;

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    auto interpolation_policy_to_use = _info.interpolation_policy;
    if(_info.interpolation_policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f)
    {
        interpolation_policy_to_use = InterpolationPolicy::NEAREST_NEIGHBOR;
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(src, dst, _info, border);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
    build_opts.add_option("-DCONSTANT_VALUE=" + string_from_pixel_value(info.constant_border_value, src->data_type()));
    build_opts.add_option("-DBORDER_SIZE=" + support::cpp11::to_string(border.right));
    build_opts.add_option_if(info.border_mode == BorderMode::REPLICATE, "-DBORDER_MODE_REPLICATE");
    build_opts.add_option_if(is_nhwc, "-DDEPTH_OUT=" + support::cpp11::to_string(dst->dimension(2)));
    build_opts.add_option_if_else(_info.sampling_policy == SamplingPolicy::CENTER, "-DSAMPLING_POLICY_CENTER", "-DSAMPLING_POLICY_TOP_LEFT");
    build_opts.add_option_if(info.align_corners, "-DALIGN_CORNERS");
    if(call_quantized_kernel)
    {
        const UniformQuantizationInfo qinfo = src->quantization_info().uniform();
        build_opts.add_option("-DSCALE=" + support::cpp11::to_string(qinfo.scale));
        build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(qinfo.offset));
    }
    std::string interpolation_name = string_from_interpolation_policy(interpolation_policy_to_use);
    std::transform(interpolation_name.begin(), interpolation_name.end(), interpolation_name.begin(), ::tolower);
    std::string kernel_name = "scale_" + interpolation_name;
    kernel_name += call_quantized_kernel ? "_quantized_" : "_";
    kernel_name += lower_string(string_from_data_layout(_data_layout));

    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());
    if(is_nhwc)
    {
        ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
    }

    const int          idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int          idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    unsigned int       idx        = is_nhwc ? 2 * num_arguments_per_4D_tensor() : 2 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    const unsigned int src_width  = src->dimension(idx_width);
    const unsigned int dst_height = src->dimension(idx_height);

    _kernel.setArg<float>(idx++, src_width);
    _kernel.setArg<float>(idx++, dst_height);
    _kernel.setArg<float>(idx++, wr);
    _kernel.setArg<float>(idx++, hr);

    // Set to enable static tuning
    _output_x_dim = dst->dimension(0);

    // Set config_id for enabling LWS tuning
    _config_id = "scale_";
    _config_id += (_info.border_mode == BorderMode::REPLICATE ? "Bord_rep" : "");
    _config_id += (_info.sampling_policy == SamplingPolicy::CENTER ? "center" : "topleft");
    _config_id += (is_nhwc ? "nhwc" : "nchw");
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(3));
}

void ClScaleKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            Window slice = window.first_slice_window_2D();

            do
            {
                unsigned int idx = 0;
                add_2D_tensor_argument(idx, src, slice);
                add_2D_tensor_argument(idx, dst, slice);
                enqueue(queue, *this, slice, lws_hint());
            }
            while(window.slide_window_slice_2D(slice));
            break;
        }
        case DataLayout::NHWC:
        {
            Window collapsed = window.collapse(ICLKernel::window(), Window::DimZ);
            Window slice     = collapsed.first_slice_window_4D();

            unsigned int idx = 0;
            add_4D_tensor_argument(idx, src, slice);
            add_4D_tensor_argument(idx, dst, slice);
            enqueue(queue, *this, slice, lws_hint());
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Data layout not supported");
    }
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
