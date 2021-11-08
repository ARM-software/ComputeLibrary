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
#include "src/gpu/cl/kernels/ClScaleKernel.h"

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

    float scale_x = arm_compute::scale_utils::calculate_resize_ratio(src_width, dst_width, align_corners);
    float scale_y = arm_compute::scale_utils::calculate_resize_ratio(src_height, dst_height, align_corners);

    return std::make_pair(scale_x, scale_y);
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
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized(src->data_type()) && !is_data_type_quantized_asymmetric(src->data_type()));

    float            scale_x     = 0.f;
    float            scale_y     = 0.f;
    const DataLayout data_layout = info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : info.data_layout;
    std::tie(scale_x, scale_y) = calculate_scale_factors(src, dst, data_layout, info.align_corners);

    ARM_COMPUTE_RETURN_ERROR_ON(info.interpolation_policy == InterpolationPolicy::AREA && (scale_x > 1.f || scale_y > 1.f));

    return Status{};
}
} // namespace

Status ClScaleKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, info));
    return Status{};
}

ClScaleKernel::ClScaleKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClScaleKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, info));
    auto padding_info = get_padding_info({ src, dst });

    // Info required for the static tuning
    _data_layout = info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : info.data_layout;

    const bool is_nhwc = _data_layout == DataLayout::NHWC;

    float scale_x = 0.f;
    float scale_y = 0.f;
    std::tie(scale_x, scale_y) = calculate_scale_factors(src, dst, _data_layout, info.align_corners);
    const bool is_qasymm_bilinear = is_data_type_quantized_asymmetric(src->data_type()) && info.interpolation_policy == InterpolationPolicy::BILINEAR;

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    auto interpolation_policy_to_use = info.interpolation_policy;
    if(info.interpolation_policy == InterpolationPolicy::AREA && scale_x <= 1.f && scale_y <= 1.f)
    {
        interpolation_policy_to_use = InterpolationPolicy::NEAREST_NEIGHBOR;
    }

    // Create kernel
    const int          idx_width         = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int          idx_height        = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const int          idx_channel       = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);
    const unsigned int src_width         = src->dimension(idx_width);
    const unsigned int src_height        = src->dimension(idx_height);
    const unsigned int dst_width         = dst->dimension(idx_width);
    const unsigned int dst_channels      = dst->dimension(idx_channel);
    unsigned int       vec_size          = 0;
    unsigned int       vec_size_leftover = 0;

    CLBuildOptions build_opts;
    if(_data_layout == DataLayout::NHWC)
    {
        vec_size          = adjust_vec_size(src->data_type() == DataType::F32 ? 4 : 8, dst_channels);
        vec_size_leftover = dst_channels % vec_size;
        build_opts.add_option("-DSRC_TENSOR_TYPE=BUFFER");
        build_opts.add_option("-DSRC_DATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
        build_opts.add_option("-DDST_TENSOR_TYPE=BUFFER");
        build_opts.add_option("-DDST_DATA_TYPE=" + get_cl_type_from_data_type(dst->data_type()));
        build_opts.add_option("-DCONSTANT_VALUE=" + string_from_pixel_value(info.constant_border_value, src->data_type()));
        build_opts.add_option("-DN0=" + support::cpp11::to_string(vec_size));
        build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(vec_size_leftover));
        build_opts.add_option("-DSCALE_" + string_from_interpolation_policy(interpolation_policy_to_use));
        build_opts.add_option_if(src->num_dimensions() > 3, "-DBATCHED_EXECUTION");
        build_opts.add_option_if(info.border_mode == BorderMode::REPLICATE, "-DBORDER_MODE_REPLICATE");
        build_opts.add_option_if(info.border_mode == BorderMode::CONSTANT, "-DBORDER_MODE_CONSTANT");
        build_opts.add_option_if(info.align_corners, "-DALIGN_CORNERS");
        build_opts.add_option_if(is_data_type_float(src->data_type()), "-DIS_FLOATING_POINT");
        build_opts.add_option_if_else(info.sampling_policy == SamplingPolicy::CENTER, "-DSAMPLING_POLICY_CENTER", "-DSAMPLING_POLICY_TOP_LEFT");
        if(is_qasymm_bilinear)
        {
            const UniformQuantizationInfo qinfo = src->quantization_info().uniform();
            build_opts.add_option("-DSCALE=" + support::cpp11::to_string(qinfo.scale));
            build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(qinfo.offset));
        }
        else
        {
            build_opts.add_option("-DSCALE=" + support::cpp11::to_string(1));
            build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(0));
        }
    }
    else if(_data_layout == DataLayout::NCHW)
    {
        vec_size          = adjust_vec_size(4, dst_width);
        vec_size_leftover = dst_width % vec_size;
        build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
        build_opts.add_option("-DCONSTANT_VALUE=" + string_from_pixel_value(info.constant_border_value, src->data_type()));
        build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(src_width));
        build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(src_height));
        build_opts.add_option("-DSCALE_X=" + float_to_string_with_full_precision(scale_x));
        build_opts.add_option("-DSCALE_Y=" + float_to_string_with_full_precision(scale_y));
        build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size));
        build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + ((vec_size_leftover == 0) ? support::cpp11::to_string(vec_size) : support::cpp11::to_string(vec_size_leftover)));
        build_opts.add_option_if(info.border_mode == BorderMode::REPLICATE, "-DBORDER_MODE_REPLICATE");
        build_opts.add_option_if(info.border_mode == BorderMode::CONSTANT, "-DBORDER_MODE_CONSTANT");
        build_opts.add_option_if(info.align_corners, "-DALIGN_CORNERS");
        build_opts.add_option_if_else(info.sampling_policy == SamplingPolicy::CENTER, "-DSAMPLING_POLICY_CENTER", "-DSAMPLING_POLICY_TOP_LEFT");
        if(is_qasymm_bilinear)
        {
            const UniformQuantizationInfo qinfo = src->quantization_info().uniform();
            build_opts.add_option("-DSCALE=" + support::cpp11::to_string(qinfo.scale));
            build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(qinfo.offset));
        }
    }
    else
    {
        ARM_COMPUTE_ERROR_ON("Unsupported data layout");
    }

    std::string interpolation_name = string_from_interpolation_policy(interpolation_policy_to_use);
    std::transform(interpolation_name.begin(), interpolation_name.end(), interpolation_name.begin(), ::tolower);
    std::string kernel_name = "scale_" + interpolation_name + "_";
    kernel_name += lower_string(string_from_data_layout(_data_layout));

    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps(vec_size));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));

    // Pass scale kernel arguments
    if(is_nhwc)
    {
        unsigned int idx = 2 * num_arguments_per_4d_tensor_nhwc();
        _kernel.setArg<cl_float>(idx++, scale_x);
        _kernel.setArg<cl_float>(idx++, scale_y);
    }
    // Set config_id for enabling LWS tuning
    _config_id = "scale_";
    _config_id += (info.border_mode == BorderMode::REPLICATE ? "Bord_rep" : "");
    _config_id += (info.sampling_policy == SamplingPolicy::CENTER ? "center" : "topleft");
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
            add_4d_tensor_nhwc_argument(idx, src);
            add_4d_tensor_nhwc_argument(idx, dst);
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
