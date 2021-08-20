/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/cpu/operators/CpuScale.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/utils/ScaleUtils.h"
#include "src/cpu/kernels/CpuScaleKernel.h"
#include "support/Rounding.h"

namespace arm_compute
{
namespace cpu
{
namespace
{
void precompute_dx_dy_offsets(ITensor *dx, ITensor *dy, ITensor *offsets, float wr, float hr, SamplingPolicy sampling_policy, bool align_corners)
{
    ARM_COMPUTE_ERROR_ON(offsets == nullptr);
    float sampling_offset = 0.0f;
    if(sampling_policy == SamplingPolicy::CENTER)
    {
        sampling_offset = 0.5f;
    }

    Window win;
    win.set(Window::DimX, Window::Dimension(0, offsets->info()->dimension(0), 1));
    win.set(Window::DimY, Window::Dimension(0, offsets->info()->dimension(1), 1));

    if(dx != nullptr && dy != nullptr)
    {
        // Pre-compute the offset and pixel's distance for BILINEAR interpolation
        Iterator offsets_it(offsets, win);
        Iterator dx_it(dx, win);
        Iterator dy_it(dy, win);

        execute_window_loop(win, [&](const Coordinates & id)
        {
            const float in_x  = (id.x() + sampling_offset) * wr - sampling_offset;
            const float in_y  = (id.y() + sampling_offset) * hr - sampling_offset;
            const int   in_xi = std::floor(in_x);
            const int   in_yi = std::floor(in_y);

            *reinterpret_cast<int32_t *>(offsets_it.ptr()) = in_xi;
            *reinterpret_cast<float *>(dx_it.ptr())        = in_x - in_xi;
            *reinterpret_cast<float *>(dy_it.ptr())        = in_y - in_yi;
        },
        offsets_it, dx_it, dy_it);
    }
    else
    {
        // Pre-compute the offset for NEAREST interpolation
        Iterator offsets_it(offsets, win);

        execute_window_loop(win, [&](const Coordinates & id)
        {
            const float float_in_xi                        = (id.x() + sampling_offset) * wr;
            const auto  in_xi                              = static_cast<size_t>(align_corners ? arm_compute::utils::rounding::round_half_away_from_zero(float_in_xi) : std::floor(float_in_xi));
            *reinterpret_cast<int32_t *>(offsets_it.ptr()) = in_xi;
        },
        offsets_it);
    }
}
} // namespace

void CpuScale::configure(ITensorInfo *src, ITensorInfo *dst, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(CpuScale::validate(src, dst, info));

    _scale_info  = info;
    _is_prepared = false;

    // Get data layout and width/height indices
    _data_layout         = _scale_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : _scale_info.data_layout;
    const int idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

    // Compute the ratio between source width/height and destination width/height
    const bool is_align_corners_used = _scale_info.align_corners && arm_compute::scale_utils::is_align_corners_allowed_sampling_policy(_scale_info.sampling_policy);
    const auto wr                    = arm_compute::scale_utils::calculate_resize_ratio(src->dimension(idx_width), dst->dimension(idx_width), is_align_corners_used);
    const auto hr                    = arm_compute::scale_utils::calculate_resize_ratio(src->dimension(idx_height), dst->dimension(idx_height), is_align_corners_used);

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    InterpolationPolicy policy_to_use = (_scale_info.interpolation_policy == InterpolationPolicy::AREA && wr <= 1.f
                                         && hr <= 1.f) ?
                                        InterpolationPolicy::NEAREST_NEIGHBOR :
                                        _scale_info.interpolation_policy;

    // Get the tensor shape
    TensorShape shape(dst->dimension(idx_width));
    shape.set(1, dst->dimension(idx_height), false);

    TensorInfo tensor_info_offsets(shape, Format::S32);
    TensorInfo tensor_info_dxdy(shape, Format::F32);

    auto dx           = std::make_unique<TensorInfo>(tensor_info_dxdy);
    auto dy           = std::make_unique<TensorInfo>(tensor_info_dxdy);
    auto offsets      = std::make_unique<TensorInfo>(tensor_info_offsets);
    auto scale_kernel = std::make_unique<kernels::CpuScaleKernel>();
    switch(policy_to_use)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
        {
            scale_kernel->configure(src, nullptr, nullptr, offsets.get(), dst, info);
            break;
        }
        case InterpolationPolicy::BILINEAR:
        {
            scale_kernel->configure(src, dx.get(), dy.get(), offsets.get(), dst, info);
            break;
        }
        case InterpolationPolicy::AREA:
        {
            scale_kernel->configure(src, nullptr, nullptr, nullptr, dst, info);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported interpolation mode");
    }
    _kernel = std::move(scale_kernel);
}

Status CpuScale::validate(const ITensorInfo *src, const ITensorInfo *dst, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(info.sampling_policy != SamplingPolicy::CENTER && info.sampling_policy != SamplingPolicy::TOP_LEFT);

    ITensorInfo *offsets = nullptr;
    ITensorInfo *dx      = nullptr;
    ITensorInfo *dy      = nullptr;

    // Get data layout and width/height indices
    const DataLayout data_layout = info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : info.data_layout;
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Compute the ratio between source width/height and destination width/height
    const bool is_align_corners_used = info.align_corners && arm_compute::scale_utils::is_align_corners_allowed_sampling_policy(info.sampling_policy);
    const auto wr                    = arm_compute::scale_utils::calculate_resize_ratio(src->dimension(idx_width), dst->dimension(idx_width), is_align_corners_used);
    const auto hr                    = arm_compute::scale_utils::calculate_resize_ratio(src->dimension(idx_height), dst->dimension(idx_height), is_align_corners_used);

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    InterpolationPolicy policy_to_use = (info.interpolation_policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f) ? InterpolationPolicy::NEAREST_NEIGHBOR : info.interpolation_policy;

    // Get the tensor shape of auxilary buffers
    const TensorShape shape(dst->dimension(idx_width), dst->dimension(idx_height));
    TensorInfo        tensor_info_offsets(shape, Format::S32);
    TensorInfo        tensor_info_dx(shape, Format::F32);
    TensorInfo        tensor_info_dy(shape, Format::F32);
    switch(policy_to_use)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
            offsets = &tensor_info_offsets;
            break;
        case InterpolationPolicy::BILINEAR:
            offsets = &tensor_info_offsets;
            dx      = &tensor_info_dx;
            dy      = &tensor_info_dy;
            break;
        default:
            break;
    }

    ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuScaleKernel::validate(src->clone().get(), dx, dy, offsets, dst->clone().get(), info));
    return Status{};
}

void CpuScale::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        _is_prepared       = true;
        const auto src     = tensors.get_const_tensor(TensorType::ACL_SRC);
        auto       dst     = tensors.get_tensor(TensorType::ACL_DST);
        auto       dx      = tensors.get_tensor(TensorType::ACL_INT_0);
        auto       dy      = tensors.get_tensor(TensorType::ACL_INT_1);
        auto       offsets = tensors.get_tensor(TensorType::ACL_INT_2);

        // Get data layout and width/height indices
        const int idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

        // Compute the ratio between source width/height and destination width/height
        const bool is_align_corners_used = _scale_info.align_corners && arm_compute::scale_utils::is_align_corners_allowed_sampling_policy(_scale_info.sampling_policy);
        const auto wr                    = arm_compute::scale_utils::calculate_resize_ratio(src->info()->dimension(idx_width), dst->info()->dimension(idx_width), is_align_corners_used);
        const auto hr                    = arm_compute::scale_utils::calculate_resize_ratio(src->info()->dimension(idx_height), dst->info()->dimension(idx_height), is_align_corners_used);

        // Area interpolation behaves as Nearest Neighbour in case of up-sampling
        InterpolationPolicy policy_to_use = (_scale_info.interpolation_policy == InterpolationPolicy::AREA && wr <= 1.f
                                             && hr <= 1.f) ?
                                            InterpolationPolicy::NEAREST_NEIGHBOR :
                                            _scale_info.interpolation_policy;
        const SamplingPolicy sampling_policy = _scale_info.sampling_policy;

        switch(policy_to_use)
        {
            case InterpolationPolicy::NEAREST_NEIGHBOR:
            {
                // Pre-compute offsets for nearest interpolation
                precompute_dx_dy_offsets(nullptr, nullptr, offsets, wr, hr, sampling_policy, is_align_corners_used);
                break;
            }
            case InterpolationPolicy::BILINEAR:
            {
                // Pre-compute dx, dy and offsets for bilinear interpolation
                precompute_dx_dy_offsets(dx, dy, offsets, wr, hr, sampling_policy, is_align_corners_used);
                break;
            }
            case InterpolationPolicy::AREA:
            {
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported interpolation mode");
        }
    }
}

void CpuScale::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    prepare(tensors);
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}
} // namespace cpu
} // namespace arm_compute
