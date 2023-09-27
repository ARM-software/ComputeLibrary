/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEScale.h"

#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/core/utils/ScaleUtils.h"
#include "src/cpu/operators/CpuScale.h"

namespace arm_compute
{
struct NEScale::Impl
{
    const ITensor *src{nullptr};
    ITensor       *dst{nullptr};
    Tensor dx{nullptr}; /**< Element's distance between the X real coordinate and the smallest X following integer */
    Tensor dy{nullptr}; /**< Element's distance between the Y real coordinate and the smallest Y following integer */
    Tensor offsets{
        nullptr}; /**< Offset to access the element with NEAREST interpolation or the top-left element with BILINEAR interpolation in the input tensor */
    std::unique_ptr<cpu::CpuScale> op{nullptr};
};

NEScale::NEScale() : _impl(std::make_unique<Impl>())
{
}
NEScale::~NEScale() = default;

void NEScale::configure(ITensor *input, ITensor *output, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, info);

    _impl->src = input;
    _impl->dst = output;
    _impl->op  = std::make_unique<cpu::CpuScale>();
    _impl->op->configure(input->info(), output->info(), info);

    // Configure for size of allocation of internal tensors
    // Get data layout and width/height indices
    const DataLayout data_layout =
        info.data_layout == DataLayout::UNKNOWN ? input->info()->data_layout() : info.data_layout;
    const int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Compute the ratio between source width/height and destination width/height
    const bool is_align_corners_used =
        info.align_corners && arm_compute::scale_utils::is_align_corners_allowed_sampling_policy(info.sampling_policy);
    const auto wr = arm_compute::scale_utils::calculate_resize_ratio(
        input->info()->dimension(idx_width), output->info()->dimension(idx_width), is_align_corners_used);
    const auto hr = arm_compute::scale_utils::calculate_resize_ratio(
        input->info()->dimension(idx_height), output->info()->dimension(idx_height), is_align_corners_used);

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    InterpolationPolicy policy_to_use =
        (info.interpolation_policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f)
            ? InterpolationPolicy::NEAREST_NEIGHBOR
            : info.interpolation_policy;

    // Get the tensor shape
    TensorShape shape(output->info()->dimension(idx_width));
    shape.set(1, output->info()->dimension(idx_height), false);

    bool precompute_indices_weights = arm_compute::scale_utils::is_precomputation_required(
        data_layout, input->info()->data_type(), policy_to_use, info.border_mode);

    if (precompute_indices_weights)
    {
        const TensorInfo tensor_info_dxdy(shape, Format::F32);
        const TensorInfo tensor_info_offsets(shape, Format::S32);

        _impl->dx.allocator()->init(tensor_info_dxdy);
        _impl->dy.allocator()->init(tensor_info_dxdy);
        _impl->offsets.allocator()->init(tensor_info_offsets);
        switch (policy_to_use)
        {
            case InterpolationPolicy::NEAREST_NEIGHBOR:
            {
                // Allocate once the configure methods have been called
                _impl->offsets.allocator()->allocate();
                break;
            }
            case InterpolationPolicy::BILINEAR:
            {
                // Allocate once the configure methods have been called
                _impl->dx.allocator()->allocate();
                _impl->dy.allocator()->allocate();
                _impl->offsets.allocator()->allocate();
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
    else
    {
        if (policy_to_use != InterpolationPolicy::NEAREST_NEIGHBOR && policy_to_use != InterpolationPolicy::BILINEAR &&
            policy_to_use != InterpolationPolicy::AREA)
        {
            ARM_COMPUTE_ERROR("Unsupported interpolation mode");
        }
    }
}

Status NEScale::validate(const ITensorInfo *input, const ITensorInfo *output, const ScaleKernelInfo &info)
{
    return cpu::CpuScale::validate(input, output, info);
}

void NEScale::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    pack.add_tensor(TensorType::ACL_INT_0, &_impl->dx);
    pack.add_tensor(TensorType::ACL_INT_1, &_impl->dy);
    pack.add_tensor(TensorType::ACL_INT_2, &_impl->offsets);
    _impl->op->run(pack);
}
} // namespace arm_compute
