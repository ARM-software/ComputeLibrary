/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#include "src/cpu/kernels/CpuDirectConv2dKernel.h"
#include "src/cpu/kernels/directconv2d/list.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

using namespace arm_compute::detail;

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
static const std::vector<CpuDirectConv2dKernel::DirectConv2dKernel> available_kernels =
{
    {
        "neon_fp32_nhwc_directconv2d",
        [](const DataTypeDataLayoutISASelectorData & data) { return data.dt == DataType::F32 && data.dl == DataLayout::NHWC; },
        REGISTER_FP32_NEON(arm_compute::cpu::kernels::neon_fp32_nhwc_directconv2d)
    },
    {
        "neon_fp32_nchw_directconv2d",
        [](const DataTypeDataLayoutISASelectorData & data) { return data.dt == DataType::F32 && data.dl == DataLayout::NCHW; },
        REGISTER_FP32_NEON(arm_compute::cpu::kernels::neon_fp32_nchw_directconv2d)
    },
    {
        "neon_fp16_nchw_directconv2d",
        [](const DataTypeDataLayoutISASelectorData & data) { return data.dt == DataType::F16 && data.dl == DataLayout::NCHW && data.isa.fp16; },
        REGISTER_FP16_NEON(arm_compute::cpu::kernels::neon_fp16_nchw_directconv2d)
    },
};

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);

    const DataLayout data_layout = src->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(channel_idx) != src->dimension(channel_idx));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(width_idx) != weights->dimension(height_idx));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(data_layout == DataLayout::NHWC && src->data_type() != DataType::F32);
    ARM_COMPUTE_UNUSED(width_idx);
    // Checks performed when output is configured
    if(dst->total_size() != 0)
    {
        TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info);

        DataType data_type = src->data_type();

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON(dst->data_type() != data_type);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON(src->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_UNUSED(src);

    Window win{};
    bool   window_changed = false;

    // Configure window without any padding
    win = calculate_max_window(*dst, Steps());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

void CpuDirectConv2dKernel::configure(ITensorInfo *src, ITensorInfo *weights, ITensorInfo *dst, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);

    _conv_info   = conv_info;
    _data_layout = src->data_layout();
    _kernel_size = weights->dimension(get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH));

    // Get convolved dimensions
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info);

    DataType data_type = src->data_type();

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, output_shape, 1, data_type);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, dst, conv_info));

    // Configure kernel window
    auto win_config = validate_and_configure_window(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuDirectConv2dKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, dst, conv_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(),
                                                              dst->clone().get())
                                .first);

    return Status{};
}

void CpuDirectConv2dKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src     = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto dst     = tensors.get_tensor(TensorType::ACL_DST);

    const auto *uk = CpuDirectConv2dKernel::get_implementation(DataTypeDataLayoutISASelectorData{ src->info()->data_type(), _data_layout, CPUInfo::get().get_isa() });
    ARM_COMPUTE_ERROR_ON(uk == nullptr);

    uk->ukernel(window, src, weights, dst, _conv_info);
}
const char *CpuDirectConv2dKernel::name() const
{
    return "CpuDirectConvolutionLayerKernel";
}

const std::vector<CpuDirectConv2dKernel::DirectConv2dKernel> &CpuDirectConv2dKernel::get_available_kernels()
{
    return available_kernels;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
