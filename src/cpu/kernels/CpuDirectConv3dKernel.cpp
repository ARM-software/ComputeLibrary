/*
 * Copyright (c) 2021-2022, 2024 Arm Limited.
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
#include "src/cpu/kernels/CpuDirectConv3dKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Steps.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/conv3d/list.h"

using namespace arm_compute::detail;

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
static const std::vector<CpuDirectConv3dKernel::DirectConv3dKernel> available_kernels = {
    {"neon_fp16_directconv3d",
     [](const DataTypeISASelectorData &data) { return data.dt == DataType::F16 && data.isa.fp16; },
     REGISTER_FP16_NEON(directconv3d_fp16_neon_ndhwc)},
    {"neon_fp32_directconv3d", [](const DataTypeISASelectorData &data) { return data.dt == DataType::F32; },
     REGISTER_FP32_NEON(directconv3d_fp32_neon_ndhwc)},
    {"neon_qasymm8_directconv3d", [](const DataTypeISASelectorData &data) { return data.dt == DataType::QASYMM8; },
     REGISTER_QASYMM8_NEON(directconv3d_qu8_neon_ndhwc)},
    {"neon_qasymm8_signed_directconv3d",
     [](const DataTypeISASelectorData &data) { return data.dt == DataType::QASYMM8_SIGNED; },
     REGISTER_QASYMM8_SIGNED_NEON(directconv3d_qs8_neon_ndhwc)}};

Status validate_arguments(const ITensorInfo *src0,
                          const ITensorInfo *src1,
                          const ITensorInfo *src2,
                          const ITensorInfo *dst,
                          const Conv3dInfo  &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src0->data_layout() != DataLayout::NDHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src0, src1, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src0, 1, DataType::F16, DataType::F32, DataType::QASYMM8,
                                                         DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, src1);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.dilation != Size3D(1U, 1U, 1U));

    const auto *uk =
        CpuDirectConv3dKernel::get_implementation(DataTypeISASelectorData{src0->data_type(), CPUInfo::get().get_isa()});

    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    const DataLayout data_layout = src0->data_layout();
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    // Weight layout is D, H, W, Cin, Cout
    ARM_COMPUTE_RETURN_ERROR_ON(src1->num_dimensions() > 5);
    ARM_COMPUTE_RETURN_ERROR_ON(src1->dimension(1) != src0->dimension(channel_idx));

    if (src2 != nullptr)
    {
        if (is_data_type_quantized(src0->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src2, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src1, src2);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(src2->dimension(0) != src1->dimension(0),
                                        "Biases size and number of dst feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(src2->num_dimensions() > 1, "Biases should be one dimensional");
    }

    // Checks performed when output is configured
    if (dst->total_size() != 0)
    {
        TensorShape output_shape =
            misc::shape_calculator::compute_conv3d_shape(src0->tensor_shape(), src1->tensor_shape(), conv_info);

        DataType data_type = src0->data_type();

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON(dst->data_type() != data_type);
    }

    return Status{};
}
} // namespace

void CpuDirectConv3dKernel::configure(const ITensorInfo *src0,
                                      const ITensorInfo *src1,
                                      const ITensorInfo *src2,
                                      ITensorInfo       *dst,
                                      const Conv3dInfo  &conv_info)
{
    ARM_COMPUTE_UNUSED(src2);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

    const auto *uk =
        CpuDirectConv3dKernel::get_implementation(DataTypeISASelectorData{src0->data_type(), CPUInfo::get().get_isa()});

    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _conv_info  = conv_info;
    _run_method = uk->ukernel;
    _name       = std::string("CpuDirectConv3dKernel").append("/").append(uk->name);

    // Get convolved dimensions
    TensorShape output_shape =
        misc::shape_calculator::compute_conv3d_shape(src0->tensor_shape(), src1->tensor_shape(), conv_info);

    DataType data_type = src0->data_type();

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, output_shape, 1, data_type);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src0, src1, src2, dst, conv_info));

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps());
    ICpuKernel::configure(win);
}

Status CpuDirectConv3dKernel::validate(const ITensorInfo *src0,
                                       const ITensorInfo *src1,
                                       const ITensorInfo *src2,
                                       const ITensorInfo *dst,
                                       const Conv3dInfo  &conv_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src0, src1, src2, dst, conv_info));

    return Status{};
}

void CpuDirectConv3dKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    auto src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto src2 = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src0, src1, src2, dst, _conv_info, window);
}

const char *CpuDirectConv3dKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuDirectConv3dKernel::DirectConv3dKernel> &CpuDirectConv3dKernel::get_available_kernels()
{
    return available_kernels;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
