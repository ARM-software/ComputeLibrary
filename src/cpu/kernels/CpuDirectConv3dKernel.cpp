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
#include "src/cpu/kernels/CpuDirectConv3dKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/cpu/kernels/conv3d/neon/list.h"

#include <algorithm>

using namespace arm_compute::detail;

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
struct DirectConv3dSelectorData
{
    DataType       dt;
    const CPUInfo &ci;
};
using DirectConv3dSelectorPtr = std::add_pointer<bool(const DirectConv3dSelectorData &data)>::type;
using DirectConv3dKernelPtr   = std::add_pointer<void(const ITensor *, const ITensor *, const ITensor *, ITensor *, const Conv3dInfo &, const Window &)>::type;
struct DirectConv3dKernel
{
    const char                   *name;
    const DirectConv3dSelectorPtr is_selected;
    DirectConv3dKernelPtr         ukernel;
};

static const DirectConv3dKernel available_kernels[] =
{
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_fp16_directconv3d",
        [](const DirectConv3dSelectorData & data) { return data.dt == DataType::F16 && data.ci.has_fp16(); },
        REGISTER_FP16_NEON(arm_compute::cpu::directconv3d_float_neon_ndhwc<float16_t>)
    },
#endif /* !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "neon_fp32_directconv3d",
        [](const DirectConv3dSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::directconv3d_float_neon_ndhwc<float>)
    }
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const DirectConv3dKernel *get_implementation(const DirectConv3dSelectorData &data)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected(data))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const Conv3dInfo &conv_info)
{
    const auto *uk = get_implementation(DirectConv3dSelectorData{ src0->data_type(), CPUInfo::get() });
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src0->data_layout() != DataLayout::NDHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src0, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, src1);

    const DataLayout data_layout = src0->data_layout();
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    // Weight layout is D, H, W, Cin, Cout
    ARM_COMPUTE_RETURN_ERROR_ON(src1->num_dimensions() > 5);
    ARM_COMPUTE_RETURN_ERROR_ON(src1->dimension(1) != src0->dimension(channel_idx));

    if(src2 != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src1, src2);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(src2->dimension(0) != src1->dimension(0),
                                        "biases size and number of output feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(src2->num_dimensions() > 1, "biases should be one dimensional");
    }

    // Checks performed when output is configured
    if(dst->total_size() != 0)
    {
        TensorShape output_shape = misc::shape_calculator::compute_conv3d_shape(src0->tensor_shape(), src1->tensor_shape(), conv_info);

        DataType data_type = src0->data_type();

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON(dst->data_type() != data_type);
    }

    return Status{};
}
}

void CpuDirectConv3dKernel::configure(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, ITensorInfo *dst, const Conv3dInfo &conv_info)
{
    ARM_COMPUTE_UNUSED(src2);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

    const auto *uk = get_implementation(DirectConv3dSelectorData{ src0->data_type(), CPUInfo::get() });
    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _conv_info  = conv_info;
    _run_method = uk->ukernel;
    _name       = std::string("CpuDirectConv3dKernel").append("/").append(uk->name);

    // Get convolved dimensions
    TensorShape output_shape = misc::shape_calculator::compute_conv3d_shape(src0->tensor_shape(), src1->tensor_shape(), conv_info);

    DataType data_type = src0->data_type();

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, output_shape, 1, data_type);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src0, src1, src2, dst, conv_info));

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps());
    ICpuKernel::configure(win);
}

Status CpuDirectConv3dKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const Conv3dInfo &conv_info)
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
} // namespace kernels
} // namespace cpu
} // namespace arm_compute