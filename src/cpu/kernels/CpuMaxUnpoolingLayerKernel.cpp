/*
 * Copyright (c) 2020-2023 Arm Limited.
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
#include "src/cpu/kernels/CpuMaxUnpoolingLayerKernel.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/maxunpool/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
using namespace misc::shape_calculator;

namespace
{
static const std::vector<CpuMaxUnpoolingLayerKernel::MaxUnpoolingKernel> available_kernels =
{
    {
        "neon_fp32_maxunpooling",
        [](const DataTypeISASelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(neon_fp32_maxunpooling)
    },
    {
        "neon_fp16_maxunpooling",
        [](const DataTypeISASelectorData & data) { return data.dt == DataType::F16 && data.isa.fp16; },
        REGISTER_FP16_NEON(neon_fp16_maxunpooling)
    },
    {
        "neon_qu8_maxunpooling",
        [](const DataTypeISASelectorData & data) { return data.dt == DataType::QASYMM8; },
        REGISTER_QASYMM8_NEON(neon_qs8_maxunpooling)
    },
    {
        "neon_qs8_maxunpooling",
        [](const DataTypeISASelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED; },
        REGISTER_QASYMM8_SIGNED_NEON(neon_qu8_maxunpooling)
    },
};

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *indices, const ITensorInfo *dst, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, indices, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, indices);

    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    PoolingType         pool_type       = pool_info.pool_type;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    const int    pool_size_x = pool_info.pool_size.width;
    const int    pool_size_y = pool_info.pool_size.height;
    const Size2D pool_size(pool_size_x, pool_size_y);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(pool_type != PoolingType::MAX, "Pooling indices only supported for MAX pooling method");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((pool_size != Size2D(2, 2)), "Pooling indices only supported for pool size 2x2");
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);
    }

    return Status{};
}
} // namespace

void CpuMaxUnpoolingLayerKernel::configure(const ITensorInfo *src, const ITensorInfo *indices, ITensorInfo *dst, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst, indices);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, indices, dst, pool_info));
    ARM_COMPUTE_UNUSED(indices);

    const auto uk = CpuMaxUnpoolingLayerKernel::get_implementation(DataTypeISASelectorData{ src->data_type(), CPUInfo::get().get_isa() });
    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);
    _run_method = uk->ukernel;

    const TensorShape output_shape = compute_unpool_shape(*src, pool_info);
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(output_shape));

    auto window = calculate_max_window(*src, Steps());
    ICpuKernel::configure(window);
}

Status CpuMaxUnpoolingLayerKernel::validate(const ITensorInfo *src, const ITensorInfo *indices, const ITensorInfo *dst, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, indices, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, indices, dst, pool_info));
    return Status{};
}

void CpuMaxUnpoolingLayerKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src     = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto indices = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto dst     = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src, indices, dst, window);
}

const char *CpuMaxUnpoolingLayerKernel::name() const
{
    return "CpuMaxUnpoolingLayerKernel";
}

const std::vector<CpuMaxUnpoolingLayerKernel::MaxUnpoolingKernel> &CpuMaxUnpoolingLayerKernel::get_available_kernels()
{
    return available_kernels;
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
