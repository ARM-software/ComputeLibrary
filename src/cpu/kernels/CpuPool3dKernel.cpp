/*
 * Copyright (c) 2022 Arm Limited.
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
#include "src/cpu/kernels/CpuPool3dKernel.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/pool3d/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
using namespace misc::shape_calculator;

static const std::vector<CpuPool3dKernel::Pooling3dKernel> available_kernels = {
    {"neon_qu8_ndhwc_poolMxNxD", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8); },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_q8_pool3d)},
    {"neon_qs8_ndhwc_poolMxNxD",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8_SIGNED); },
     REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_q8_signed_pool3d)},
    {"neon_fp16_ndhwc_poolMxNxD",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F16 && data.isa.fp16); },
     REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_pool3d)},
    {"neon_fp32_ndhwc_poolMxNxD", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F32); },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_pool3d)}};

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const Pooling3dLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->data_layout() != DataLayout::NDHWC, "Only NDHWC layout supported");
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32, DataType::QASYMM8,
                                                         DataType::QASYMM8_SIGNED);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((!is_data_type_float(src->data_type())) &&
                                        (!pool_info.exclude_padding && (pool_info.pool_type == PoolingType::AVG)),
                                    "Exclude padding is unsupported for non-float types for Avg op");

    const auto data_layout = src->data_layout();
    const int  idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int  idx_depth   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::DEPTH);

    const bool         is_global_pooling = pool_info.is_global_pooling;
    const unsigned int pool_size_x       = is_global_pooling ? src->dimension(idx_width) : pool_info.pool_size.width;
    const unsigned int pool_size_y       = is_global_pooling ? src->dimension(idx_height) : pool_info.pool_size.height;
    const unsigned int pool_size_z       = is_global_pooling ? src->dimension(idx_depth) : pool_info.pool_size.depth;

    const unsigned int stride_x = pool_info.stride.x();
    const unsigned int stride_y = pool_info.stride.y();
    const unsigned int stride_z = pool_info.stride.z();

    ARM_COMPUTE_RETURN_ERROR_ON((pool_size_x == 0) || (pool_size_y == 0) || (pool_size_z == 0));
    ARM_COMPUTE_RETURN_ERROR_ON((stride_x == 0) || (stride_y == 0) || (stride_z == 0));

    int output_width  = 0;
    int output_height = 0;
    int output_depth  = 0;

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_pool_3d_region_entirely_outside_input(pool_info),
                                    "Pooling region that is entirely outside input tensor is unsupported");

    std::tie(output_width, output_height, output_depth) =
        scaled_3d_dimensions_signed(src->tensor_shape()[idx_width], src->tensor_shape()[idx_height],
                                    src->tensor_shape()[idx_depth], pool_size_x, pool_size_y, pool_size_z, pool_info);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((output_width < 1 || output_height < 1 || output_depth < 1),
                                    "Calculated output dimension size is invalid");

    if (dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);
        TensorInfo out_info(
            TensorInfo(compute_pool3d_shape(src->tensor_shape(), pool_info), 1, dst->data_type(), DataLayout::NDHWC));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &out_info);
    }

    const auto *uk =
        CpuPool3dKernel::get_implementation(DataTypeISASelectorData{src->data_type(), CPUInfo::get().get_isa()});
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    return Status{};
}
} //namespace

void CpuPool3dKernel::configure(const ITensorInfo *src, ITensorInfo *dst, const Pooling3dLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, pool_info));

    // dst auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_pool3d_shape(src->tensor_shape(), pool_info)));

    // Get data layout
    const auto data_layout = src->data_layout();
    const int  idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int  idx_depth   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::DEPTH);

    // Update pool size in case of global pooling
    const bool   is_global_pooling = pool_info.is_global_pooling;
    const Size3D pool_size(is_global_pooling ? src->dimension(idx_width) : pool_info.pool_size.width,
                           is_global_pooling ? src->dimension(idx_height) : pool_info.pool_size.height,
                           is_global_pooling ? src->dimension(idx_depth) : pool_info.pool_size.depth);

    const auto *uk =
        CpuPool3dKernel::get_implementation(DataTypeISASelectorData{src->data_type(), CPUInfo::get().get_isa()});
    ARM_COMPUTE_ERROR_ON(uk == nullptr);

    // Set instance variables
    _pool_info  = pool_info;
    _run_method = uk->ukernel;
    _name       = std::string("CpuPool3dKernel").append("/").append(uk->name);

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps());
    ICpuKernel::configure(win);
}

Status CpuPool3dKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const Pooling3dLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, pool_info));

    return Status{};
}

void CpuPool3dKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    ITensor       *dst = tensors.get_tensor(TensorType::ACL_DST_0);

    _run_method(src, dst, _pool_info, window);
}

const char *CpuPool3dKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuPool3dKernel::Pooling3dKernel> &CpuPool3dKernel::get_available_kernels()
{
    return available_kernels;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
