/*
 * Copyright (c) 2024 Arm Limited.
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
#include "src/cpu/kernels/CpuScatterKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/common/utils/Log.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/scatter/list.h"

#include <cstdint>
#include <vector>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

constexpr int max_index_length = 5;

/* Scatter */
static const std::vector<typename CpuScatterKernel::ScatterKernel> available_kernels = {
    {"neon_fp32_scatter", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F32); },
     REGISTER_FP32_NEON(arm_compute::cpu::scatter_fp32_neon)}};

const std::vector<typename CpuScatterKernel::ScatterKernel> &CpuScatterKernel::get_available_kernels()
{
    return available_kernels;
}

void CpuScatterKernel::configure(const ITensorInfo *updates,
                                 const ITensorInfo *indices,
                                 ITensorInfo       *dst,
                                 const ScatterInfo &scatter_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(updates, dst, indices);
    ARM_COMPUTE_LOG_PARAMS(updates, indices, dst, scatter_info);

    const auto uk = CpuScatterKernel::get_implementation<DataTypeISASelectorData>(
        DataTypeISASelectorData{updates->data_type(), CPUInfo::get().get_isa()});

    _run_method   = uk->ukernel;
    _scatter_info = scatter_info;
    _name         = std::string("CpuScatterKernel").append("/").append(uk->name);

    const int index_len = indices->dimension(0);

    // Check for single element data block
    const bool is_scalar_block = (dst->num_dimensions() == static_cast<uint32_t>(index_len));

    _data_block_length = is_scalar_block ? 1 : updates->dimension(0);

    // The GWS will be 2D [x, y]
    //  x-dimension refers to the x coordinate of the dst tensor
    //  y-dimension refers to the collapsed y-coordinate of the data part of the dst tensor
    Window win;

    if (!is_scalar_block)
    {
        win = calculate_max_window(*dst, Steps(_data_block_length));

        // Collapse the dimensions corresponding to indices in the execution window
        for (int i = 0; i < index_len; ++i)
        {
            win.set(dst->num_dimensions() - (i + 1), Window::Dimension(0, 1, 1));
        }

        win = win.collapse(win, 1);
    }

    ICpuKernel::configure(win);
}

Status CpuScatterKernel::validate(const ITensorInfo *updates,
                                  const ITensorInfo *indices,
                                  const ITensorInfo *dst,
                                  const ScatterInfo &scatter_info)
{
    ARM_COMPUTE_UNUSED(scatter_info);

    const TensorShape &ind_shape = indices->tensor_shape();
    const TensorShape &upt_shape = updates->tensor_shape();
    const TensorShape &dst_shape = dst->tensor_shape();

    const int32_t upt_dims = upt_shape.num_dimensions();
    const int32_t dst_dims = dst_shape.num_dimensions();
    const int32_t ind_dims = ind_shape.num_dimensions();
    const int32_t data_dim = upt_dims - (ind_dims - 1); // Number of batch dims is the number of indices dims - 1

    const int32_t index_len = ind_shape[0];

    bool unsupported_padding_config =
        (dst_dims == index_len) && index_len > 1 && (dst->has_padding() || updates->has_padding());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(unsupported_padding_config, "Padding is not supported with these shapes.");
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(updates, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(indices, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(dst, DataType::F32);

    // Check data dims in update tensor and output tensor are equal
    for (int32_t i = 0; i < data_dim; i++)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(upt_shape[i] != dst_shape[i],
                                        "Data dims should be same size in both updates and ouput tensor.");
    }

    // Check if batch dims in indices and updates tensor are equal.
    for (int32_t i = 0; i < ind_dims - 1; i++)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(upt_shape[data_dim + i] != ind_shape[i + 1],
                                        "Batch dimensions should be the same in updates and indices tensor.");
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(ind_shape[1] != upt_shape[data_dim],
                                    "Height of indices tensor should match size of highest dimension in updates tensor "
                                    "(Excluding batch dimension)");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        data_dim >= dst_dims, "Update tensor cannot have more dims than output tensor. (Excluding batch dimensions)");
    ARM_COMPUTE_RETURN_ERROR_ON(index_len != dst_dims - data_dim);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((ind_dims < 2), "Shape of Indices tensor must be at least 2D");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(index_len > max_index_length, "Maximum supported index length is 5!");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(index_len > dst_dims && dst_dims != 1,
                                    "Index length should be smaller than or equal to number of output dims");

    return Status{};
}

void CpuScatterKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *updates = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const ITensor *indices = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    ITensor       *dst     = tensors.get_tensor(TensorType::ACL_DST);

    const ITensorInfo *dst_info  = dst->info();
    const ITensorInfo *upd_info  = updates->info();
    const int          num_dims  = dst_info->num_dimensions();
    const int          index_len = indices->info()->dimension(0);

    bool unsupported_padding_config =
        num_dims == index_len && index_len > 1 && (dst_info->has_padding() || upd_info->has_padding());
    if (unsupported_padding_config)
    {
        ARM_COMPUTE_ERROR("Unsupported Configuration! Padding not supported with these shapes.");
    }

    _run_method(updates, indices, dst, _scatter_info, window, _data_block_length);
}

const char *CpuScatterKernel::name() const
{
    return _name.c_str();
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
