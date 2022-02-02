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
#include "src/cpu/kernels/CpuGemmMatrixMultiplyKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/gemm_matrix_mul/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
static const std::vector<CpuGemmMatrixMultiplyKernel::GemmMatrixMulKernel> available_kernels =
{
    {
        "neon_fp32_gemm_matrix_mul",
        [](const DataTypeISASelectorData & data)
        {
            return (data.dt == DataType::F32);
        },
        REGISTER_FP32_NEON(neon_fp32_gemm_matrix_mul)
    },
    {
        "neon_fp16_gemm_matrix_mul",
        [](const DataTypeISASelectorData & data)
        {
            return (data.dt == DataType::F16) && data.isa.fp16;
        },
        REGISTER_FP16_NEON(neon_fp16_gemm_matrix_mul)
    },
};

inline Status validate_arguments(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst, float alpha, bool is_interleaved, const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_UNUSED(alpha);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(lhs);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs, dst);

    if(!is_interleaved)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(lhs->dimension(0) != rhs->dimension(1));

        if(dst->total_size() != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(rhs->dimension(0) != dst->dimension(0));
            ARM_COMPUTE_RETURN_ERROR_ON(lhs->dimension(1) != dst->dimension(1));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);
        }
    }
    else
    {
        const int m                         = reshape_info.m();
        const int n                         = reshape_info.n();
        const int k                         = reshape_info.k();
        const int mult_transpose1xW_width   = reshape_info.mult_transpose1xW_width();
        const int mult_interleave4x4_height = reshape_info.mult_interleave4x4_height();

        /* Interleave */
        TensorShape tensor_shape0{ lhs->tensor_shape() };
        tensor_shape0.set(0, k);
        tensor_shape0.set(1, m);

        const TensorInfo tensor_info0          = lhs->clone()->set_tensor_shape(tensor_shape0);
        const TensorInfo tensor_info_reshaped0 = lhs->clone()->set_tensor_shape(misc::shape_calculator::compute_interleaved_shape(tensor_info0, mult_interleave4x4_height));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(lhs, &tensor_info_reshaped0);

        if(n != 0) /* Transpose */
        {
            TensorShape tensor_shape1{ rhs->tensor_shape() };
            tensor_shape1.set(0, n);
            tensor_shape1.set(1, k);

            const TensorInfo tensor_info1          = rhs->clone()->set_tensor_shape(tensor_shape1);
            const TensorInfo tensor_info_reshaped1 = rhs->clone()->set_tensor_shape(misc::shape_calculator::compute_transpose1xW_with_element_size_shape(tensor_info1, mult_transpose1xW_width));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(rhs, &tensor_info_reshaped1);
        }

        if(dst->total_size() != 0)
        {
            if(n != 0)
            {
                ARM_COMPUTE_RETURN_ERROR_ON(dst->dimension(0) != static_cast<size_t>(n));
            }
            ARM_COMPUTE_RETURN_ERROR_ON(dst->dimension(1) != static_cast<size_t>(m));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);
        }
    }

    return Status{};
}

} // namespace

void CpuGemmMatrixMultiplyKernel::configure(const ITensorInfo *lhs, const ITensorInfo *rhs, ITensorInfo *dst, float alpha, bool is_interleaved, const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);

    // dst tensor auto inizialitation if not yet initialized
    TensorShape tensor_shape{ lhs->tensor_shape() };
    tensor_shape.set(0, is_interleaved ? reshape_info.n() : rhs->dimension(0));
    tensor_shape.set(1, is_interleaved ? reshape_info.m() : lhs->dimension(1));

    auto_init_if_empty(*dst, lhs->clone()->set_tensor_shape(tensor_shape));

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(lhs, rhs, dst, alpha, is_interleaved, reshape_info));

    _alpha = alpha;

    // Configure kernel window
    Window win{};

    // Check if the dst tensor is a vector. If so,the kernel runs the vector-matrix multiplication
    const bool is_dst_vector = (dst->dimension(1) == 1);
    if(is_dst_vector)
    {
        const unsigned int num_elems_processed_per_iteration_x = (lhs->data_type() == DataType::F32) ? 16 : 32;

        win = calculate_max_window(*dst, Steps(num_elems_processed_per_iteration_x));
    }
    else
    {
        constexpr unsigned int num_elems_processed_per_iteration_x = 8;
        constexpr unsigned int num_elems_processed_per_iteration_y = 4;

        win = calculate_max_window(*dst, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    }

    const auto uk = CpuGemmMatrixMultiplyKernel::get_implementation(DataTypeISASelectorData{ lhs->data_type(), CPUInfo::get().get_isa() });
    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);
    _func = uk->ukernel;

    ICPPKernel::configure(win);
}

Status CpuGemmMatrixMultiplyKernel::validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst, float alpha, bool is_interleaved,
                                             const GEMMReshapeInfo &reshape_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(lhs, rhs, dst, alpha, is_interleaved, reshape_info));

    return Status{};
}

void CpuGemmMatrixMultiplyKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    const ITensor *lhs = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *rhs = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst = tensors.get_tensor(TensorType::ACL_DST);

    const bool is_dst_vector = (dst->info()->dimension(1) == 1);
    (*_func)(lhs, rhs, dst, window, info, _alpha, is_dst_vector);
}

const char *CpuGemmMatrixMultiplyKernel::name() const
{
    return "CpuGemmMatrixMultiplyKernel";
}

const std::vector<CpuGemmMatrixMultiplyKernel::GemmMatrixMulKernel> &CpuGemmMatrixMultiplyKernel::get_available_kernels()
{
    return available_kernels;
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
