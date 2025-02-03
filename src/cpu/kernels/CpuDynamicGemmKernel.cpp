/*
 * Copyright (c) 2025 Arm Limited.
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
#include "src/cpu/kernels/CpuDynamicGemmKernel.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/function_info/GEMMInfo.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/kernels/CpuDynamicGemmKernelHeuristics.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"
#if defined(__aarch64__)
/* To verify that external headers are visible. */
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#endif /* __aarch64__ */

using namespace arm_compute::experimental;
using namespace arm_compute::cpu::kernels::heuristics;

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

void CpuDynamicGemmKernel::configure(const ITensorInfo *a,
                                     const ITensorInfo *b,
                                     const ITensorInfo *c,
                                     ITensorInfo       *d,
                                     float              alpha,
                                     float              beta,
                                     size_t             base_aux_slot,
                                     const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_ERROR_THROW_ON(CpuDynamicGemmKernel::validate(a, b, c, d, alpha, beta, gemm_info));

    _heuristics = CpuDynamicGemmKernelHeuristics{a, b, c, d, alpha, beta, gemm_info};

    _name = std::string{"CpuDynamicGemmKernel"}.append("/").append(_heuristics.name());

    _base_aux_slot = base_aux_slot;
    _aux_mem.reserve(Count);
    _reshape_b_and_c_only_on_first_run = b->are_values_constant() && c->are_values_constant();

    ICPPKernel::configure(_heuristics.window());
}

Status CpuDynamicGemmKernel::validate(const ITensorInfo *a,
                                      const ITensorInfo *b,
                                      const ITensorInfo *c,
                                      const ITensorInfo *d,
                                      float              alpha,
                                      float              beta,
                                      const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(a, b, c, d);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(gemm_info);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b, c, d);

    // If both a and b are static, so are c and d, rendering this kernel moot.
    ARM_COMPUTE_RETURN_ERROR_ON(!a->is_dynamic() && !b->is_dynamic());
    // ...conversely, when either a or b is dynamic, so is d.
    ARM_COMPUTE_RETURN_ERROR_ON(!d->is_dynamic());
    // What remains that could possibly be static is exactly one of a or b, and
    // optionally c. Dimensions are checked in run_op.

    // We expect to be able to pre-pack b and c if the values are constant, so
    // they must be static.
    if (b->are_values_constant())
    {
        ARM_COMPUTE_RETURN_ERROR_ON(b->is_dynamic());
    }
    if (c->are_values_constant())
    {
        ARM_COMPUTE_RETURN_ERROR_ON(c->is_dynamic());
    }

    ARM_COMPUTE_RETURN_ERROR_ON(alpha != 1.0f);
    ARM_COMPUTE_RETURN_ERROR_ON(beta != 1.0f);

    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.is_a_reshaped());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.is_b_reshaped());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.reshape_b_only_on_first_run() &&
                                (!b->are_values_constant() || !c->are_values_constant()));
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.depth_output_gemm3d() != 0);
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.reinterpret_input_as_3d());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.retain_internal_weights());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.gemmlowp_output_stage() != GEMMLowpOutputStageInfo{});
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.fast_math());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.fp_mixed_precision());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.broadcast_bias());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.pretranspose_A());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.pretranspose_B());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.activation_info() != ActivationLayerInfo{});
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.fixed_format());
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.weight_format() != WeightFormat::UNSPECIFIED);
    ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.accumulate());

    return Status{};
}

void CpuDynamicGemmKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_EXIT_ON_MSG(tensors.empty(), "No inputs provided");

    ICPPKernel::configure(window);

    const ITensor *a = tensors.get_const_tensor(ACL_SRC_0);
    const ITensor *b = tensors.get_const_tensor(ACL_SRC_1);
    const ITensor *c = tensors.get_const_tensor(ACL_SRC_2);
    ITensor       *d = tensors.get_tensor(ACL_DST);

    ARM_COMPUTE_EXIT_ON_MSG(
        a->info()->dimension(0) != b->info()->dimension(1),
        "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_EXIT_ON_MSG(a->info()->dimension(1) != d->info()->dimension(1),
                            "The number of rows in Output must equal the number of rows in Lhs");
    ARM_COMPUTE_EXIT_ON_MSG(b->info()->dimension(0) != d->info()->dimension(0),
                            "The number of columns in Output must equal the number of columns in Rhs");
    ARM_COMPUTE_EXIT_ON_MSG(c->info()->dimension(0) != d->info()->dimension(0),
                            "The number of columns in Output must equal the number of columns in Bias");
    ARM_COMPUTE_EXIT_ON_MSG(c->info()->dimension(1) != 1, "Bias must be a vector");

    ITensor *pack_b_and_c = tensors.get_tensor(offset_int_vec(_base_aux_slot + PackedRHS));
    if (_has_packed_b_and_c)
    {
        b = pack_b_and_c;
    }
    ITensor *pack_b_and_c_output = _has_packed_b_and_c ? nullptr : pack_b_and_c;
    _heuristics.kernel()(a, b, c, d, pack_b_and_c_output, window);

    if (_reshape_b_and_c_only_on_first_run)
    {
        _has_packed_b_and_c = true;
    }
}

const char *CpuDynamicGemmKernel::name() const
{
    return _name.c_str();
}

const MemoryRequirements &CpuDynamicGemmKernel::workspace(const ITensorPack &tensors) const
{
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *const b = tensors.get_const_tensor(ACL_SRC_1);
    ARM_COMPUTE_ERROR_ON_NULLPTR(b);

    // The ukernel needs a tensor allocation for the packed RHS.
    const TensorShape &b_shape     = b->info()->tensor_shape();
    const size_t       pack_b_size = size_of_packed_rhs(b_shape.y(), b_shape.x());
    _aux_mem[PackedRHS]            = MemoryInfo{offset_int_vec(_base_aux_slot + PackedRHS), MemoryLifetime::Persistent,
                                     std::max(pack_b_size, size_t{1})};

    return _aux_mem;
}

size_t CpuDynamicGemmKernel::size_of_packed_rhs(size_t rows, size_t columns) const
{
#if defined(__aarch64__)
    // The 0.5.0 documentation is wrong. In a kxn matrix, k=rows and n=columns.
    return kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(columns, rows);
#else  // __aarch64__
    ARM_COMPUTE_UNUSED(rows);
    ARM_COMPUTE_UNUSED(columns);

    return 0;
#endif // __aarch64__
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
