/*
 * Copyright (c) 2023 Arm Limited.
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

#include "src/cpu/operators/CpuMatMul.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEMatMul.h"
#include "src/common/utils/Log.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/utils/quantization/AsymmHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{
namespace
{
Status get_gemmlowp_output_stage_info(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const ActivationLayerInfo &act,
                                      GEMMLowpOutputStageInfo &gemmlowp_output_stage_info)
{
    const auto                    data_type = src->data_type();
    const QuantizationInfo        oq_info   = dst->quantization_info();
    const UniformQuantizationInfo iq_unif   = src->quantization_info().uniform();
    const UniformQuantizationInfo wq_unif   = weights->quantization_info().uniform();
    const UniformQuantizationInfo oq_unif   = oq_info.uniform();

    float   multiplier = (iq_unif.scale * wq_unif.scale) / oq_unif.scale;
    int32_t output_multiplier;
    int32_t output_shift;

    ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift));

    int32_t type_min = 0;
    int32_t type_max = 0;
    std::tie(type_min, type_max) = quantization::get_quantized_asymmetric_output_min_max(oq_info, act, data_type);

    gemmlowp_output_stage_info.gemmlowp_multiplier = output_multiplier;
    gemmlowp_output_stage_info.gemmlowp_shift      = output_shift;
    gemmlowp_output_stage_info.gemmlowp_offset     = oq_unif.offset;
    gemmlowp_output_stage_info.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_output_stage_info.gemmlowp_min_bound  = type_min;
    gemmlowp_output_stage_info.gemmlowp_max_bound  = type_max;

    return Status{};
}
} // namespace

CpuMatMul::CpuMatMul()
    : _transpose_kernel_lhs(), _transpose_kernel_rhs(), _asm_glue(), _lhs_transposed(), _rhs_transposed(), _original_lhs_shape(), _original_rhs_shape(), _original_dst_shape()
{
}

Status CpuMatMul::validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst, const MatMulInfo &info, const CpuMatMulSettings &settings, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::F32, DataType::F16, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs->are_values_constant(), "LHS Tensor must be dynamic.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(rhs->are_values_constant(), "RHS Tensor must be dynamic.");
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(lhs);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_BF16_UNSUPPORTED(lhs);

    const auto adj_lhs = info.adj_lhs();
    const auto adj_rhs = info.adj_rhs();

    const ITensorInfo *lhs_to_use = lhs;
    const ITensorInfo *rhs_to_use = rhs;
    TensorInfo         lhs_transposed{};
    TensorInfo         rhs_transposed{};

    auto gemm_info            = AsmGemmInfo();
    gemm_info.activation_info = act_info;
    gemm_info.fast_mode       = settings.fast_math();

    // Validate and then permute a/b
    if(adj_lhs)
    {
        auto_init_if_empty(lhs_transposed, lhs->clone()->set_tensor_shape(misc::shape_calculator::compute_transposed_shape(*lhs)));
        ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuTransposeKernel::validate(lhs_to_use, &lhs_transposed));
        // Assign lhs_to_use pointer to use transposed TensorInfo
        lhs_to_use = &lhs_transposed;
    }
    if(adj_rhs)
    {
        auto_init_if_empty(rhs_transposed, rhs->clone()->set_tensor_shape(misc::shape_calculator::compute_transposed_shape(*rhs)));
        ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuTransposeKernel::validate(rhs_to_use, &rhs_transposed));
        // Assign rhs_to_use pointer to use transposed TensorInfo
        rhs_to_use = &rhs_transposed;
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs_to_use->dimension(0) != rhs_to_use->dimension(1),
                                    "The product AB is defined only if the number of columns in A is equal to the number of rows in B (after transpose)");

    // Iterate over dimensions to be collapsed in operator - check dimensions are equivalent between tensors
    for(unsigned int i = 2; i < Coordinates::num_max_dimensions; i++)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs_to_use->dimension(i) != rhs_to_use->dimension(i), "Broadcasting in Batch dimension is unsupported by this operator.");
    }

    // Quantized-specific configuration
    if(is_data_type_quantized(lhs->data_type()))
    {
        ARM_COMPUTE_RETURN_ON_ERROR(get_gemmlowp_output_stage_info(lhs_to_use, rhs_to_use, dst, gemm_info.activation_info, gemm_info.output_stage));
    }

    cpu::CpuGemmAssemblyDispatch::validate(lhs_to_use, rhs_to_use, nullptr, dst, gemm_info);

    return Status{};
}

void CpuMatMul::configure(ITensorInfo *lhs, ITensorInfo *rhs, ITensorInfo *dst, const MatMulInfo &info, const CpuMatMulSettings &settings, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_LOG_PARAMS(lhs, rhs, dst, info, settings);
    ARM_COMPUTE_ERROR_THROW_ON(CpuMatMul::validate(lhs, rhs, dst, info, settings));

    _adj_lhs   = info.adj_lhs();
    _adj_rhs   = info.adj_rhs();
    _fast_math = settings.fast_math();

    // 1. Create and reshape tensors
    // ------------------------------------------------------
    // a. Clone TensorInfo to prevent changing original tensor values during setup
    // b. Change shape of lhs/dst to [x, y, 1, collapsed(z)] to match assembly kernel configuration
    // c. For rhs collapse all dimensions larger than 3 to z dimension
    TensorInfo lhs_to_use = *lhs->clone();
    TensorInfo dst_to_use = *dst->clone();
    TensorInfo rhs_to_use = *rhs->clone();

    // Save starting shape of tensors
    _original_lhs_shape = lhs_to_use.tensor_shape();
    _original_dst_shape = dst_to_use.tensor_shape();
    _original_rhs_shape = rhs_to_use.tensor_shape();

    // Reshape lhs for use with assembly kernels.
    lhs_to_use.set_tensor_shape(TensorShape(_original_lhs_shape.x(), _original_lhs_shape.y(), 1, _original_lhs_shape.collapsed_from(2).z()));
    dst_to_use.set_tensor_shape(TensorShape(_original_dst_shape.x(), _original_dst_shape.y(), 1, _original_dst_shape.collapsed_from(2).z()));
    rhs_to_use.set_tensor_shape(_original_rhs_shape.collapsed_from(2));

    // 2.  Configuration for transpose of lhs/rhs
    // ------------------------------------------------------
    // Initialise transposed TensorInfo class for aux tensors (intermediary tensors)
    if(_adj_lhs)
    {
        // Setup transpose LHS
        _transpose_kernel_lhs = std::make_unique<cpu::kernels::CpuTransposeKernel>();
        _transpose_kernel_lhs->configure(&lhs_to_use, &_lhs_transposed);
    }

    if(_adj_rhs)
    {
        // Setup transpose RHS
        _transpose_kernel_rhs = std::make_unique<cpu::kernels::CpuTransposeKernel>();
        _transpose_kernel_rhs->configure(&rhs_to_use, &_rhs_transposed);
    }

    // 3. Configure assembly kernel using transposed tensors.
    // -----------------------------------------------------
    // Use transposed tensors if the corresponding transpose flags are set
    // Fill AsmGemmInfo class object before configuration
    _gemm_info.activation_info = act_info;
    _gemm_info.fast_mode       = settings.fast_math();
    _gemm_info.negated_offsets = false;

    lhs_to_use = (_adj_lhs) ? _lhs_transposed : lhs_to_use;
    rhs_to_use = (_adj_rhs) ? _rhs_transposed : rhs_to_use;

    // Quantized-specific configuration
    if(is_data_type_quantized(lhs->data_type()))
    {
        get_gemmlowp_output_stage_info(&lhs_to_use, &rhs_to_use, &dst_to_use, _gemm_info.activation_info, _gemm_info.output_stage);
    }

    // Configure Asm Kernel
    _asm_glue = std::make_unique<cpu::CpuGemmAssemblyDispatch>();
    _asm_glue->configure(&lhs_to_use, &rhs_to_use, nullptr, &dst_to_use, _gemm_info); // c is nullptr as bias not supported in MatMul

    // Specify memory requirements for intermediate tensors
    auto asm_mem_req = _asm_glue->workspace();
    // Specify memory required by gemm kernel
    int idx = 0;
    for(const auto &aux : asm_mem_req)
    {
        _aux_mem[idx] = aux;
        idx++;
    }
    // Memory requirements for transposed tensors
    _aux_mem[TransposeLHS] = MemoryInfo(offset_int_vec(TransposeLHS), MemoryLifetime::Temporary, lhs->total_size());
    _aux_mem[TransposeRHS] = MemoryInfo(offset_int_vec(TransposeRHS), MemoryLifetime::Temporary, rhs->total_size());
}

void CpuMatMul::run(ITensorPack &tensors)
{
    // Retrieve tensors from tensor pack
    auto lhs = tensors.get_tensor(ACL_SRC_0);
    auto rhs = tensors.get_const_tensor(ACL_SRC_1);
    auto dst = tensors.get_tensor(ACL_DST);

    // Reshape LHS and DST to ensure compatibility with GEMM asm kernel (Batch dimensions is 4th for lhs and dst within asm)
    // Collapse RHS (necessary to support dimensions larger than 3 in gemm assembly)
    lhs->info()->set_tensor_shape(TensorShape(_original_lhs_shape.x(), _original_lhs_shape.y(), 1, _original_lhs_shape.collapsed_from(2).z())); // Collapsed 3+ dimensions into z
    dst->info()->set_tensor_shape(TensorShape(_original_dst_shape.x(), _original_dst_shape.y(), 1, _original_dst_shape.collapsed_from(2).z())); // Collapsed 3+ dimensions into z
    rhs->info()->set_tensor_shape(_original_rhs_shape.collapsed_from(2));

    // Initialise object to handle stored transposed tensors in auxillary memory
    CpuAuxTensorHandler lhs_transposed(offset_int_vec(TransposeLHS), _lhs_transposed, tensors, true);
    CpuAuxTensorHandler rhs_transposed(offset_int_vec(TransposeRHS), _rhs_transposed, tensors, true);

    // Create tensor pack for asm kernel
    ITensorPack asm_tensors(tensors);

    // Run transpose lhs if necessary
    if(_adj_lhs)
    {
        ITensorPack lhs_transpose_pack = { { TensorType::ACL_SRC, lhs }, { TensorType::ACL_DST, lhs_transposed.get() } };
        NEScheduler::get().schedule_op(_transpose_kernel_lhs.get(), Window::DimY, _transpose_kernel_lhs->window(), lhs_transpose_pack);
        asm_tensors.add_const_tensor(TensorType::ACL_SRC_0, lhs_transposed.get());
    }
    // Run transpose rhs if necessary
    if(_adj_rhs)
    {
        ITensorPack rhs_transpose_pack = { { TensorType::ACL_SRC, rhs }, { TensorType::ACL_DST, rhs_transposed.get() } };
        NEScheduler::get().schedule_op(_transpose_kernel_rhs.get(), Window::DimY, _transpose_kernel_rhs->window(), rhs_transpose_pack);
        asm_tensors.add_const_tensor(TensorType::ACL_SRC_1, rhs_transposed.get());
    }
    // Run asm kernel
    _asm_glue->run(asm_tensors);

    // Undo reshape of tensors
    dst->info()->set_tensor_shape(_original_dst_shape);
    lhs->info()->set_tensor_shape(_original_lhs_shape);
    rhs->info()->set_tensor_shape(_original_rhs_shape);
}

experimental::MemoryRequirements CpuMatMul::workspace() const
{
    return _aux_mem;
}
} // namespace cpu
} // namespace arm_compute
