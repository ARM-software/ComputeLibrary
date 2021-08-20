/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/runtime/gpu/cl/operators/ClGemm.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/ITensorAllocator.h"

#include "src/common/utils/Log.h"
#include "src/core/gpu/cl/IClKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/utils/helpers/float_ops.h"
#include "src/runtime/CL/gemm/CLGEMMKernelSelection.h"
#include "src/runtime/CL/gemm_auto_heuristics/CLGEMMAutoHeuristics.h"
#include "src/runtime/gpu/cl/utils/ClAuxTensorHandler.h"

#include "support/Cast.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace opencl
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::cl_gemm;
using namespace arm_compute::experimental;
using namespace arm_compute::utils::cast;
using namespace arm_compute::opencl::kernels;

namespace
{
inline bool validate_gemm_kernel(CLGEMMKernelType kernel_type)
{
    switch(kernel_type)
    {
        case CLGEMMKernelType::NATIVE_V1:
        case CLGEMMKernelType::RESHAPED_ONLY_RHS:
        case CLGEMMKernelType::RESHAPED_V1:
        case CLGEMMKernelType::RESHAPED:
        {
            return true;
        }
        default:
        {
            return false;
        }
    }
}
//Automatically select between mlgo (prioritized) and default heuristics for gemm kernel type
inline CLGEMMKernelType auto_select_gemm_kernel(auto_heuristics::CommonQuery query, bool reshape_b_only_on_first_run, bool constant_weights)
{
    if(!constant_weights)
    {
        return CLGEMMKernelType::NATIVE_V1;
    }

    auto gemm_kernel = auto_heuristics::select_mlgo_gemm_kernel(query, reshape_b_only_on_first_run);
    if(bool(gemm_kernel))
    {
        if(validate_gemm_kernel(gemm_kernel.gemm_type))
        {
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use gemm kernel from mlgo heuristics: %s.", to_string(gemm_kernel.gemm_type).c_str());
            return gemm_kernel.gemm_type;
        }
    }
    gemm_kernel = auto_heuristics::select_default_gemm_kernel(query, reshape_b_only_on_first_run);
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use gemm kernel from default heuristics: %s.", to_string(gemm_kernel.gemm_type).c_str());
    return gemm_kernel.gemm_type;
}
// Validate lhs_info and rhs_info for reshaped only rhs kernel
inline bool validate_lhs_rhs_info_reshaped_only_rhs(const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c,
                                                    const ITensorInfo *output, GEMMKernelInfo gemm_kernel_info)
{
    // Validate GEMMLHSMatrixInfo and GEMMRHSMatrixInfo for reshaped only rhs kernel
    TensorInfo tmp_b_info{};
    // Validate reshape RHS kernel
    auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
    if(!bool(ClGemmReshapeRhsMatrixKernel::validate(b, &tmp_b_info, rhs_info)))
    {
        return false;
    }
    // Validate mm kernel
    gemm_kernel_info.lhs_info  = lhs_info;
    gemm_kernel_info.rhs_info  = rhs_info;
    gemm_kernel_info.has_pad_y = false;
    if(!bool(ClGemmMatrixMultiplyReshapedOnlyRhsKernel::validate(a, &tmp_b_info, c, output, 1.f, 0.f, lhs_info, rhs_info, gemm_kernel_info)))
    {
        return false;
    }
    gemm_kernel_info.has_pad_y = true;
    if(!bool(ClGemmMatrixMultiplyReshapedOnlyRhsKernel::validate(a, &tmp_b_info, c, output, 1.f, 0.f, lhs_info, rhs_info, gemm_kernel_info)))
    {
        return false;
    }
    return true;
}

//Automatically select between mlgo (prioritized) and default heuristics for reshaped only rhs kernel configs
inline std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> auto_select_gemm_config_reshaped_only_rhs(auto_heuristics::CommonQuery query, GEMMKernelInfo kernel_info, const ITensorInfo *a,
                                                                                                 const ITensorInfo *b,
                                                                                                 const ITensorInfo *c, const ITensorInfo *output)
{
    auto config = auto_heuristics::select_mlgo_gemm_config_reshaped_only_rhs(query);
    if(config)
    {
        if(validate_lhs_rhs_info_reshaped_only_rhs(config.lhs_info, config.rhs_info, a, b, c, output, kernel_info))
        {
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use reshaped_only_rhs config from mlgo heuristics: LHS info: %s ; RHS info: %s ", to_string(config.lhs_info).c_str(), to_string(config.rhs_info).c_str());
            return { config.lhs_info, config.rhs_info };
        }
    }
    config = auto_heuristics::select_default_gemm_config_reshaped_only_rhs(query);
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use reshaped_only_rhs config from default heuristics: LHS info: %s ; RHS info: %s ", to_string(config.lhs_info).c_str(), to_string(config.rhs_info).c_str());
    return { config.lhs_info, config.rhs_info };
}

// Validate lhs_info and rhs_info for reshaped kernel
inline bool validate_lhs_rhs_info_reshaped(const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c,
                                           const ITensorInfo *output, GEMMKernelInfo gemm_kernel_info, bool reinterpret_input_as_3d)
{
    // Validate GEMMLHSMatrixInfo and GEMMRHSMatrixInfo for reshaped kernel
    TensorInfo tmp_a_info{};
    TensorInfo tmp_b_info{};

    // Validate reshape LHS kernel
    auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_lhs_reshaped_shape(*a, lhs_info, reinterpret_input_as_3d)));
    if(!bool(ClGemmReshapeLhsMatrixKernel::validate(a, &tmp_a_info, lhs_info, reinterpret_input_as_3d)))
    {
        return false;
    }

    // Validate reshape RHS kernel
    auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
    if(!bool(ClGemmReshapeRhsMatrixKernel::validate(b, &tmp_b_info, rhs_info)))
    {
        return false;
    }
    // Validate mm kernel
    gemm_kernel_info.lhs_info = lhs_info;
    gemm_kernel_info.rhs_info = rhs_info;
    if(!bool(ClGemmMatrixMultiplyReshapedKernel::validate(&tmp_a_info, &tmp_b_info, c, output, 1.f, 0.f, lhs_info, rhs_info, gemm_kernel_info)))
    {
        return false;
    }
    return true;
}

//Automatically select between mlgo (prioritized) and default heuristics for reshaped kernel configs
inline std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> auto_select_gemm_config_reshaped(auto_heuristics::CommonQuery query, GEMMKernelInfo kernel_info, const ITensorInfo *a, const ITensorInfo *b,
                                                                                        const ITensorInfo *c, const ITensorInfo *output, bool reinterpret_input_as_3d)
{
    auto config = auto_heuristics::select_mlgo_gemm_config_reshaped(query);
    if(config)
    {
        if(validate_lhs_rhs_info_reshaped(config.lhs_info, config.rhs_info, a, b, c, output, kernel_info, reinterpret_input_as_3d))
        {
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use reshaped config from mlgo heuristics: LHS info: %s ; RHS info: %s ", to_string(config.lhs_info).c_str(), to_string(config.rhs_info).c_str());
            return { config.lhs_info, config.rhs_info };
        }
    }
    config = auto_heuristics::select_default_gemm_config_reshaped(query);
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use reshaped config from default heuristics: LHS info: %s ; RHS info: %s ", to_string(config.lhs_info).c_str(), to_string(config.rhs_info).c_str());
    return { config.lhs_info, config.rhs_info };
}
} // namespace

ClGemm::ClGemm()
    : _mm_kernel(std::make_unique<ClGemmMatrixMultiplyKernel>()),
      _reshape_lhs_kernel(std::make_unique<ClGemmReshapeLhsMatrixKernel>()),
      _reshape_rhs_kernel(std::make_unique<ClGemmReshapeRhsMatrixKernel>()),
      _mm_reshaped_kernel(std::make_unique<ClGemmMatrixMultiplyReshapedKernel>()),
      _mm_reshaped_only_rhs_kernel(std::make_unique<ClGemmMatrixMultiplyReshapedOnlyRhsKernel>()),
      _mm_reshaped_only_rhs_fallback_kernel(std::make_unique<ClGemmMatrixMultiplyReshapedOnlyRhsKernel>()),
      _tmp_a(),
      _tmp_b(),
      _reshape_b_only_on_first_run(false),
      _gemm_kernel_type(CLGEMMKernelType::NATIVE_V1),
      _is_prepared(false),
      _aux_mem(AuxTensorIdx::Count)
{
}

void ClGemm::configure_native_v1(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta,
                                 const GEMMInfo &gemm_info)
{
    const unsigned int m          = gemm_info.reinterpret_input_as_3d() ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n          = b->dimension(0);
    const unsigned int k          = a->dimension(0);
    const GPUTarget    gpu_target = CLScheduler::get().target();

    // Set the target for the kernels
    _mm_kernel->set_target(gpu_target);

    GEMMReshapeInfo reshape_info(m, n, k, 1, 1, gemm_info.depth_output_gemm3d(), gemm_info.reinterpret_input_as_3d(), gemm_info.broadcast_bias());

    // Configure and tune matrix multiply kernel
    _mm_kernel->configure(compile_context, a, b, c, output, alpha, beta, false, reshape_info, gemm_info.fp_mixed_precision(), gemm_info.activation_info());

    // Tune kernel statically
    CLScheduler::get().tune_kernel_static(*_mm_kernel);
}

void ClGemm::configure_reshaped_v1(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta,
                                   const GEMMInfo &gemm_info)
{
    bool               reinterpret_input_as_3d   = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                         = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                         = b->dimension(0);
    const unsigned int k                         = a->dimension(0);
    const int          depth_output_gemm3d       = gemm_info.depth_output_gemm3d();
    const GPUTarget    gpu_target                = CLScheduler::get().target();
    int                mult_transpose1xW_width   = 1;
    int                mult_interleave4x4_height = 1;

    // Set the target for the kernels
    _reshape_lhs_kernel->set_target(gpu_target);
    _mm_kernel->set_target(gpu_target);

    if(get_arch_from_target(gpu_target) == GPUTarget::BIFROST)
    {
        mult_transpose1xW_width   = 4;
        mult_interleave4x4_height = 2;
    }

    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0         = 16 / b->element_size();
    rhs_info.k0         = 1;
    rhs_info.h0         = mult_transpose1xW_width;
    rhs_info.interleave = false;
    rhs_info.transpose  = false;

    GEMMLHSMatrixInfo lhs_info;
    lhs_info.m0         = 4;
    lhs_info.k0         = 4;
    lhs_info.v0         = mult_interleave4x4_height;
    lhs_info.interleave = true;
    lhs_info.transpose  = true;

    GEMMReshapeInfo reshape_info(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, depth_output_gemm3d, false, gemm_info.broadcast_bias());

    // Configure interleave kernel
    _reshape_lhs_kernel->configure(compile_context, a, &_tmp_a, lhs_info, reinterpret_input_as_3d);

    // Configure transpose kernel
    _reshape_rhs_kernel->configure(compile_context, b, &_tmp_b, rhs_info);

    // Configure and tune matrix multiply kernel
    _mm_kernel->configure(compile_context, &_tmp_a, &_tmp_b, c, output, alpha, beta, true, reshape_info, gemm_info.fp_mixed_precision(), gemm_info.activation_info());

    CLScheduler::get().tune_kernel_static(*_mm_kernel);

    // Request memory for LHS and RHS reshape matrix
    _aux_mem[LhsReshape] = MemoryInfo(offset_int_vec(LhsReshape), MemoryLifetime::Temporary, _tmp_a.total_size());
    _aux_mem[RhsReshape] = MemoryInfo(offset_int_vec(RhsReshape), _reshape_b_only_on_first_run ? MemoryLifetime::Persistent : MemoryLifetime::Temporary, _tmp_b.total_size());
}

void ClGemm::configure_reshaped_v2(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta,
                                   const GEMMInfo &gemm_info)
{
    DataType           data_type               = a->data_type();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    bool               broadcast_bias          = gemm_info.broadcast_bias();

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = m;
    kernel_info.n                       = n;
    kernel_info.k                       = k;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = false;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = gemm_info.activation_info();

    // Set the target for the kernels
    _reshape_lhs_kernel->set_target(gpu_target);
    _mm_kernel->set_target(gpu_target);

    GEMMLHSMatrixInfo lhs_info{};
    GEMMRHSMatrixInfo rhs_info{};

    // Pick up the GEMM configuration
    std::tie(lhs_info, rhs_info) = auto_select_gemm_config_reshaped(auto_heuristics::CommonQuery{ gpu_target, data_type, m, n, k, batch_size }, kernel_info, a, b,
                                                                    c, output, gemm_info.reinterpret_input_as_3d());

    _reshape_lhs_kernel->configure(compile_context, a, &_tmp_a, lhs_info, gemm_info.reinterpret_input_as_3d());
    _reshape_rhs_kernel->configure(compile_context, b, &_tmp_b, rhs_info);

    // Configure and tune matrix multiply kernel
    _mm_reshaped_kernel->configure(compile_context, &_tmp_a, &_tmp_b, c, output, alpha, beta, lhs_info, rhs_info, kernel_info);

    // Request memory for LHS and RHS reshape matrix
    _aux_mem[LhsReshape] = MemoryInfo(offset_int_vec(LhsReshape), MemoryLifetime::Temporary, _tmp_a.total_size());
    _aux_mem[RhsReshape] = MemoryInfo(offset_int_vec(RhsReshape), _reshape_b_only_on_first_run ? MemoryLifetime::Persistent : MemoryLifetime::Temporary, _tmp_b.total_size());
}

void ClGemm::configure_reshaped_only_rhs(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta,
                                         const GEMMInfo &gemm_info)
{
    DataType           data_type               = a->data_type();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    bool               broadcast_bias          = gemm_info.broadcast_bias();

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = m;
    kernel_info.n                       = n;
    kernel_info.k                       = k;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = reinterpret_input_as_3d;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = gemm_info.activation_info();

    // Set the target for the kernels
    _mm_kernel->set_target(gpu_target);

    GEMMLHSMatrixInfo lhs_info{};
    GEMMRHSMatrixInfo rhs_info{};

    // Pick up the GEMM configuration
    std::tie(lhs_info, rhs_info) = auto_select_gemm_config_reshaped_only_rhs(auto_heuristics::CommonQuery{ gpu_target, data_type, m, n, k, batch_size }, kernel_info, a, b, c, output);

    // Transpose matrix
    _reshape_rhs_kernel->configure(compile_context, b, &_tmp_b, rhs_info);

    // Configure two variants of CLGEMMMatrixMultiplyReshapedOnlyRHSKernel (has_pad_y = false/true)
    // During the prepare stage we check the padding requirement for the lhs and dst tensors. If they do not have
    // pad y, we dispatch CLGEMMMatrixMultiplyReshapedOnlyRHSKernel with has_pad_y = false

    // Configure matrix multiply kernel with no y padding support
    kernel_info.has_pad_y = false;
    _mm_reshaped_only_rhs_kernel->configure(compile_context, a, &_tmp_b, c, output, alpha, beta, lhs_info, rhs_info, kernel_info);

    // Configure matrix multiply kernel with y padding support
    kernel_info.has_pad_y = true;
    _mm_reshaped_only_rhs_fallback_kernel->configure(compile_context, a, &_tmp_b, c, output, alpha, beta, lhs_info, rhs_info, kernel_info);

    // Request memory for RHS reshape matrix
    _aux_mem[RhsReshape] = MemoryInfo(offset_int_vec(RhsReshape), _reshape_b_only_on_first_run ? MemoryLifetime::Persistent : MemoryLifetime::Temporary, _tmp_b.total_size());
}

Status ClGemm::validate_native_v1(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(output);

    // Get the GPU target
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d, gemm_info.broadcast_bias());

    // Validate matrix multiply
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmMatrixMultiplyKernel::validate(a, b, c, output, alpha, beta,
                                                                     false, reshape_info, gpu_target, gemm_info.fp_mixed_precision(), gemm_info.activation_info()));

    return Status{};
}

Status ClGemm::validate_reshaped_v1(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(output);

    TensorInfo tmp_a_info{};
    TensorInfo tmp_b_info{};

    // Get the GPU target
    const GPUTarget    gpu_target                = CLScheduler::get().target();
    const unsigned int m                         = gemm_info.reinterpret_input_as_3d() ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                         = b->dimension(0);
    const unsigned int k                         = a->dimension(0);
    int                mult_transpose1xW_width   = 1;
    int                mult_interleave4x4_height = 1;
    const int          depth_output_gemm3d       = gemm_info.depth_output_gemm3d();

    if(get_arch_from_target(gpu_target) == GPUTarget::BIFROST)
    {
        mult_transpose1xW_width   = 4;
        mult_interleave4x4_height = 2;
    }

    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0         = 16 / b->element_size();
    rhs_info.k0         = 1;
    rhs_info.h0         = mult_transpose1xW_width;
    rhs_info.interleave = false;
    rhs_info.transpose  = false;

    GEMMLHSMatrixInfo lhs_info;
    lhs_info.m0         = 4;
    lhs_info.k0         = 4;
    lhs_info.v0         = mult_interleave4x4_height;
    lhs_info.interleave = true;
    lhs_info.transpose  = true;

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, depth_output_gemm3d, false, gemm_info.broadcast_bias());

    // Validate interleave kernel
    auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_lhs_reshaped_shape(*a, lhs_info, gemm_info.reinterpret_input_as_3d())));
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmReshapeLhsMatrixKernel::validate(a, &tmp_a_info, lhs_info, gemm_info.reinterpret_input_as_3d()));

    // Validate transpose kernel
    auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmReshapeRhsMatrixKernel::validate(b, &tmp_b_info, rhs_info));

    // Validate matrix multiply
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmMatrixMultiplyKernel::validate(&tmp_a_info, &tmp_b_info, c, output, alpha, beta,
                                                                     true, reshape_info, gpu_target, gemm_info.fp_mixed_precision(), gemm_info.activation_info()));

    return Status{};
}

Status ClGemm::validate_reshaped(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(output);

    TensorInfo tmp_a_info{};
    TensorInfo tmp_b_info{};

    // Get the GPU target
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    DataType           data_type               = a->data_type();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();
    const bool         broadcast_bias          = gemm_info.broadcast_bias();

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = m;
    kernel_info.n                       = n;
    kernel_info.k                       = k;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = false;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = gemm_info.activation_info();

    GEMMLHSMatrixInfo lhs_info;
    GEMMRHSMatrixInfo rhs_info;

    // Pick up the GEMM configuration
    // NOTE: No need to validate mlgo configurations as they automatically fall back to default heuristics if validation fails
    const auto gemm_config = select_default_gemm_config_reshaped(auto_heuristics::CommonQuery{ gpu_target, data_type, m, n, k, batch_size });
    lhs_info               = gemm_config.lhs_info;
    rhs_info               = gemm_config.rhs_info;

    auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_lhs_reshaped_shape(*a, lhs_info, gemm_info.reinterpret_input_as_3d())));
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmReshapeLhsMatrixKernel::validate(a, &tmp_a_info, lhs_info, gemm_info.reinterpret_input_as_3d()));

    auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmReshapeRhsMatrixKernel::validate(b, &tmp_b_info, rhs_info));

    // Validate matrix multiply
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmMatrixMultiplyReshapedKernel::validate(&tmp_a_info, &tmp_b_info, c, output, alpha, beta, lhs_info, rhs_info, kernel_info));

    return Status{};
}

Status ClGemm::validate_reshaped_only_rhs(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(output);

    TensorInfo tmp_b_info{};

    // Get the GPU target
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    const DataType     data_type               = a->data_type();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();
    const bool         broadcast_bias          = gemm_info.broadcast_bias();

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = m;
    kernel_info.n                       = n;
    kernel_info.k                       = k;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = reinterpret_input_as_3d;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = gemm_info.activation_info();

    GEMMLHSMatrixInfo lhs_info;
    GEMMRHSMatrixInfo rhs_info;

    // Pick up the GEMM configuration
    // NOTE: No need to validate mlgo configurations as they automatically fall back to default heuristics if validation fails
    const auto gemm_config = select_default_gemm_config_reshaped_only_rhs(auto_heuristics::CommonQuery{ gpu_target, data_type, m, n, k, batch_size });
    lhs_info               = gemm_config.lhs_info;
    rhs_info               = gemm_config.rhs_info;

    auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmReshapeRhsMatrixKernel::validate(b, &tmp_b_info, rhs_info));

    // Validate matrix multiply
    kernel_info.has_pad_y = false;
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmMatrixMultiplyReshapedOnlyRhsKernel::validate(a, &tmp_b_info, c, output, alpha, beta, lhs_info, rhs_info, kernel_info));

    kernel_info.has_pad_y = true;
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemmMatrixMultiplyReshapedOnlyRhsKernel::validate(a, &tmp_b_info, c, output, alpha, beta, lhs_info, rhs_info, kernel_info));

    return Status{};
}

void ClGemm::configure(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate(a, b, c, output, alpha, beta, gemm_info));

    // Check if we need to reshape the matrix B only on the first run
    _reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();
    _is_prepared                 = gemm_info.retain_internal_weights();

    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);

    // Select GEMMType
    _gemm_kernel_type = auto_select_gemm_kernel(auto_heuristics::CommonQuery{ CLScheduler::get().target(), a->data_type(), m, n, k, batch_size }, _reshape_b_only_on_first_run,
                                                gemm_info.constant_weights());

    const bool fuse_add_c = (!(helpers::float_ops::is_zero(beta)) && c != nullptr);

    ITensorInfo *c_to_use = fuse_add_c ? c : nullptr;

    switch(_gemm_kernel_type)
    {
        case CLGEMMKernelType::NATIVE_V1:
        {
            configure_native_v1(compile_context, a, b, c_to_use, output, alpha, beta, gemm_info);
            break;
        }
        case CLGEMMKernelType::RESHAPED_V1:
        {
            configure_reshaped_v1(compile_context, a, b, c_to_use, output, alpha, beta, gemm_info);
            break;
        }
        case CLGEMMKernelType::RESHAPED:
        {
            configure_reshaped_v2(compile_context, a, b, c_to_use, output, alpha, beta, gemm_info);
            break;
        }
        case CLGEMMKernelType::RESHAPED_ONLY_RHS:
        {
            configure_reshaped_only_rhs(compile_context, a, b, c_to_use, output, alpha, beta, gemm_info);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("GEMMType not supported");
        }
    }
}

Status ClGemm::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    // Get the GPU target
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);

    // Select GEMMType
    CLGEMMKernelType gemm_kernel_type = auto_select_gemm_kernel(auto_heuristics::CommonQuery
    {
        CLScheduler::get().target(), a->data_type(), m, n, k, batch_size,
    },
    gemm_info.reshape_b_only_on_first_run(), gemm_info.constant_weights());

    const bool fuse_add_c = (!(helpers::float_ops::is_zero(beta)) && c != nullptr);

    const ITensorInfo *c_to_use = fuse_add_c ? c : nullptr;

    switch(gemm_kernel_type)
    {
        case CLGEMMKernelType::NATIVE_V1:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(validate_native_v1(a, b, c_to_use, output, alpha, beta, gemm_info));
            break;
        }
        case CLGEMMKernelType::RESHAPED_V1:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(validate_reshaped_v1(a, b, c_to_use, output, alpha, beta, gemm_info));
            break;
        }
        case CLGEMMKernelType::RESHAPED:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(validate_reshaped(a, b, c_to_use, output, alpha, beta, gemm_info));
            break;
        }
        case CLGEMMKernelType::RESHAPED_ONLY_RHS:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(validate_reshaped_only_rhs(a, b, c_to_use, output, alpha, beta, gemm_info));
            break;
        }
        default:
        {
            ARM_COMPUTE_RETURN_ERROR_MSG("GEMMType not supported");
        }
    }

    return Status{};
}

void ClGemm::run(ITensorPack &tensors)
{
    const ITensor *lhs  = tensors.get_const_tensor(ACL_SRC_0);
    const ITensor *rhs  = tensors.get_const_tensor(ACL_SRC_1);
    const ITensor *src2 = tensors.get_const_tensor(ACL_SRC_2);
    ITensor       *dst  = tensors.get_tensor(ACL_DST);

    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, dst);

    CLAuxTensorHandler lhs_reshaped(offset_int_vec(LhsReshape), _tmp_a, tensors, true);
    CLAuxTensorHandler rhs_reshaped(offset_int_vec(RhsReshape), _tmp_b, tensors, true);

    // Prepare the consts if needed
    prepare(tensors);

    // Run matrix multiply kernel
    switch(_gemm_kernel_type)
    {
        case CLGEMMKernelType::NATIVE_V1:
        {
            CLScheduler::get().enqueue_op(*_mm_kernel, tensors, true);
            break;
        }
        case CLGEMMKernelType::RESHAPED_V1:
        case CLGEMMKernelType::RESHAPED:
        {
            // Run interleave kernel
            ITensorPack reshape_lhs_pack{ { ACL_SRC, lhs }, { ACL_DST, lhs_reshaped.get() } };
            CLScheduler::get().enqueue_op(*_reshape_lhs_kernel, reshape_lhs_pack, false);

            if(!_reshape_b_only_on_first_run)
            {
                // Run transpose kernel
                ITensorPack reshape_rhs_pack{ { ACL_SRC, rhs }, { ACL_DST, rhs_reshaped.get() } };
                CLScheduler::get().enqueue_op(*_reshape_rhs_kernel, reshape_rhs_pack, false);
            }

            ITensorPack gemm_reshaped_pack{ { ACL_SRC_0, lhs_reshaped.get() }, { ACL_SRC_1, rhs_reshaped.get() }, { ACL_SRC_2, src2 }, { ACL_DST, dst } };

            if(_gemm_kernel_type == CLGEMMKernelType::RESHAPED)
            {
                CLScheduler::get().enqueue_op(*_mm_reshaped_kernel, gemm_reshaped_pack, true);
            }
            else
            {
                CLScheduler::get().enqueue_op(*_mm_kernel, gemm_reshaped_pack, true);
            }
            break;
        }
        case CLGEMMKernelType::RESHAPED_ONLY_RHS:
        {
            if(!_reshape_b_only_on_first_run)
            {
                // Run transpose kernel
                ITensorPack reshape_rhs_pack{ { ACL_SRC, rhs }, { ACL_DST, rhs_reshaped.get() } };
                CLScheduler::get().enqueue_op(*_reshape_rhs_kernel, reshape_rhs_pack, false);
            }
            // In case of RESHAPED_ONLY_RHS, we need to check the padding requirement
            // Check if the lhs or dst tensors have padding
            const unsigned int cross_plane_pad_lhs = lhs->info()->padding().top + lhs->info()->padding().bottom;
            const unsigned int cross_plane_pad_dst = dst->info()->padding().top + dst->info()->padding().bottom;
            bool               has_pad_y           = (cross_plane_pad_lhs != 0) || (cross_plane_pad_dst != 0);

            ITensorPack gemm_reshaped_onlyrhs_pack{ { ACL_SRC_0, lhs }, { ACL_SRC_1, rhs_reshaped.get() }, { ACL_SRC_2, src2 }, { ACL_DST, dst } };
            if(has_pad_y)
            {
                CLScheduler::get().enqueue_op(*_mm_reshaped_only_rhs_fallback_kernel, gemm_reshaped_onlyrhs_pack, true);
            }
            else
            {
                CLScheduler::get().enqueue_op(*_mm_reshaped_only_rhs_kernel, gemm_reshaped_onlyrhs_pack, true);
            }
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("GEMMType not supported");
        }
    }
}

void ClGemm::prepare(ITensorPack &constants)
{
    if(!_is_prepared)
    {
        const ITensor *src1    = constants.get_const_tensor(ACL_SRC_1);
        ICLTensor     *rhs_aux = utils::cast::polymorphic_downcast<ICLTensor *>(constants.get_tensor(offset_int_vec(RhsReshape)));

        // If memory for RHS is persistent and src1 is provided re-transform else assume that RHS is transformed
        if((_aux_mem[AuxTensorIdx::RhsReshape].lifetime == MemoryLifetime::Persistent) && (src1 != nullptr && rhs_aux != nullptr) && rhs_aux)
        {
            ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Transforming RHS Matrix!");

            CLAuxTensorHandler rhs_reshaped(_tmp_b, *rhs_aux);
            ARM_COMPUTE_ERROR_ON(rhs_reshaped.get()->cl_buffer().get() == nullptr);

            ITensorPack reshape_rhs_pack{ { ACL_SRC, src1 }, { ACL_DST, rhs_reshaped.get() } };
            CLScheduler::get().enqueue_op(*_reshape_rhs_kernel, reshape_rhs_pack, true);
        }
        _is_prepared = true;
    }
}

experimental::MemoryRequirements ClGemm::workspace() const
{
    return _aux_mem;
}
} // namespace opencl
} // namespace arm_compute
