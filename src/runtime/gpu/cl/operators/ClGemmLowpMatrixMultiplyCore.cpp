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
#include "src/runtime/gpu/cl/operators/ClGemmLowpMatrixMultiplyCore.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/core/gpu/cl/kernels/ClCastKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmLowpMatrixMultiplyNativeKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmLowpOffsetContributionKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmLowpOffsetContributionOutputStageKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmLowpReductionKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmReshapeRhsMatrixKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/CL/gemm_auto_heuristics/CLGEMMAutoHeuristics.h"
#include "src/runtime/gpu/cl/utils/ClAuxTensorHandler.h"

#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace opencl
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::cl_gemm;
using namespace arm_compute::opencl::kernels;
using namespace arm_compute::experimental;

namespace
{
inline bool validate_gemm_kernel(CLGEMMKernelType kernel_type)
{
    switch(kernel_type)
    {
        case CLGEMMKernelType::NATIVE:
        case CLGEMMKernelType::RESHAPED_ONLY_RHS:
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
inline CLGEMMKernelType auto_select_gemm_kernel(auto_heuristics::CommonQuery query, bool reshape_b_only_on_first_run)
{
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

// Validate lhs_info and rhs_info for native kernel
inline bool validate_lhs_rhs_info_native(const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, const ITensorInfo *a, const ITensorInfo *b, const GEMMReshapeInfo &reshape_info)
{
    // Validate GEMMLHSMatrixInfo and GEMMRHSMatrixInfo for reshaped only rhs kernel
    TensorInfo mm_result_s32_info{};
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(mm_result_s32_info, a->clone()->set_tensor_shape(compute_mm_shape(*a, *b, false, reshape_info)).set_data_type(DataType::S32));
    // Validate mm kernel
    // NOTE: Ignore all other parameters (eg. output stage etc.) and only validate lhs and rhs info
    // NOTE: This assumes:
    //  1. lhs and rhs info's validity does not depend on these other parameters and vice versa(in CLGEMMLowpMatrixMultiplyNativeKernel.cpp validate_arguments).
    //  2. lhs and rhs info does not cause window and padding issues through side effects (in CLGEMMLowpMatrixMultiplyNativeKernel.cpp validate_and_configure_window).
    if(!bool(ClGemmLowpMatrixMultiplyNativeKernel::validate(a, b, &mm_result_s32_info, lhs_info, rhs_info, reshape_info)))
    {
        return false;
    }
    return true;
}

// Automatically select between mlgo (prioritized) and default heuristics for native kernel configs
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> auto_select_gemm_config_native(auto_heuristics::CommonQuery query, const ITensorInfo *a, const ITensorInfo *b, const GEMMReshapeInfo &reshape_info)
{
    auto config = auto_heuristics::select_mlgo_gemm_config_native(query);
    if(config)
    {
        if(validate_lhs_rhs_info_native(config.lhs_info, config.rhs_info, a, b, reshape_info))
        {
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use native config from mlgo heuristics: LHS info: %s ; RHS info: %s ", to_string(config.lhs_info).c_str(), to_string(config.rhs_info).c_str());
            return { config.lhs_info, config.rhs_info };
        }
    }
    config = auto_heuristics::select_default_gemm_config_native(query);
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use native config from default heuristics: LHS info: %s ; RHS info: %s ", to_string(config.lhs_info).c_str(), to_string(config.rhs_info).c_str());
    return { config.lhs_info, config.rhs_info };
}

// Validate lhs_info and rhs_info for reshaped only rhs kernel
inline bool validate_lhs_rhs_info_reshaped_only_rhs(const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *output,
                                                    unsigned int m, unsigned int n, unsigned int k, bool reinterpret_input_as_3d, int depth_output_gemm3d)
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
    // NOTE: Ignore all other parameters (eg. depth_output_gemm3d, output stage etc.) and only validate lhs and rhs info
    // NOTE: This assumes:
    //  1. lhs and rhs info's validity does not depend on these other parameters and vice versa(in ClGemmLowpMatrixMultiplyReshapedOnlyRHSKernel.cpp validate_arguments).
    //  2. lhs and rhs info does not cause window and padding issues through side effects (in ClGemmLowpMatrixMultiplyReshapedOnlyRHSKernel.cpp validate_and_configure_window).
    GEMMKernelInfo gemm_kernel_info;
    gemm_kernel_info.m                       = m;
    gemm_kernel_info.n                       = n;
    gemm_kernel_info.k                       = k;
    gemm_kernel_info.reinterpret_input_as_3d = reinterpret_input_as_3d;
    gemm_kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    gemm_kernel_info.lhs_info                = lhs_info;
    gemm_kernel_info.rhs_info                = rhs_info;
    // Since we ignore the output stage, output data type has to be S32 to pass the validation
    TensorInfo output_info_copy(*output);
    output_info_copy.set_data_type(DataType::S32);
    if(!bool(ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel::validate(a, &tmp_b_info, &output_info_copy, gemm_kernel_info)))
    {
        return false;
    }
    return true;
}

// Automatically select between mlgo (prioritized) and default heuristics for reshaped only rhs kernel configs
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> auto_select_gemm_config_reshaped_only_rhs(auto_heuristics::CommonQuery query, bool reinterpret_input_as_3d, int depth_output_gemm3d,
                                                                                          const ITensorInfo *a,
                                                                                          const ITensorInfo *b, const ITensorInfo *output)
{
    auto config = auto_heuristics::select_mlgo_gemm_config_reshaped_only_rhs(query);
    if(config)
    {
        if(validate_lhs_rhs_info_reshaped_only_rhs(config.lhs_info, config.rhs_info, a, b, output, query.m, query.n, query.k, reinterpret_input_as_3d, depth_output_gemm3d))
        {
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use reshaped_only_rhs config from mlgo heuristics: LHS info: %s ; RHS info: %s ", to_string(config.lhs_info).c_str(), to_string(config.rhs_info).c_str());
            return { config.lhs_info, config.rhs_info };
        }
    }
    config = auto_heuristics::select_default_gemm_config_reshaped_only_rhs(query);
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Use reshaped_only_rhs config from default heuristics: LHS info: %s ; RHS info: %s ", to_string(config.lhs_info).c_str(), to_string(config.rhs_info).c_str());
    return { config.lhs_info, config.rhs_info };
}

inline bool is_gemm_reshaped(CLGEMMKernelType kernel_type)
{
    switch(kernel_type)
    {
        case CLGEMMKernelType::NATIVE:
            return false;
        case CLGEMMKernelType::RESHAPED_ONLY_RHS:
            return true;
        default:
            ARM_COMPUTE_ERROR("Not supported gemmlowp kernel!");
    }
}
} // namespace

ClGemmLowpMatrixMultiplyCore::ClGemmLowpMatrixMultiplyCore()
    : _weights_to_qasymm8(std::make_unique<ClCastKernel>()),
      _mm_native_kernel(std::make_unique<ClGemmLowpMatrixMultiplyNativeKernel>()),
      _mm_reshaped_only_rhs_kernel(std::make_unique<ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel>()),
      _mtx_b_reshape_kernel(std::make_unique<ClGemmReshapeRhsMatrixKernel>()),
      _mtx_a_reduction_kernel(std::make_unique<ClGemmLowpMatrixAReductionKernel>()),
      _mtx_b_reduction_kernel(std::make_unique<ClGemmLowpMatrixBReductionKernel>()),
      _offset_contribution_kernel(std::make_unique<ClGemmLowpOffsetContributionKernel>()),
      _offset_contribution_output_stage_kernel(std::make_unique<ClGemmLowpOffsetContributionOutputStageKernel>()),
      _aux_mem(AuxTensorIdx::Count)
{
}

ClGemmLowpMatrixMultiplyCore::~ClGemmLowpMatrixMultiplyCore() = default;

void ClGemmLowpMatrixMultiplyCore::configure(const CLCompileContext &compile_context,
                                             ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output,
                                             const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
    ARM_COMPUTE_ERROR_THROW_ON(ClGemmLowpMatrixMultiplyCore::validate(a, b, c != nullptr ? c : nullptr, output, gemm_info));

    _reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();
    _a_offset                    = a->quantization_info().uniform().offset;
    _convert_to_qasymm8          = is_data_type_quantized_per_channel(b->data_type()) && is_data_type_quantized_symmetric(b->data_type())
                                   && a->data_type() == DataType::QASYMM8;
    _b_offset  = _convert_to_qasymm8 ? -128 : b->quantization_info().uniform().offset;
    _gemm_info = gemm_info;

    // Get the GPU target
    const GPUTarget gpu_target = CLScheduler::get().target();

    // Set the target for the kernels
    _mm_native_kernel->set_target(gpu_target);
    _mm_reshaped_only_rhs_kernel->set_target(gpu_target);

    GEMMRHSMatrixInfo rhs_info;
    GEMMLHSMatrixInfo lhs_info;

    // Arguments used by GEMMReshapeInfo
    // If we pass the matrix A and matrix B reshaped to CLGEMMMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to CLGEMMReshapeInfo
    // in order to know how the matrices have been reshaped
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();

    const auto reshape_info = GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d);

    // Check if we need to reshape the matrix A and matrix B
    _is_gemm_reshaped = is_gemm_reshaped(auto_select_gemm_kernel(auto_heuristics::CommonQuery{ gpu_target, a->data_type(), m, n, k, batch_size }, _reshape_b_only_on_first_run));

    if(_convert_to_qasymm8)
    {
        // Set data type for converted weights
        _qasymm8_weights = *b;
        _qasymm8_weights.set_data_type(DataType::QASYMM8);
        _weights_to_qasymm8->configure(compile_context, b, &_qasymm8_weights, ConvertPolicy::WRAP);
    }

    ITensorInfo *matrix_b = _convert_to_qasymm8 ? &_qasymm8_weights : b;
    if(_is_gemm_reshaped)
    {
        matrix_b = &_tmp_b;

        // Pick up the GEMM configuration
        // It doesn't matter whether Datatype is DataType::QASYMM8 or DataType::QASYMM8_SIGNED, since it only affect the shape configuration
        std::tie(lhs_info, rhs_info) = auto_select_gemm_config_reshaped_only_rhs(auto_heuristics::CommonQuery{ gpu_target, DataType::QASYMM8, m, n, k, batch_size }, reinterpret_input_as_3d,
                                                                                 depth_output_gemm3d,
                                                                                 a, _convert_to_qasymm8 ? &_qasymm8_weights : b, output);

        // Configure reshape RHS kernel
        _mtx_b_reshape_kernel->configure(compile_context, _convert_to_qasymm8 ? &_qasymm8_weights : b, &_tmp_b, rhs_info);
    }

    // Using default reduction info
    const GEMMLowpReductionKernelInfo reduction_info {};

    // Initialize matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0)
    {
        _vector_sum_col = TensorInfo(compute_reductionA_shape(*b), 1, DataType::S32);

        // Configure Matrix B reduction kernel
        _mtx_b_reduction_kernel->configure(compile_context, _convert_to_qasymm8 ? &_qasymm8_weights : b, &_vector_sum_col, reduction_info);
    }

    // Initialize Matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        _vector_sum_row = TensorInfo(compute_reductionB_shape(*a), 1, DataType::S32);

        // Configure matrix A reduction kernel
        _mtx_a_reduction_kernel->configure(compile_context, a, &_vector_sum_row, reduction_info);
    }

    GEMMKernelInfo gemm_kernel_info;
    gemm_kernel_info.m                       = m;
    gemm_kernel_info.n                       = n;
    gemm_kernel_info.k                       = k;
    gemm_kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    gemm_kernel_info.reinterpret_input_as_3d = reinterpret_input_as_3d;
    gemm_kernel_info.lhs_info                = lhs_info;
    gemm_kernel_info.rhs_info                = rhs_info;
    gemm_kernel_info.a_offset                = _a_offset;
    gemm_kernel_info.b_offset                = _b_offset;
    // If GEMMLowpOutputStage != NONE, fuse the offset contribution with the output stage
    if(gemm_info.gemmlowp_output_stage().type != GEMMLowpOutputStageType::NONE)
    {
        // Configure offset contribution kernel
        const size_t num_filters = (gemm_info.gemmlowp_output_stage().is_quantized_per_channel) ? gemm_info.gemmlowp_output_stage().gemmlowp_multipliers.size() : 1;

        _gemm_output_stage_multipliers = TensorInfo(TensorShape(num_filters), 1, DataType::S32);
        _gemm_output_stage_shifts      = TensorInfo(TensorShape(num_filters), 1, DataType::S32);

        GEMMLowpOutputStageInfo gemmlowp_output_stage = gemm_info.gemmlowp_output_stage();
        gemmlowp_output_stage.output_data_type        = a->data_type();
        if(num_filters == 1)
        {
            // Per-channel quantization with OFM == 1 is equivalent to uniform quantization.
            // Setting this flag to false prevents the kernel from adding useless padding to the output multipliers and shifts
            gemmlowp_output_stage.is_quantized_per_channel = false;
        }

        gemm_kernel_info.output_stage = gemmlowp_output_stage;

        if(_is_gemm_reshaped && gemmlowp_output_stage.type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
        {
            // Configure and tune matrix multiply kernel with fused output stage
            _mm_reshaped_only_rhs_kernel->configure(compile_context, a, matrix_b, output, gemm_kernel_info, _a_offset == 0 ? nullptr : &_vector_sum_col,
                                                    _b_offset == 0 ? nullptr : &_vector_sum_row, c != nullptr ? c : nullptr, &_gemm_output_stage_multipliers, &_gemm_output_stage_shifts);
        }
        else
        {
            _run_output_stage = true;

            if(_is_gemm_reshaped)
            {
                _mm_reshaped_only_rhs_kernel->configure(compile_context, a, matrix_b, &_mm_result_s32, gemm_kernel_info);
            }
            else
            {
                // Pick up the GEMM configuration
                // It doesn't matter whether Datatype is DataType::QASYMM8 or DataType::QASYMM8_SIGNED, since it only affect the shape configuration
                std::tie(lhs_info, rhs_info) = auto_select_gemm_config_native(auto_heuristics::CommonQuery{ gpu_target, DataType::QASYMM8, m, n, k, batch_size },
                                                                              a, _convert_to_qasymm8 ? &_qasymm8_weights : matrix_b, reshape_info);

                // Configure matrix multiply kernel
                _mm_native_kernel->configure(compile_context, a, matrix_b, &_mm_result_s32, lhs_info, rhs_info, reshape_info);

                _offset_contribution_output_stage_kernel->configure(compile_context, &_mm_result_s32, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row,
                                                                    c != nullptr ? c : nullptr, output, a->dimension(0), _a_offset, _b_offset, gemmlowp_output_stage,
                                                                    &_gemm_output_stage_multipliers, &_gemm_output_stage_shifts);
            }
        }
    }
    else
    {
        _run_offset_contribution = true;
        if(_is_gemm_reshaped)
        {
            // Configure and tune matrix multiply kernel
            _mm_reshaped_only_rhs_kernel->configure(compile_context, a, matrix_b, output, gemm_kernel_info);
        }
        else
        {
            // Pick up the GEMM configuration
            // It doesn't matter whether Datatype is DataType::QASYMM8 or DataType::QASYMM8_SIGNED, since it only affect the shape configuration
            std::tie(lhs_info, rhs_info) = auto_select_gemm_config_native(auto_heuristics::CommonQuery{ gpu_target, DataType::QASYMM8, m, n, k, batch_size },
                                                                          a, _convert_to_qasymm8 ? &_qasymm8_weights : b, reshape_info);

            // Configure matrix multiply kernel
            _mm_native_kernel->configure(compile_context, a, matrix_b, output, lhs_info, rhs_info, reshape_info);
        }

        // Configure offset contribution kernel
        _offset_contribution_kernel->configure(compile_context, output, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row,
                                               c != nullptr ? c : nullptr, a->dimension(0), _a_offset, _b_offset);
    }

    // Request memory
    _aux_mem[RhsQAsymm8] = MemoryInfo(offset_int_vec(RhsQAsymm8), _reshape_b_only_on_first_run ? MemoryLifetime::Persistent : MemoryLifetime::Temporary, _qasymm8_weights.total_size());
    if(_is_gemm_reshaped)
    {
        // Overwrite Rhs as prepare if gemm is reshaped as there will be a two-step transformation
        _aux_mem[RhsQAsymm8] = MemoryInfo(offset_int_vec(RhsQAsymm8), _reshape_b_only_on_first_run ? MemoryLifetime::Prepare : MemoryLifetime::Temporary, _qasymm8_weights.total_size());
        _aux_mem[RhsReshape] = MemoryInfo(offset_int_vec(RhsReshape), _reshape_b_only_on_first_run ? MemoryLifetime::Persistent : MemoryLifetime::Temporary, _tmp_b.total_size());
    }
    if(_a_offset != 0)
    {
        _aux_mem[VecSumCol] = MemoryInfo(offset_int_vec(VecSumCol), _reshape_b_only_on_first_run ? MemoryLifetime::Persistent : MemoryLifetime::Temporary, _vector_sum_col.total_size());
    }
    if(_b_offset != 0)
    {
        _aux_mem[VecSumRow] = MemoryInfo(offset_int_vec(VecSumRow), MemoryLifetime::Temporary, _vector_sum_row.total_size());
    }
    _aux_mem[ResultS32]   = MemoryInfo(offset_int_vec(ResultS32), MemoryLifetime::Temporary, _mm_result_s32.total_size());
    _aux_mem[Multipliers] = MemoryInfo(offset_int_vec(Multipliers), MemoryLifetime::Persistent, _gemm_output_stage_multipliers.total_size());
    _aux_mem[Shifts]      = MemoryInfo(offset_int_vec(Shifts), MemoryLifetime::Persistent, _gemm_output_stage_shifts.total_size());
}

Status ClGemmLowpMatrixMultiplyCore::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(b, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8, DataType::QSYMM8_PER_CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON(a->data_type() == DataType::QASYMM8 && b->data_type() == DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON(a->data_type() == DataType::QASYMM8_SIGNED && b->data_type() == DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    int32_t a_offset = a->quantization_info().uniform().offset;
    int32_t b_offset = b->quantization_info().uniform().offset;

    const ITensorInfo *matrix_a_info = a;

    TensorInfo        tmp_b_info{};
    GEMMRHSMatrixInfo rhs_info;
    GEMMLHSMatrixInfo lhs_info;

    // Get the GPU target
    const GPUTarget gpu_target = CLScheduler::get().target();

    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();

    bool reshape_matrix_b = is_gemm_reshaped(auto_select_gemm_kernel(auto_heuristics::CommonQuery{ gpu_target, a->data_type(), m, n, k, batch_size }, gemm_info.reshape_b_only_on_first_run()));

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d);

    bool convert_to_qasymm8 = is_data_type_quantized_per_channel(b->data_type()) && is_data_type_quantized_symmetric(b->data_type())
                              && is_data_type_quantized_asymmetric(a->data_type());
    TensorInfo weights_info(*b);
    if(convert_to_qasymm8)
    {
        b_offset = -128;
        weights_info.set_data_type(DataType::QASYMM8);
        ARM_COMPUTE_RETURN_ON_ERROR(ClCastKernel::validate(b, &weights_info, ConvertPolicy::WRAP));
    }
    const ITensorInfo *matrix_b_info = &weights_info;
    if(reshape_matrix_b)
    {
        matrix_b_info = &tmp_b_info;

        // Pick up the GEMM configuration
        // NOTE: No need to validate mlgo configurations as they automatically fall back to default heuristics if validation fails
        // It doesn't matter whether Datatype is DataType::QASYMM8 or DataType::QASYMM8_SIGNED, since it only affect the shape configuration
        const auto res = select_default_gemm_config_reshaped_only_rhs(auto_heuristics::CommonQuery{ gpu_target, DataType::QASYMM8, m, n, k, batch_size });
        lhs_info       = res.lhs_info;
        rhs_info       = res.rhs_info;

        // Validate reshape RHS kernel
        auto_init_if_empty(tmp_b_info, weights_info.clone()->set_tensor_shape(compute_rhs_reshaped_shape(weights_info, rhs_info)));
        ARM_COMPUTE_RETURN_ON_ERROR(ClGemmReshapeRhsMatrixKernel::validate(&weights_info, &tmp_b_info, rhs_info));
    }

    TensorInfo info_vector_sum_col{};
    TensorInfo info_vector_sum_row{};

    const GEMMLowpReductionKernelInfo reduction_info;
    // Validate matrix B reduction kernel only if _a_offset is not equal to 0
    if(a_offset != 0)
    {
        info_vector_sum_col = TensorInfo(compute_reductionA_shape(weights_info), 1, DataType::S32);

        // Configure Matrix B reduction kernel
        ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpMatrixBReductionKernel::validate(&weights_info, &info_vector_sum_col, reduction_info));
    }

    // Validate Matrix A reduction kernel only if _b_offset is not equal to 0
    if(b_offset != 0)
    {
        info_vector_sum_row = TensorInfo(compute_reductionB_shape(*a), 1, DataType::S32);

        // Configure matrix A reduction kernel
        ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpMatrixAReductionKernel::validate(a, &info_vector_sum_row, reduction_info));
    }

    GEMMKernelInfo gemm_kernel_info;
    gemm_kernel_info.m                       = m;
    gemm_kernel_info.n                       = n;
    gemm_kernel_info.k                       = k;
    gemm_kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    gemm_kernel_info.reinterpret_input_as_3d = reinterpret_input_as_3d;
    gemm_kernel_info.lhs_info                = lhs_info;
    gemm_kernel_info.rhs_info                = rhs_info;
    gemm_kernel_info.a_offset                = a_offset;
    gemm_kernel_info.b_offset                = b_offset;
    if(gemm_info.gemmlowp_output_stage().type != GEMMLowpOutputStageType::NONE)
    {
        const size_t num_filters = (gemm_info.gemmlowp_output_stage().is_quantized_per_channel) ? gemm_info.gemmlowp_output_stage().gemmlowp_multipliers.size() : 1;

        const TensorInfo gemm_output_stage_multipliers_shifts_info(TensorInfo(TensorShape(num_filters), 1, DataType::S32));

        GEMMLowpOutputStageInfo gemmlowp_output_stage = gemm_info.gemmlowp_output_stage();
        gemmlowp_output_stage.output_data_type        = a->data_type();

        gemm_kernel_info.output_stage = gemmlowp_output_stage;
        if(reshape_matrix_b && gemm_info.gemmlowp_output_stage().type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel::validate(matrix_a_info, matrix_b_info, output, gemm_kernel_info,
                                                                                                a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                                                b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                                                c,
                                                                                                &gemm_output_stage_multipliers_shifts_info,
                                                                                                &gemm_output_stage_multipliers_shifts_info));
        }
        else
        {
            TensorInfo mm_result_s32_info{};

            if(reshape_matrix_b)
            {
                // Output tensor auto inizialitation if not yet initialized
                auto_init_if_empty(mm_result_s32_info, a->clone()->set_tensor_shape(compute_mm_shape(*matrix_a_info, *matrix_b_info, reshape_info)).set_data_type(DataType::S32));

                // Validate matrix multiply
                ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info, gemm_kernel_info));
            }
            else
            {
                // Output tensor auto inizialitation if not yet initialized
                auto_init_if_empty(mm_result_s32_info, a->clone()->set_tensor_shape(compute_mm_shape(*matrix_a_info, *matrix_b_info, false, reshape_info)).set_data_type(DataType::S32));

                // Pick up the GEMM configuration
                // NOTE: No need to validate mlgo configurations as they automatically fall back to default heuristics if validation fails
                // It doesn't matter whether Datatype is DataType::QASYMM8 or DataType::QASYMM8_SIGNED, since it only affect the shape configuration
                const auto res = select_default_gemm_config_native(auto_heuristics::CommonQuery{ gpu_target, DataType::QASYMM8, m, n, k, batch_size });
                lhs_info       = res.lhs_info;
                rhs_info       = res.rhs_info;

                // Validate matrix multiply
                ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpMatrixMultiplyNativeKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info, lhs_info, rhs_info, reshape_info));
            }

            // Validate offset contribution kernel
            ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpOffsetContributionOutputStageKernel::validate(&mm_result_s32_info,
                                                                                                a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                                                b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                                                c,
                                                                                                output,
                                                                                                a_offset, b_offset,
                                                                                                gemmlowp_output_stage,
                                                                                                &gemm_output_stage_multipliers_shifts_info,
                                                                                                &gemm_output_stage_multipliers_shifts_info));
        }
    }
    else
    {
        if(reshape_matrix_b)
        {
            // Validate matrix multiply
            ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel::validate(matrix_a_info, matrix_b_info, output, gemm_kernel_info));
        }
        else
        {
            // Pick up the GEMM configuration
            // It doesn't matter whether Datatype is DataType::QASYMM8 or DataType::QASYMM8_SIGNED, since it only affect the shape configuration
            const auto res = select_default_gemm_config_native(auto_heuristics::CommonQuery{ gpu_target, DataType::QASYMM8, m, n, k, batch_size });
            lhs_info       = res.lhs_info;
            rhs_info       = res.rhs_info;

            // Validate matrix multiply
            ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpMatrixMultiplyNativeKernel::validate(matrix_a_info, matrix_b_info, output, lhs_info, rhs_info, reshape_info));
        }

        if(output->total_size() != 0)
        {
            // Validate offset contribution kernel
            ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpOffsetContributionKernel::validate(output,
                                                                                     a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                                     b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                                     c,
                                                                                     a_offset, b_offset));
        }
    }

    return Status{};
}

void ClGemmLowpMatrixMultiplyCore::run(ITensorPack &tensors)
{
    const ITensor *a   = tensors.get_const_tensor(ACL_SRC_0);
    const ITensor *b   = tensors.get_const_tensor(ACL_SRC_1);
    const ITensor *c   = tensors.get_const_tensor(ACL_SRC_2);
    ITensor       *dst = tensors.get_tensor(ACL_DST);

    ARM_COMPUTE_ERROR_ON_NULLPTR(a, dst);

    CLAuxTensorHandler vec_sum_col(offset_int_vec(VecSumCol), _vector_sum_col, tensors, true);
    CLAuxTensorHandler vec_sum_row(offset_int_vec(VecSumRow), _vector_sum_row, tensors, true);
    CLAuxTensorHandler rhs_qasymm8(offset_int_vec(RhsQAsymm8), _qasymm8_weights, tensors, true);
    CLAuxTensorHandler tmp_b(offset_int_vec(RhsReshape), _tmp_b, tensors, true);
    CLAuxTensorHandler res32(offset_int_vec(ResultS32), _mm_result_s32, tensors, true);
    CLAuxTensorHandler shifts(offset_int_vec(Shifts), _gemm_output_stage_shifts, tensors, true);
    CLAuxTensorHandler multipliers(offset_int_vec(Multipliers), _gemm_output_stage_multipliers, tensors, true);

    // Prepare the consts if needed
    prepare(tensors);

    const ITensor *matrix_a = a;
    const ITensor *matrix_b = _convert_to_qasymm8 ? rhs_qasymm8.get() : b;

    if(_is_gemm_reshaped)
    {
        matrix_b = tmp_b.get();
        if(!_reshape_b_only_on_first_run)
        {
            // Run reshape matrix B
            ITensorPack mtx_b_reshape_pack =
            {
                { TensorType::ACL_SRC, _convert_to_qasymm8 ? rhs_qasymm8.get() : b },
                { TensorType::ACL_DST, tmp_b.get() }
            };
            CLScheduler::get().enqueue_op(*_mtx_b_reshape_kernel, mtx_b_reshape_pack, false);
        }
    }

    // Run matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0 && !_reshape_b_only_on_first_run)
    {
        ITensorPack mtx_b_red_pack =
        {
            { TensorType::ACL_SRC, _convert_to_qasymm8 ? rhs_qasymm8.get() : b },
            { TensorType::ACL_DST, vec_sum_col.get() }
        };
        CLScheduler::get().enqueue_op(*_mtx_b_reduction_kernel, mtx_b_red_pack, false);
    }

    // Run matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        ITensorPack mtx_a_red_pack =
        {
            { TensorType::ACL_SRC, matrix_a },
            { TensorType::ACL_DST, vec_sum_row.get() }
        };
        CLScheduler::get().enqueue_op(*_mtx_a_reduction_kernel, mtx_a_red_pack, false);
    }

    // Run matrix multiply
    if(_is_gemm_reshaped)
    {
        ITensorPack gemm_reshaped_pack;
        if(_run_offset_contribution)
        {
            gemm_reshaped_pack = ITensorPack({ { TensorType::ACL_SRC_0, matrix_a },
                { TensorType::ACL_SRC_1, matrix_b },
                { TensorType::ACL_DST, _run_output_stage ? res32.get() : dst }
            });
        }
        else
        {
            gemm_reshaped_pack = ITensorPack(
            {
                { TensorType::ACL_SRC, matrix_a },
                { TensorType::ACL_SRC_1, matrix_b },
                { TensorType::ACL_BIAS, c },
                { TensorType::ACL_VEC_ROW_SUM, _b_offset == 0 ? nullptr : vec_sum_row.get() },
                { TensorType::ACL_VEC_COL_SUM, _a_offset == 0 ? nullptr : vec_sum_col.get() },
                { TensorType::ACL_SHIFTS, shifts.get() },
                { TensorType::ACL_MULTIPLIERS, multipliers.get() },
                { TensorType::ACL_DST, dst },
            });
        }
        CLScheduler::get().enqueue_op(*_mm_reshaped_only_rhs_kernel, gemm_reshaped_pack, false);
    }
    else
    {
        ITensorPack gemm_native_pack =
        {
            { TensorType::ACL_SRC_0, matrix_a },
            { TensorType::ACL_SRC_1, matrix_b },
            { TensorType::ACL_DST, _run_offset_contribution ? dst : res32.get() }
        };
        CLScheduler::get().enqueue_op(*_mm_native_kernel, gemm_native_pack, false);
    }
    if(_run_output_stage)
    {
        // Run offset contribution/output stage kernel
        ITensorPack output_stage_pack =
        {
            { TensorType::ACL_SRC, res32.get() },
            { TensorType::ACL_BIAS, c },
            { TensorType::ACL_VEC_ROW_SUM, _b_offset == 0 ? nullptr : vec_sum_row.get() },
            { TensorType::ACL_VEC_COL_SUM, _a_offset == 0 ? nullptr : vec_sum_col.get() },
            { TensorType::ACL_SHIFTS, shifts.get() },
            { TensorType::ACL_MULTIPLIERS, multipliers.get() },
            { TensorType::ACL_DST, dst },
        };
        CLScheduler::get().enqueue_op(*_offset_contribution_output_stage_kernel, output_stage_pack, true);
    }
    if(_run_offset_contribution)
    {
        // Run offset contribution kernel
        ITensorPack offset_contrib_pack =
        {
            { TensorType::ACL_SRC_DST, dst },
            { TensorType::ACL_BIAS, c },
            { TensorType::ACL_VEC_ROW_SUM, _b_offset == 0 ? nullptr : vec_sum_row.get() },
            { TensorType::ACL_VEC_COL_SUM, _a_offset == 0 ? nullptr : vec_sum_col.get() }
        };
        CLScheduler::get().enqueue_op(*_offset_contribution_kernel, offset_contrib_pack, true);
    }
}

void ClGemmLowpMatrixMultiplyCore::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        auto               b = tensors.get_const_tensor(TensorType::ACL_SRC_1);
        CLAuxTensorHandler tmp_b(offset_int_vec(RhsReshape), _tmp_b, tensors, true);
        CLAuxTensorHandler vec_sum_col(offset_int_vec(VecSumCol), _vector_sum_col, tensors, true);
        CLAuxTensorHandler rhs_qasymm8(offset_int_vec(RhsQAsymm8), _qasymm8_weights, tensors, false);

        ARM_COMPUTE_ERROR_ON_NULLPTR(b);

        if(_convert_to_qasymm8)
        {
            ITensorPack convert_to_qs8_pack = { { ACL_SRC, b }, { ACL_DST, rhs_qasymm8.get() } };
            CLScheduler::get().enqueue_op(*_weights_to_qasymm8, convert_to_qs8_pack, false);
            b->mark_as_unused();
        }

        if(_is_gemm_reshaped && _reshape_b_only_on_first_run)
        {
            // Run reshape kernel and mark original weights tensor as unused
            ITensorPack mtx_b_pack =
            {
                { TensorType::ACL_SRC, _convert_to_qasymm8 ? rhs_qasymm8.get() : b },
                { TensorType::ACL_DST, tmp_b.get() }
            };
            CLScheduler::get().enqueue_op(*_mtx_b_reshape_kernel, mtx_b_pack, false);
            b->mark_as_unused();
        }

        // Run matrix B reduction kernel only if _a_offset is not equal to 0
        if(_a_offset != 0 && _reshape_b_only_on_first_run)
        {
            ITensorPack mtx_b_red_pack =
            {
                { TensorType::ACL_SRC, _convert_to_qasymm8 ? rhs_qasymm8.get() : b },
                { TensorType::ACL_DST, vec_sum_col.get() }
            };
            CLScheduler::get().enqueue_op(*_mtx_b_reduction_kernel, mtx_b_red_pack, false);
        }

        // Compute GEMM output multipliers and shifts for output stage
        {
            const size_t num_filters = (_gemm_info.gemmlowp_output_stage().is_quantized_per_channel) ? _gemm_info.gemmlowp_output_stage().gemmlowp_multipliers.size() : 1;

            CLAuxTensorHandler multipliers(offset_int_vec(Multipliers), _gemm_output_stage_multipliers, tensors, false);
            CLAuxTensorHandler shifts(offset_int_vec(Shifts), _gemm_output_stage_shifts, tensors, false);

            ICLTensor *multiplier_tensor = multipliers.get();
            if(multiplier_tensor != nullptr && multiplier_tensor->info()->total_size() > 0)
            {
                multiplier_tensor->map(CLScheduler::get().queue(), true);
                std::memcpy(multiplier_tensor->ptr_to_element(Coordinates(0)), _gemm_info.gemmlowp_output_stage().gemmlowp_multipliers.data(), num_filters * sizeof(int32_t));
                multiplier_tensor->unmap(CLScheduler::get().queue());
            }

            ICLTensor *shifts_tensor = shifts.get();
            if(shifts.get() != nullptr && shifts_tensor->info()->total_size() > 0)
            {
                shifts_tensor->map(CLScheduler::get().queue(), true);
                std::memcpy(shifts_tensor->ptr_to_element(Coordinates(0)), _gemm_info.gemmlowp_output_stage().gemmlowp_shifts.data(), num_filters * sizeof(int32_t));
                shifts_tensor->unmap(CLScheduler::get().queue());
            }
        }
        CLScheduler::get().queue().finish();
        _is_prepared = true;
    }
}

experimental::MemoryRequirements ClGemmLowpMatrixMultiplyCore::workspace() const
{
    return _aux_mem;
}
} // namespace opencl
} // namespace arm_compute
