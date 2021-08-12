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
#include "src/runtime/cpu/operators/CpuGemmLowpMatrixMultiplyCore.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"

#include "src/core/cpu/kernels/CpuConvertQuantizedSignednessKernel.h"
#include "src/core/cpu/kernels/CpuGemmInterleave4x4Kernel.h"
#include "src/core/cpu/kernels/CpuGemmLowpMatrixMultiplyKernel.h"
#include "src/core/cpu/kernels/CpuGemmLowpMatrixReductionKernel.h"
#include "src/core/cpu/kernels/CpuGemmLowpOffsetContributionKernel.h"
#include "src/core/cpu/kernels/CpuGemmLowpOffsetContributionOutputStageKernel.h"
#include "src/core/cpu/kernels/CpuGemmTranspose1xWKernel.h"
#include "src/runtime/cpu/operators/CpuActivation.h"
#include "src/runtime/cpu/operators/internal/CpuGemmAssemblyDispatch.h"
#include "src/runtime/cpu/utils/CpuAuxTensorHandler.h"

using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{
namespace
{
cpu::AsmGemmInfo init_assembly_metadata(const GEMMInfo &info)
{
    cpu::AsmGemmInfo asm_info;
    asm_info.method                  = cpu::AsmConvMethod::Im2Col;
    asm_info.reinterpret_input_as_3d = info.reinterpret_input_as_3d();
    asm_info.depth_output_gemm3d     = info.depth_output_gemm3d();
    asm_info.activation_info         = info.activation_info();
    asm_info.output_stage            = info.gemmlowp_output_stage();
    asm_info.fast_mode               = info.fast_math();

    return asm_info;
}
} // namespace

CpuGemmLowpMatrixMultiplyCore::CpuGemmLowpMatrixMultiplyCore()
    : _asm_glue(std::make_unique<CpuGemmAssemblyDispatch>()),
      _mm_kernel(),
      _mtx_a_reshape_kernel(),
      _mtx_b_reshape_kernel(),
      _mtx_a_reduction_kernel(),
      _mtx_b_reduction_kernel(),
      _offset_contribution_kernel(),
      _offset_contribution_output_stage_kernel(),
      _activation_func(),
      _convert_to_signed_asymm(),
      _convert_from_signed_asymm(),
      _vector_sum_col(),
      _vector_sum_row(),
      _tmp_a(),
      _tmp_b(),
      _mm_result_s32(),
      _signed_a(),
      _signed_output(),
      _a_offset(0),
      _b_offset(0),
      _run_vector_matrix_multiplication(false),
      _assembly_path(false),
      _fused_assembly_path(false),
      _reshape_b_only_on_first_run(false),
      _is_prepared(false),
      _fuse_output_stage(false),
      _run_activation(false),
      _flip_signedness(false),
      _gemm_info(),
      _aux_mem(Count)
{
}
CpuGemmLowpMatrixMultiplyCore::~CpuGemmLowpMatrixMultiplyCore() = default;

void CpuGemmLowpMatrixMultiplyCore::configure(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, ITensorInfo *dst, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, dst);
    ARM_COMPUTE_ERROR_THROW_ON(CpuGemmLowpMatrixMultiplyCore::validate(a, b, c, dst, gemm_info));

    const ITensorInfo *matrix_a = a;
    const ITensorInfo *matrix_b = b;
    GEMMInfo           info     = gemm_info;

    // Set internal variables
    _a_offset                         = a->quantization_info().uniform().offset;
    _b_offset                         = b->quantization_info().uniform().offset;
    _run_vector_matrix_multiplication = a->dimension(1) < 2;
    _reshape_b_only_on_first_run      = info.reshape_b_only_on_first_run();
    _is_prepared                      = false;
    _fused_assembly_path              = false;
    _flip_signedness                  = is_data_type_quantized_per_channel(b->data_type()) && (a->data_type() == DataType::QASYMM8) && _reshape_b_only_on_first_run;
    _gemm_info                        = gemm_info;

    _asm_glue = std::make_unique<cpu::CpuGemmAssemblyDispatch>();

    const ITensorInfo *a_to_use = a;

    // Convert to QASYMM8 -> QASYMM8_SIGNED and back
    if(_flip_signedness)
    {
        const int32_t                 offset_correction = 128;
        const DataType                dt                = DataType::QASYMM8_SIGNED;
        const UniformQuantizationInfo iqinfo            = a_to_use->quantization_info().uniform();

        _signed_a                = a_to_use->clone()->set_data_type(dt).set_quantization_info(QuantizationInfo(iqinfo.scale, iqinfo.offset + offset_correction));
        _convert_to_signed_asymm = std::make_unique<kernels::CpuConvertQuantizedSignednessKernel>();
        _convert_to_signed_asymm->configure(a_to_use, &_signed_a);
        a_to_use  = &_signed_a;
        _a_offset = _signed_a.quantization_info().uniform().offset;

        const UniformQuantizationInfo oqinfo = dst->quantization_info().uniform();
        _signed_output                       = dst->clone()->set_data_type(dt).set_quantization_info(QuantizationInfo(oqinfo.scale, oqinfo.offset - offset_correction));

        // Output stage correction
        GEMMLowpOutputStageInfo output_stage_corr = info.gemmlowp_output_stage();
        output_stage_corr.gemmlowp_offset         = _signed_output.quantization_info().uniform().offset;
        output_stage_corr.gemmlowp_min_bound -= offset_correction;
        output_stage_corr.gemmlowp_max_bound -= offset_correction;
        info.set_gemmlowp_output_stage(output_stage_corr);

        // Update matrix a
        matrix_a = &_signed_a;
    }

    // If GEMMLowpOutputStage != NONE, fuse the offset contribution with the output stage
    if(info.gemmlowp_output_stage().type != GEMMLowpOutputStageType::NONE)
    {
        _fuse_output_stage = true;
        _mm_result_s32     = TensorInfo(dst->tensor_shape(), 1, DataType::S32);
    }

    // Initialize assembly kernel meta-data
    const cpu::AsmGemmInfo asm_info = init_assembly_metadata(gemm_info);
#ifdef __aarch64__
    switch(a->data_type())
    {
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::U8:
        case DataType::S8:
        {
            if(is_data_type_quantized_asymmetric(a_to_use->data_type()) && info.gemmlowp_output_stage().type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
            {
                auto c_info_to_use = c == nullptr ? nullptr : c;
                _asm_glue->configure(a_to_use, b, c_info_to_use, dst, asm_info);
                _fused_assembly_path = _asm_glue->is_configured();
            }
            else
            {
                auto output_to_use = (_fuse_output_stage ? &_mm_result_s32 : dst);
                _asm_glue->configure(a_to_use, b, nullptr, output_to_use, asm_info);
            }
            _assembly_path = _asm_glue->is_configured();
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Datatype not supported");
            break;
        }
    }
#endif /* __aarch64__ */
    if(!(_assembly_path || _run_vector_matrix_multiplication))
    {
        matrix_a = &_tmp_a;
        matrix_b = &_tmp_b;

        // The interleaved output matrix will have the following shape: [ a_height * 4, ceil(a_width / 4.0f) ]
        _tmp_a = TensorInfo(compute_interleaved_shape(*a_to_use), 1, a_to_use->data_type(), a_to_use->quantization_info());
        // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
        _tmp_b = TensorInfo(compute_transpose1xW_shape(*b), 1, b->data_type(), b->quantization_info());

        // Configure interleave kernel
        _mtx_a_reshape_kernel = std::make_unique<kernels::CpuGemmInterleave4x4Kernel>();
        _mtx_a_reshape_kernel->configure(a_to_use, &_tmp_a);

        // Configure transpose kernel
        _mtx_b_reshape_kernel = std::make_unique<kernels::CpuGemmTranspose1xWKernel>();
        _mtx_b_reshape_kernel->configure(b, &_tmp_b);
    }

    if(!_fused_assembly_path)
    {
        // Build reduction info
        const GEMMLowpReductionKernelInfo reduction_info(a_to_use->dimension(0), false, 0, false);

        // Initialize matrix B reduction kernel only if _a_offset is not equal to 0
        if(_a_offset != 0)
        {
            _vector_sum_col = TensorInfo(compute_reductionA_shape(*b), 1, DataType::S32);

            // Configure Matrix B reduction kernel
            _mtx_b_reduction_kernel = std::make_unique<kernels::CpuGemmLowpMatrixBReductionKernel>();
            _mtx_b_reduction_kernel->configure(b, &_vector_sum_col, reduction_info);
        }

        // Initialize Matrix A reduction kernel only if _b_offset is not equal to 0
        if(_b_offset != 0)
        {
            _vector_sum_row = TensorInfo(compute_reductionB_shape(*a_to_use), 1, DataType::S32);

            // Configure matrix A reduction kernel
            _mtx_a_reduction_kernel = std::make_unique<kernels::CpuGemmLowpMatrixAReductionKernel>();
            _mtx_a_reduction_kernel->configure(a_to_use, &_vector_sum_row, reduction_info);
        }

        if(_fuse_output_stage)
        {
            // Configure matrix multiply kernel
            if(!_assembly_path)
            {
                _mm_kernel = std::make_unique<kernels::CpuGemmLowpMatrixMultiplyKernel>();
                _mm_kernel->configure(matrix_a, matrix_b, &_mm_result_s32);
            }

            _offset_contribution_output_stage_kernel = std::make_unique<kernels::CpuGemmLowpOffsetContributionOutputStageKernel>();
            _offset_contribution_output_stage_kernel->configure(&_mm_result_s32,
                                                                _a_offset == 0 ? nullptr : &_vector_sum_col,
                                                                _b_offset == 0 ? nullptr : &_vector_sum_row, c,
                                                                _flip_signedness ? &_signed_output : dst,
                                                                a->dimension(0),
                                                                _a_offset, _b_offset, info.gemmlowp_output_stage());

            if(_flip_signedness)
            {
                _convert_from_signed_asymm = std::make_unique<kernels::CpuConvertQuantizedSignednessKernel>();
                _convert_from_signed_asymm->configure(&_signed_output, dst);
            }
        }
        else
        {
            // Configure matrix multiply kernel
            if(!_assembly_path)
            {
                _mm_kernel = std::make_unique<kernels::CpuGemmLowpMatrixMultiplyKernel>();
                _mm_kernel->configure(matrix_a, matrix_b, dst);
            }
            // Configure offset contribution kernel
            _offset_contribution_kernel = std::make_unique<kernels::CpuGemmLowpOffsetContributionKernel>();
            _offset_contribution_kernel->configure(dst, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, a_to_use->dimension(0),
                                                   _a_offset, _b_offset);
        }
    }
    // Configure activation
    const ActivationLayerInfo &activation = gemm_info.activation_info();
    _run_activation                       = activation.enabled() && (!_assembly_path || !cpu::CpuGemmAssemblyDispatch::is_activation_supported(activation));
    if(_run_activation)
    {
        _activation_func = std::make_unique<CpuActivation>();
        _activation_func->configure(dst, nullptr, activation);
    }

    if(_assembly_path)
    {
        auto asm_mem_req           = _asm_glue->workspace();
        _aux_mem[AsmGemmWorkspace] = asm_mem_req[AsmGemmWorkspace];
        _aux_mem[Pretranspose]     = asm_mem_req[Pretranspose];
    }

    // Request memory for LHS and RHS reshape matrix
    _aux_mem[VectorSumCol] = MemoryInfo(offset_int_vec(VectorSumCol), !_fused_assembly_path && _a_offset != 0
                                        && _reshape_b_only_on_first_run ?
                                        MemoryLifetime::Persistent :
                                        MemoryLifetime::Temporary,
                                        _vector_sum_col.total_size());
    _aux_mem[VectorSumRow] = MemoryInfo(offset_int_vec(VectorSumRow), MemoryLifetime::Temporary, _vector_sum_row.total_size());
    _aux_mem[TmpA]         = MemoryInfo(offset_int_vec(TmpA), MemoryLifetime::Temporary, _tmp_a.total_size());
    _aux_mem[TmpB]         = MemoryInfo(offset_int_vec(TmpB), _reshape_b_only_on_first_run ? MemoryLifetime::Persistent : MemoryLifetime::Temporary, _tmp_b.total_size());
    _aux_mem[MMResultS32]  = MemoryInfo(offset_int_vec(MMResultS32), MemoryLifetime::Temporary, _mm_result_s32.total_size());
    _aux_mem[SignedA]      = MemoryInfo(offset_int_vec(SignedA), MemoryLifetime::Temporary, _signed_a.total_size());
    _aux_mem[SignedOutput] = MemoryInfo(offset_int_vec(SignedOutput), MemoryLifetime::Temporary, _signed_output.total_size());
}

Status CpuGemmLowpMatrixMultiplyCore::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(b, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8, DataType::QSYMM8_PER_CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(c != nullptr && gemm_info.gemmlowp_output_stage().type == GEMMLowpOutputStageType::NONE, "Bias addition not supported in NEGEMMLowpMatrixMultiplyCore for output S32");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((a)->dimension(0) != (b)->dimension(1),
                                    "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    GEMMInfo           info          = gemm_info;
    const ITensorInfo *matrix_a_info = a;
    const ITensorInfo *matrix_b_info = b;

    const ITensorInfo *a_to_use = a;

    TensorInfo tmp_a_info{};
    TensorInfo tmp_b_info{};
    TensorInfo mm_result_s32_info{};

    int32_t a_offset = a->quantization_info().uniform().offset;
    int32_t b_offset = b->quantization_info().uniform().offset;

    bool fuse_output_stage = info.gemmlowp_output_stage().type != GEMMLowpOutputStageType::NONE;
    if(fuse_output_stage)
    {
        auto_init_if_empty(mm_result_s32_info, a->clone()->set_tensor_shape(output->tensor_shape()).set_data_type(DataType::S32));
    }

    // Convert QASYMM8->QASYMM8_SIGNED
    TensorInfo signed_a{};
    TensorInfo signed_output{};
    bool       flip_signedness = is_data_type_quantized_per_channel(b->data_type()) && (a->data_type() == DataType::QASYMM8) && info.reshape_b_only_on_first_run();
    if(flip_signedness)
    {
        const int32_t                 offset_correction = 128;
        const DataType                dt                = DataType::QASYMM8_SIGNED;
        const UniformQuantizationInfo iqinfo            = a_to_use->quantization_info().uniform();

        signed_a = a_to_use->clone()->set_data_type(dt).set_quantization_info(QuantizationInfo(iqinfo.scale, iqinfo.offset + offset_correction));
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuConvertQuantizedSignednessKernel::validate(a_to_use, &signed_a));
        a_to_use = &signed_a;
        a_offset = signed_a.quantization_info().uniform().offset;

        const UniformQuantizationInfo oqinfo = output->quantization_info().uniform();
        signed_output                        = output->clone()->set_data_type(dt).set_quantization_info(QuantizationInfo(oqinfo.scale, oqinfo.offset - offset_correction));

        // Output stage correction
        GEMMLowpOutputStageInfo output_stage_corr = info.gemmlowp_output_stage();
        output_stage_corr.gemmlowp_offset         = signed_output.quantization_info().uniform().offset;
        output_stage_corr.gemmlowp_min_bound -= offset_correction;
        output_stage_corr.gemmlowp_max_bound -= offset_correction;
        info.set_gemmlowp_output_stage(output_stage_corr);

        // Update matrix a
        matrix_a_info = &signed_a;
    }

    // Initialize assembly kernel meta-data
    const AsmGemmInfo asm_info = init_assembly_metadata(info);

    // Check if we need to run the optimized assembly kernel
    bool run_optimised             = false;
    bool run_optimised_requantized = false;
    if(is_data_type_quantized_asymmetric(a_to_use->data_type()) && info.gemmlowp_output_stage().type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
    {
        run_optimised             = bool(CpuGemmAssemblyDispatch::validate(a_to_use, b, c, output, asm_info));
        run_optimised_requantized = run_optimised;
    }
    else
    {
        run_optimised = bool(CpuGemmAssemblyDispatch::validate(a_to_use, b, nullptr, fuse_output_stage ? &mm_result_s32_info : output, asm_info));
    }

    if(run_optimised)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(b->dimension(0) != output->dimension(0));
        if(info.depth_output_gemm3d() != 0)
        {
            if(info.reinterpret_input_as_3d())
            {
                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1));
                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(2) != output->dimension(2));
            }
            else
            {
                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1) * output->dimension(2));
            }
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1));
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.reinterpret_input_as_3d(), "NEGEMM cannot reinterpret the input tensor as 3D");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.depth_output_gemm3d() != 0, "NEGEMM cannot reinterpret the output tensor as 3D");

        const bool run_vector_matrix_multiplication = a->dimension(1) < 2;
        if(!run_vector_matrix_multiplication)
        {
            matrix_a_info = &tmp_a_info;
            matrix_b_info = &tmp_b_info;

            // The interleaved output matrix will have the following shape: [ a_height * 4, ceil(a_width / 4.0f) ]
            TensorShape shape_tmp_a = a->tensor_shape();
            shape_tmp_a.set(0, a->dimension(0) * 4);
            shape_tmp_a.set(1, std::ceil(a->dimension(1) / 4.f));

            // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
            TensorShape shape_tmp_b = b->tensor_shape();
            shape_tmp_b.set(0, b->dimension(1) * 16);
            shape_tmp_b.set(1, std::ceil(b->dimension(0) / 16.f));

            // Validate interleave kernel
            auto_init_if_empty(tmp_a_info, a_to_use->clone()->set_tensor_shape(shape_tmp_a));
            auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(shape_tmp_b));

            ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuGemmInterleave4x4Kernel::validate(a_to_use, &tmp_a_info));
            ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuGemmTranspose1xWKernel::validate(b, &tmp_b_info));
        }
    }

    if(!run_optimised_requantized)
    {
        TensorInfo info_vector_sum_col{};
        TensorInfo info_vector_sum_row{};

        const GEMMLowpReductionKernelInfo reduction_info(a_to_use->dimension(0), false, 0, false);

        // Validate matrix B reduction kernel only if _a_offset is not equal to 0
        if(a_offset != 0)
        {
            info_vector_sum_col = TensorInfo(compute_reductionA_shape(*b), 1, DataType::S32);

            // Configure Matrix B reduction kernel
            ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuGemmLowpMatrixBReductionKernel::validate(b, &info_vector_sum_col, reduction_info));
        }

        // Validate Matrix A reduction kernel only if _b_offset is not equal to 0
        if(b_offset != 0)
        {
            info_vector_sum_row = TensorInfo(compute_reductionB_shape(*a), 1, DataType::S32);

            // Configure matrix A reduction kernel
            ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuGemmLowpMatrixAReductionKernel::validate(a_to_use, &info_vector_sum_row, reduction_info));
        }

        if(fuse_output_stage)
        {
            if(!run_optimised)
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.reinterpret_input_as_3d(), "CpuGemmLowpMatrixMultiplyKernel cannot reinterpret the input tensor as 3D");
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.depth_output_gemm3d() != 0, "CpuGemmLowpMatrixMultiplyKernel cannot reinterpret the output tensor as 3D");

                ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuGemmLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info));
            }

            // Validate offset contribution kernel
            ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuGemmLowpOffsetContributionOutputStageKernel::validate(&mm_result_s32_info,
                                                                                                          a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                                                          b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                                                          c,
                                                                                                          flip_signedness ? &signed_output : output,
                                                                                                          a_offset, b_offset,
                                                                                                          info.gemmlowp_output_stage()));
        }
        else
        {
            if(!run_optimised)
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.reinterpret_input_as_3d(), "CpuGemmLowpMatrixMultiplyKernel cannot reinterpret the input tensor as 3D");
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.depth_output_gemm3d() != 0, "CpuGemmLowpMatrixMultiplyKernel cannot reinterpret the output tensor as 3D");

                ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuGemmLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, output));
            }
            // Validate offset contribution kernel
            ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuGemmLowpOffsetContributionKernel::validate(output,
                                                                                               a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                                               b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                                               a_offset, b_offset));
        }
    }

    // Validate activation
    const ActivationLayerInfo &activation = gemm_info.activation_info();
    if(activation.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CpuActivation::validate(output, nullptr, activation));
    }

    return Status{};
}

void CpuGemmLowpMatrixMultiplyCore::run(ITensorPack &tensors)
{
    prepare(tensors);

    auto a        = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto b        = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto c        = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto dst      = tensors.get_tensor(TensorType::ACL_DST);
    auto a_to_use = a;
    auto matrix_a = a;
    auto matrix_b = b;

    CpuAuxTensorHandler vector_sum_col(offset_int_vec(VectorSumCol), _vector_sum_col, tensors, false);
    CpuAuxTensorHandler vector_sum_row(offset_int_vec(VectorSumRow), _vector_sum_row, tensors, false);
    CpuAuxTensorHandler tmp_a(offset_int_vec(TmpA), _tmp_a, tensors, false);
    CpuAuxTensorHandler tmp_b(offset_int_vec(TmpB), _tmp_b, tensors, true);
    CpuAuxTensorHandler mm_result_s32(offset_int_vec(MMResultS32), _mm_result_s32, tensors, false);
    CpuAuxTensorHandler signed_a(offset_int_vec(SignedA), _signed_a, tensors, false);
    CpuAuxTensorHandler signed_output(offset_int_vec(SignedOutput), _signed_output, tensors, false);

    // Convert QASYMM8->QASYMM8_SIGNED
    if(_flip_signedness)
    {
        ITensorPack pack =
        {
            { TensorType::ACL_SRC, a },
            { TensorType::ACL_DST, signed_a.get() }
        };
        NEScheduler::get().schedule_op(_convert_to_signed_asymm.get(), Window::DimY, _convert_to_signed_asymm->window(), pack);
        a_to_use = signed_a.get();
    }

    // Run GEMM
    if(_asm_glue->is_configured())
    {
        ITensorPack asm_glue_tensors = tensors;
        auto        output_to_use    = (_fuse_output_stage ? mm_result_s32.get() : dst);
        if(is_data_type_quantized_asymmetric(a_to_use->info()->data_type()) && _gemm_info.gemmlowp_output_stage().type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
        {
            asm_glue_tensors.add_const_tensor(TensorType::ACL_SRC_0, a_to_use);
            asm_glue_tensors.add_const_tensor(TensorType::ACL_SRC_1, b);
            asm_glue_tensors.add_const_tensor(TensorType::ACL_SRC_2, c);
            asm_glue_tensors.add_tensor(TensorType::ACL_DST, dst);
        }
        else
        {
            asm_glue_tensors.add_const_tensor(TensorType::ACL_SRC_0, a_to_use);
            asm_glue_tensors.add_const_tensor(TensorType::ACL_SRC_1, b);
            asm_glue_tensors.add_tensor(TensorType::ACL_DST, output_to_use);
        }
        _asm_glue->run(asm_glue_tensors);
    }
    else
    {
        if(!_run_vector_matrix_multiplication)
        {
            matrix_a = tmp_a.get();
            matrix_b = tmp_b.get();
            // Run interleave kernel
            ITensorPack pack_a =
            {
                { TensorType::ACL_SRC, a_to_use },
                { TensorType::ACL_DST, tmp_a.get() }
            };
            NEScheduler::get().schedule_op(_mtx_a_reshape_kernel.get(), Window::DimY, _mtx_a_reshape_kernel->window(), pack_a);

            if(!_reshape_b_only_on_first_run)
            {
                ITensorPack pack_b =
                {
                    { TensorType::ACL_SRC, b },
                    { TensorType::ACL_DST, tmp_b.get() }
                };
                // Run transpose kernel
                NEScheduler::get().schedule_op(_mtx_b_reshape_kernel.get(), Window::DimY, _mtx_b_reshape_kernel->window(), pack_b);
            }
        }
        ITensorPack pack_mm =
        {
            { TensorType::ACL_SRC_0, matrix_a },
            { TensorType::ACL_SRC_1, matrix_b }
        };
        if(_fuse_output_stage)
        {
            pack_mm.add_tensor(TensorType::ACL_DST, mm_result_s32.get());
        }
        else
        {
            pack_mm.add_tensor(TensorType::ACL_DST, dst);
        }
        NEScheduler::get().schedule_op(_mm_kernel.get(), Window::DimY, _mm_kernel->window(), pack_mm);
    }

    if(!_fused_assembly_path)
    {
        // Run matrix A reduction kernel only if _b_offset is not equal to 0
        if(_b_offset != 0)
        {
            ITensorPack pack =
            {
                { TensorType::ACL_SRC, a_to_use },
                { TensorType::ACL_DST, vector_sum_row.get() }
            };
            NEScheduler::get().schedule_op(_mtx_a_reduction_kernel.get(), Window::DimX, _mtx_a_reduction_kernel->window(), pack);
        }

        // Run matrix B reduction kernel only if _a_offset is not equal to 0
        if(_a_offset != 0 && !_reshape_b_only_on_first_run)
        {
            ITensorPack pack =
            {
                { TensorType::ACL_SRC, b },
                { TensorType::ACL_DST, vector_sum_col.get() }
            };
            NEScheduler::get().schedule_op(_mtx_b_reduction_kernel.get(), Window::DimX, _mtx_b_reduction_kernel->window(), pack);
        }

        if(_fuse_output_stage)
        {
            ITensorPack pack;
            pack.add_tensor(TensorType::ACL_SRC_0, mm_result_s32.get());
            pack.add_tensor(TensorType::ACL_SRC_1, _a_offset == 0 ? nullptr : vector_sum_col.get());
            pack.add_tensor(TensorType::ACL_SRC_2, _b_offset == 0 ? nullptr : vector_sum_row.get());
            pack.add_tensor(TensorType::ACL_SRC_3, c);
            pack.add_tensor(TensorType::ACL_DST, _flip_signedness ? signed_output.get() : dst);

            // Run offset contribution kernel
            NEScheduler::get().schedule_op(_offset_contribution_output_stage_kernel.get(), Window::DimY, _offset_contribution_output_stage_kernel->window(), pack);
        }
        else
        {
            ITensorPack pack;
            pack.add_tensor(TensorType::ACL_SRC_0, _a_offset == 0 ? nullptr : vector_sum_col.get());
            pack.add_tensor(TensorType::ACL_SRC_1, _b_offset == 0 ? nullptr : vector_sum_row.get());
            pack.add_tensor(TensorType::ACL_DST, dst);

            // Run offset contribution kernel
            NEScheduler::get().schedule_op(_offset_contribution_kernel.get(), Window::DimY, _offset_contribution_kernel->window(), pack);
        }
    }

    // Convert QASYMM8_SIGNED->QASYMM8
    if(!_fused_assembly_path && _fuse_output_stage && _flip_signedness)
    {
        ITensorPack pack =
        {
            { TensorType::ACL_SRC, signed_output.get() },
            { TensorType::ACL_DST, dst }
        };
        NEScheduler::get().schedule_op(_convert_from_signed_asymm.get(), Window::DimY, _convert_from_signed_asymm->window(), pack);
    }

    // Run fused activation unless already run in the fused assembly
    if(_run_activation)
    {
        ITensorPack pack =
        {
            { TensorType::ACL_SRC, dst },
            { TensorType::ACL_DST, dst }
        };
        _activation_func->run(pack);
    }
}

void CpuGemmLowpMatrixMultiplyCore::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        auto original_b = tensors.get_const_tensor(TensorType::ACL_SRC_1);
        // Run assembly reshape
        if(_asm_glue->is_configured())
        {
            _asm_glue->prepare(tensors);
        }
        // Run non-assembly reshape
        else if(_reshape_b_only_on_first_run && !_run_vector_matrix_multiplication && !_asm_glue->is_configured())
        {
            // Run reshape kernel and mark original weights tensor as unused
            ITensor            *tmp_b_p = utils::cast::polymorphic_downcast<ITensor *>(tensors.get_tensor(offset_int_vec(TmpB)));
            CpuAuxTensorHandler tmp_b(_tmp_b, *tmp_b_p);
            ITensorPack         pack =
            {
                { TensorType::ACL_SRC, original_b },
                { TensorType::ACL_DST, tmp_b.get() }
            };
            NEScheduler::get().schedule_op(_mtx_b_reshape_kernel.get(), Window::DimY, _mtx_b_reshape_kernel->window(), pack);
        }

        // Run matrix B reduction kernel only if _a_offset is not equal to 0
        if(!_fused_assembly_path && _a_offset != 0 && _reshape_b_only_on_first_run)
        {
            ITensor            *vector_sum_col_p = utils::cast::polymorphic_downcast<ITensor *>(tensors.get_tensor(offset_int_vec(VectorSumCol)));
            CpuAuxTensorHandler vector_sum_col(_vector_sum_col, *vector_sum_col_p);
            ITensorPack         pack =
            {
                { TensorType::ACL_SRC, original_b },
                { TensorType::ACL_DST, vector_sum_col.get() }
            };
            NEScheduler::get().schedule_op(_mtx_b_reduction_kernel.get(), Window::DimX, _mtx_b_reduction_kernel->window(), pack);
        }
        _is_prepared = true;
    }
}
experimental::MemoryRequirements CpuGemmLowpMatrixMultiplyCore::workspace() const
{
    return _aux_mem;
}
} // namespace cpu
} // namespace arm_compute
