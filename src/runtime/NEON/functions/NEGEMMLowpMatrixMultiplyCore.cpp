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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "src/core/NEON/kernels/NEConvertQuantizedSignednessKernel.h"
#include "src/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionOutputStageKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpReductionKernel.h"
#include "src/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "src/runtime/cpu/operators/internal/CpuGemmAssemblyDispatch.h"

namespace arm_compute
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

    return asm_info;
}
} // namespace

struct NEGEMMLowpMatrixMultiplyCore::Impl
{
    MemoryGroup                                                    memory_group{};
    IWeightsManager                                               *weights_manager{ nullptr };
    std::unique_ptr<cpu::CpuGemmAssemblyDispatch>                  asm_glue{ nullptr };
    std::unique_ptr<NEGEMMLowpMatrixMultiplyKernel>                mm_kernel{ nullptr };
    std::unique_ptr<NEGEMMInterleave4x4Kernel>                     mtx_a_reshape_kernel{ nullptr };
    std::unique_ptr<NEGEMMTranspose1xWKernel>                      mtx_b_reshape_kernel{ nullptr };
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel>              mtx_a_reduction_kernel{ nullptr };
    std::unique_ptr<NEGEMMLowpMatrixBReductionKernel>              mtx_b_reduction_kernel{ nullptr };
    std::unique_ptr<NEGEMMLowpOffsetContributionKernel>            offset_contribution_kernel{ nullptr };
    std::unique_ptr<NEGEMMLowpOffsetContributionOutputStageKernel> offset_contribution_output_stage_kernel{ nullptr };
    std::unique_ptr<NEActivationLayer>                             activation_func{ nullptr };
    std::unique_ptr<NEConvertQuantizedSignednessKernel>            convert_to_signed_asymm{ nullptr };
    std::unique_ptr<NEConvertQuantizedSignednessKernel>            convert_from_signed_asymm{ nullptr };

    Tensor         vector_sum_col{};
    Tensor         vector_sum_row{};
    Tensor         tmp_a{};
    Tensor         tmp_b{};
    Tensor         mm_result_s32{};
    Tensor         signed_a{};
    Tensor         signed_output{};
    const ITensor *original_b{ nullptr };
    int32_t        a_offset{ 0 };
    int32_t        b_offset{ 0 };

    bool run_vector_matrix_multiplication{ false };
    bool assembly_path{ false };
    bool fused_assembly_path{ false };
    bool reshape_b_only_on_first_run{ false };
    bool is_prepared{ false };
    bool fuse_output_stage{ false };
    bool run_activation{ false };
    bool flip_signedness{ false };

    experimental::MemoryRequirements aux_mem_req{};
    ITensorPack                      asm_glue_run_pack{};
    ITensorPack                      asm_glue_prep_pack{};
    WorkspaceData<Tensor>            asm_glue_workspace{};
};

using namespace arm_compute::experimental;
using namespace arm_compute::misc::shape_calculator;

NEGEMMLowpMatrixMultiplyCore::~NEGEMMLowpMatrixMultiplyCore() = default;

NEGEMMLowpMatrixMultiplyCore::NEGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager, IWeightsManager *weights_manager)
    : _impl(std::make_unique<struct NEGEMMLowpMatrixMultiplyCore::Impl>())
{
    _impl->memory_group    = MemoryGroup(memory_manager);
    _impl->weights_manager = weights_manager;
}

void NEGEMMLowpMatrixMultiplyCore::configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMLowpMatrixMultiplyCore::validate(a->info(), b->info(), c != nullptr ? c->info() : nullptr, output->info(), gemm_info));

    const ITensor *matrix_a = a;
    const ITensor *matrix_b = b;
    GEMMInfo       info     = gemm_info;

    // Set internal variables
    _impl->a_offset                         = a->info()->quantization_info().uniform().offset;
    _impl->b_offset                         = b->info()->quantization_info().uniform().offset;
    _impl->run_vector_matrix_multiplication = a->info()->dimension(1) < 2;
    _impl->reshape_b_only_on_first_run      = info.reshape_b_only_on_first_run();
    _impl->is_prepared                      = false;
    _impl->fused_assembly_path              = false;
    _impl->flip_signedness                  = is_data_type_quantized_per_channel(b->info()->data_type()) && (a->info()->data_type() == DataType::QASYMM8) && _impl->reshape_b_only_on_first_run;
    _impl->original_b                       = b;

    _impl->asm_glue = std::make_unique<cpu::CpuGemmAssemblyDispatch>();

    const ITensor *a_to_use = a;

    // Convert to QASYMM8 -> QASYMM8_SIGNED and back
    if(_impl->flip_signedness)
    {
        const int32_t                 offset_correction = 128;
        const DataType                dt                = DataType::QASYMM8_SIGNED;
        const UniformQuantizationInfo iqinfo            = a_to_use->info()->quantization_info().uniform();

        _impl->signed_a.allocator()->init(a_to_use->info()->clone()->set_data_type(dt).set_quantization_info(QuantizationInfo(iqinfo.scale, iqinfo.offset + offset_correction)));
        _impl->memory_group.manage(&_impl->signed_a);
        _impl->convert_to_signed_asymm = std::make_unique<NEConvertQuantizedSignednessKernel>();
        _impl->convert_to_signed_asymm->configure(a_to_use, &_impl->signed_a);
        a_to_use        = &_impl->signed_a;
        _impl->a_offset = _impl->signed_a.info()->quantization_info().uniform().offset;

        const UniformQuantizationInfo oqinfo = output->info()->quantization_info().uniform();
        _impl->memory_group.manage(&_impl->signed_output);
        _impl->signed_output.allocator()->init(output->info()->clone()->set_data_type(dt).set_quantization_info(QuantizationInfo(oqinfo.scale, oqinfo.offset - offset_correction)));

        // Output stage correction
        GEMMLowpOutputStageInfo output_stage_corr = info.gemmlowp_output_stage();
        output_stage_corr.gemmlowp_offset         = _impl->signed_output.info()->quantization_info().uniform().offset;
        output_stage_corr.gemmlowp_min_bound -= offset_correction;
        output_stage_corr.gemmlowp_max_bound -= offset_correction;
        info.set_gemmlowp_output_stage(output_stage_corr);

        // Update matrix a
        matrix_a = &_impl->signed_a;
    }

    // If GEMMLowpOutputStage != NONE, fuse the offset contribution with the output stage
    if(info.gemmlowp_output_stage().type != GEMMLowpOutputStageType::NONE)
    {
        _impl->fuse_output_stage = true;
        _impl->memory_group.manage(&_impl->mm_result_s32);
        TensorInfo info_mm_result_s32(output->info()->tensor_shape(), 1, DataType::S32);
        _impl->mm_result_s32.allocator()->init(info_mm_result_s32);
    }

    // Initialize assembly kernel meta-data
    const cpu::AsmGemmInfo asm_info = init_assembly_metadata(gemm_info);
#ifdef __aarch64__
    switch(a->info()->data_type())
    {
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::U8:
        case DataType::S8:
        {
            if(is_data_type_quantized_asymmetric(a_to_use->info()->data_type()) && info.gemmlowp_output_stage().type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
            {
                auto c_info_to_use = c == nullptr ? nullptr : c->info();
                _impl->asm_glue->configure(a_to_use->info(), b->info(), c_info_to_use, output->info(), asm_info);
                _impl->fused_assembly_path = _impl->asm_glue->is_configured();
                _impl->asm_glue_run_pack.add_const_tensor(TensorType::ACL_SRC_2, c);
                _impl->asm_glue_run_pack.add_tensor(TensorType::ACL_DST, output);
            }
            else
            {
                auto output_to_use = (_impl->fuse_output_stage ? &_impl->mm_result_s32 : output);
                _impl->asm_glue->configure(a_to_use->info(), b->info(), nullptr, output_to_use->info(), asm_info);
                _impl->asm_glue_run_pack.add_tensor(TensorType::ACL_DST, output_to_use);
            }
            _impl->assembly_path = _impl->asm_glue->is_configured();

            if(_impl->assembly_path)
            {
                _impl->asm_glue_run_pack.add_const_tensor(TensorType::ACL_SRC_0, a_to_use);

                _impl->aux_mem_req        = _impl->asm_glue->workspace();
                _impl->asm_glue_prep_pack = { { TensorType::ACL_SRC_1, b }, { TensorType::ACL_SRC_2, c } };

                _impl->asm_glue_workspace = manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->asm_glue_run_pack, _impl->asm_glue_prep_pack);
            }
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Datatype not supported");
            break;
        }
    }
#endif /* __aarch64__ */
    if(!(_impl->assembly_path || _impl->run_vector_matrix_multiplication))
    {
        matrix_a = &_impl->tmp_a;
        matrix_b = &_impl->tmp_b;

        // The interleaved output matrix will have the following shape: [ a_height * 4, ceil(a_width / 4.0f) ]
        TensorInfo a_info(compute_interleaved_shape(*a_to_use->info()), 1, a_to_use->info()->data_type(), a_to_use->info()->quantization_info());
        // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
        TensorInfo b_info(compute_transpose1xW_shape(*b->info()), 1, b->info()->data_type(), b->info()->quantization_info());
        _impl->tmp_a.allocator()->init(a_info);
        _impl->tmp_b.allocator()->init(b_info);
        _impl->memory_group.manage(&_impl->tmp_a);
        if(!_impl->reshape_b_only_on_first_run)
        {
            _impl->memory_group.manage(&_impl->tmp_b);
        }

        // Configure interleave kernel
        _impl->mtx_a_reshape_kernel = std::make_unique<NEGEMMInterleave4x4Kernel>();
        _impl->mtx_a_reshape_kernel->configure(a_to_use, &_impl->tmp_a);

        // Configure transpose kernel
        _impl->mtx_b_reshape_kernel = std::make_unique<NEGEMMTranspose1xWKernel>();
        _impl->mtx_b_reshape_kernel->configure(b, &_impl->tmp_b);
    }

    if(!_impl->fused_assembly_path)
    {
        // Build reduction info
        const GEMMLowpReductionKernelInfo reduction_info(a_to_use->info()->dimension(0), false, 0, false);

        // Initialize matrix B reduction kernel only if _impl->a_offset is not equal to 0
        if(_impl->a_offset != 0)
        {
            TensorInfo info_vector_sum_col(compute_reductionA_shape(*b->info()), 1, DataType::S32);

            _impl->vector_sum_col.allocator()->init(info_vector_sum_col);
            if(!_impl->reshape_b_only_on_first_run)
            {
                _impl->memory_group.manage(&_impl->vector_sum_col);
            }

            // Configure Matrix B reduction kernel
            _impl->mtx_b_reduction_kernel = std::make_unique<NEGEMMLowpMatrixBReductionKernel>();
            _impl->mtx_b_reduction_kernel->configure(b, &_impl->vector_sum_col, reduction_info);
        }

        // Initialize Matrix A reduction kernel only if _impl->b_offset is not equal to 0
        if(_impl->b_offset != 0)
        {
            TensorInfo info_vector_sum_row(compute_reductionB_shape(*a_to_use->info()), 1, DataType::S32);

            _impl->vector_sum_row.allocator()->init(info_vector_sum_row);
            _impl->memory_group.manage(&_impl->vector_sum_row);

            // Configure matrix A reduction kernel
            _impl->mtx_a_reduction_kernel = std::make_unique<NEGEMMLowpMatrixAReductionKernel>();
            _impl->mtx_a_reduction_kernel->configure(a_to_use, &_impl->vector_sum_row, reduction_info);
        }

        if(_impl->fuse_output_stage)
        {
            // Configure matrix multiply kernel
            if(!_impl->assembly_path)
            {
                _impl->mm_kernel = std::make_unique<NEGEMMLowpMatrixMultiplyKernel>();
                _impl->mm_kernel->configure(matrix_a, matrix_b, &_impl->mm_result_s32);
            }

            _impl->offset_contribution_output_stage_kernel = std::make_unique<NEGEMMLowpOffsetContributionOutputStageKernel>();
            _impl->offset_contribution_output_stage_kernel->configure(&_impl->mm_result_s32,
                                                                      _impl->a_offset == 0 ? nullptr : &_impl->vector_sum_col,
                                                                      _impl->b_offset == 0 ? nullptr : &_impl->vector_sum_row, c,
                                                                      _impl->flip_signedness ? &_impl->signed_output : output,
                                                                      a->info()->dimension(0),
                                                                      _impl->a_offset, _impl->b_offset, info.gemmlowp_output_stage());

            if(_impl->flip_signedness)
            {
                _impl->convert_from_signed_asymm = std::make_unique<NEConvertQuantizedSignednessKernel>();
                _impl->convert_from_signed_asymm->configure(&_impl->signed_output, output);
            }
        }
        else
        {
            // Configure matrix multiply kernel
            if(!_impl->assembly_path)
            {
                _impl->mm_kernel = std::make_unique<NEGEMMLowpMatrixMultiplyKernel>();
                _impl->mm_kernel->configure(matrix_a, matrix_b, output);
            }
            // Configure offset contribution kernel
            _impl->offset_contribution_kernel = std::make_unique<NEGEMMLowpOffsetContributionKernel>();
            _impl->offset_contribution_kernel->configure(output, _impl->a_offset == 0 ? nullptr : &_impl->vector_sum_col, _impl->b_offset == 0 ? nullptr : &_impl->vector_sum_row, a_to_use->info()->dimension(0),
                                                         _impl->a_offset, _impl->b_offset);
        }
    }
    // Configure activation
    const ActivationLayerInfo &activation = gemm_info.activation_info();
    _impl->run_activation                 = activation.enabled() && (!_impl->assembly_path || !cpu::CpuGemmAssemblyDispatch::is_activation_supported(activation));
    if(_impl->run_activation)
    {
        _impl->activation_func = std::make_unique<NEActivationLayer>();
        _impl->activation_func->configure(output, nullptr, activation);
    }

    // Allocate tensors
    if(!_impl->assembly_path && !_impl->run_vector_matrix_multiplication)
    {
        _impl->tmp_a.allocator()->allocate();
        if(!_impl->reshape_b_only_on_first_run)
        {
            _impl->tmp_b.allocator()->allocate();
        }
    }

    if(!_impl->fused_assembly_path)
    {
        if(_impl->a_offset != 0 && !_impl->reshape_b_only_on_first_run)
        {
            _impl->vector_sum_col.allocator()->allocate();
        }

        if(_impl->b_offset != 0)
        {
            _impl->vector_sum_row.allocator()->allocate();
        }
    }

    if(_impl->fuse_output_stage)
    {
        _impl->mm_result_s32.allocator()->allocate();
    }

    if(_impl->flip_signedness)
    {
        _impl->signed_a.allocator()->allocate();
        _impl->signed_output.allocator()->allocate();
    }
}

Status NEGEMMLowpMatrixMultiplyCore::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info)
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
        ARM_COMPUTE_RETURN_ON_ERROR(NEConvertQuantizedSignednessKernel::validate(a_to_use, &signed_a));
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
    const cpu::AsmGemmInfo asm_info = init_assembly_metadata(info);

    // Check if we need to run the optimized assembly kernel
    bool run_optimised             = false;
    bool run_optimised_requantized = false;
    if(is_data_type_quantized_asymmetric(a_to_use->data_type()) && info.gemmlowp_output_stage().type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
    {
        run_optimised             = bool(cpu::CpuGemmAssemblyDispatch::validate(a_to_use, b, c, output, asm_info));
        run_optimised_requantized = run_optimised;
    }
    else
    {
        run_optimised = bool(cpu::CpuGemmAssemblyDispatch::validate(a_to_use, b, nullptr, fuse_output_stage ? &mm_result_s32_info : output, asm_info));
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

            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMInterleave4x4Kernel::validate(a_to_use, &tmp_a_info));
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMTranspose1xWKernel::validate(b, &tmp_b_info));
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
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixBReductionKernel::validate(b, &info_vector_sum_col, reduction_info));
        }

        // Validate Matrix A reduction kernel only if _b_offset is not equal to 0
        if(b_offset != 0)
        {
            info_vector_sum_row = TensorInfo(compute_reductionB_shape(*a), 1, DataType::S32);

            // Configure matrix A reduction kernel
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(a_to_use, &info_vector_sum_row, reduction_info));
        }

        if(fuse_output_stage)
        {
            if(!run_optimised)
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.reinterpret_input_as_3d(), "NEGEMMLowpMatrixMultiplyKernel cannot reinterpret the input tensor as 3D");
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.depth_output_gemm3d() != 0, "NEGEMMLowpMatrixMultiplyKernel cannot reinterpret the output tensor as 3D");

                ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info));
            }

            // Validate offset contribution kernel
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpOffsetContributionOutputStageKernel::validate(&mm_result_s32_info,
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
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.reinterpret_input_as_3d(), "NEGEMMLowpMatrixMultiplyKernel cannot reinterpret the input tensor as 3D");
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.depth_output_gemm3d() != 0, "NEGEMMLowpMatrixMultiplyKernel cannot reinterpret the output tensor as 3D");

                ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, output));
            }
            // Validate offset contribution kernel
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpOffsetContributionKernel::validate(output,
                                                                                     a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                                     b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                                     a_offset, b_offset));
        }
    }

    // Validate activation
    const ActivationLayerInfo &activation = gemm_info.activation_info();
    if(activation.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(output, nullptr, activation));
    }

    return Status{};
}

void NEGEMMLowpMatrixMultiplyCore::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);

    // Convert QASYMM8->QASYMM8_SIGNED
    if(_impl->flip_signedness)
    {
        NEScheduler::get().schedule(_impl->convert_to_signed_asymm.get(), Window::DimY);
    }

    // Run GEMM
    if(_impl->asm_glue->is_configured())
    {
        _impl->asm_glue->run(_impl->asm_glue_run_pack);
    }
    else
    {
        if(!_impl->run_vector_matrix_multiplication)
        {
            // Run interleave kernel
            NEScheduler::get().schedule(_impl->mtx_a_reshape_kernel.get(), Window::DimY);

            if(!_impl->reshape_b_only_on_first_run)
            {
                // Run transpose kernel
                NEScheduler::get().schedule(_impl->mtx_b_reshape_kernel.get(), Window::DimY);
            }
        }
        NEScheduler::get().schedule(_impl->mm_kernel.get(), Window::DimY);
    }

    if(!_impl->fused_assembly_path)
    {
        // Run matrix A reduction kernel only if _impl->b_offset is not equal to 0
        if(_impl->b_offset != 0)
        {
            NEScheduler::get().schedule(_impl->mtx_a_reduction_kernel.get(), Window::DimX);
        }

        // Run matrix B reduction kernel only if _impl->a_offset is not equal to 0
        if(_impl->a_offset != 0 && !_impl->reshape_b_only_on_first_run)
        {
            NEScheduler::get().schedule(_impl->mtx_b_reduction_kernel.get(), Window::DimX);
        }

        if(_impl->fuse_output_stage)
        {
            // Run offset contribution kernel
            NEScheduler::get().schedule(_impl->offset_contribution_output_stage_kernel.get(), Window::DimY);
        }
        else
        {
            // Run offset contribution kernel
            NEScheduler::get().schedule(_impl->offset_contribution_kernel.get(), Window::DimY);
        }
    }

    // Convert QASYMM8_SIGNED->QASYMM8
    if(!_impl->fused_assembly_path && _impl->fuse_output_stage && _impl->flip_signedness)
    {
        NEScheduler::get().schedule(_impl->convert_from_signed_asymm.get(), Window::DimY);
    }

    // Run fused activation unless already run in the fused assembly
    if(_impl->run_activation)
    {
        _impl->activation_func->run();
    }
}

void NEGEMMLowpMatrixMultiplyCore::prepare()
{
    if(!_impl->is_prepared)
    {
        // Run assembly reshape
        if(_impl->asm_glue->is_configured())
        {
            _impl->asm_glue->prepare(_impl->asm_glue_prep_pack);

            auto has_reshape = std::find_if(_impl->aux_mem_req.begin(),
                                            _impl->aux_mem_req.end(),
                                            [](const MemoryInfo & m) -> bool { return m.lifetime == MemoryLifetime::Persistent; });

            if(has_reshape != std::end(_impl->aux_mem_req))
            {
                _impl->original_b->mark_as_unused();
            }
            else
            {
                _impl->asm_glue_run_pack.add_const_tensor(ACL_SRC_1, _impl->original_b);
            }
        }
        // Run non-assembly reshape
        else if(_impl->reshape_b_only_on_first_run && !_impl->run_vector_matrix_multiplication && !_impl->asm_glue->is_configured())
        {
            // Run reshape kernel and mark original weights tensor as unused
            _impl->tmp_b.allocator()->allocate();
            NEScheduler::get().schedule(_impl->mtx_b_reshape_kernel.get(), Window::DimY);
        }

        // Run matrix B reduction kernel only if _impl->a_offset is not equal to 0
        if(!_impl->fused_assembly_path && _impl->a_offset != 0 && _impl->reshape_b_only_on_first_run)
        {
            _impl->vector_sum_col.allocator()->allocate();
            NEScheduler::get().schedule(_impl->mtx_b_reduction_kernel.get(), Window::DimX);
        }

        _impl->is_prepared = true;
    }
}
} // namespace arm_compute
