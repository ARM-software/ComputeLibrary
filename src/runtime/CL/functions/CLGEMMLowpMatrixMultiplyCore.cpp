/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/gemm/native/CLGEMMNativeKernelConfiguration.h"
#include "src/core/CL/gemm/reshaped_only_rhs/CLGEMMReshapedOnlyRHSKernelConfiguration.h"
#include "src/core/CL/kernels/CLDepthConvertLayerKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpMatrixMultiplyNativeKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpOffsetContributionKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpOffsetContributionOutputStageKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpReductionKernel.h"
#include "src/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/runtime/CL/gemm/CLGEMMKernelSelection.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::cl_gemm;

namespace
{
inline bool is_gemm_reshaped(unsigned int m, unsigned int n, unsigned int k, DataType data_type, bool reshape_b_only_on_first_run)
{
    std::unique_ptr<ICLGEMMKernelSelection> gemm_kernel = CLGEMMKernelSelectionFactory::create(CLScheduler::get().target());
    ARM_COMPUTE_ERROR_ON_NULLPTR(gemm_kernel.get());

    CLGEMMKernelSelectionParams params;
    params.m               = m;
    params.n               = n;
    params.k               = k;
    params.is_rhs_constant = reshape_b_only_on_first_run;
    params.data_type       = data_type;

    switch(gemm_kernel->select_kernel(params))
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

CLGEMMLowpMatrixMultiplyCore::CLGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _weights_to_qasymm8(support::cpp14::make_unique<CLDepthConvertLayerKernel>()),
      _mm_native_kernel(support::cpp14::make_unique<CLGEMMLowpMatrixMultiplyNativeKernel>()),
      _mm_reshaped_only_rhs_kernel(support::cpp14::make_unique<CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel>()),
      _mtx_b_reshape_kernel(support::cpp14::make_unique<CLGEMMReshapeRHSMatrixKernel>()),
      _mtx_a_reduction_kernel(support::cpp14::make_unique<CLGEMMLowpMatrixAReductionKernel>()),
      _mtx_b_reduction_kernel(support::cpp14::make_unique<CLGEMMLowpMatrixBReductionKernel>()),
      _offset_contribution_kernel(support::cpp14::make_unique<CLGEMMLowpOffsetContributionKernel>()),
      _offset_contribution_output_stage_kernel(support::cpp14::make_unique<CLGEMMLowpOffsetContributionOutputStageKernel>()),
      _qasymm8_weights(),
      _vector_sum_col(),
      _vector_sum_row(),
      _tmp_b(),
      _mm_result_s32(),
      _gemm_output_stage_multipliers(),
      _gemm_output_stage_shifts(),
      _matrix_a(nullptr),
      _original_b(nullptr),
      _output(nullptr),
      _a_offset(0),
      _b_offset(0),
      _is_gemm_reshaped(true),
      _reshape_b_only_on_first_run(false),
      _is_prepared(false),
      _run_output_stage(false),
      _convert_to_qasymm8(false),
      _run_offset_contribution(false)
{
}

CLGEMMLowpMatrixMultiplyCore::~CLGEMMLowpMatrixMultiplyCore() = default;

void CLGEMMLowpMatrixMultiplyCore::configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, const GEMMInfo &gemm_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), a, b, c, output, gemm_info);
}

void CLGEMMLowpMatrixMultiplyCore::configure(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLGEMMLowpMatrixMultiplyCore::validate(a->info(), b->info(), c != nullptr ? c->info() : nullptr, output->info(), gemm_info));

    _is_prepared                 = false;
    _original_b                  = b;
    _reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();
    _a_offset                    = a->info()->quantization_info().uniform().offset;
    _matrix_a                    = a;
    _output                      = output;

    _convert_to_qasymm8 = is_data_type_quantized_per_channel(b->info()->data_type()) && is_data_type_quantized_symmetric(b->info()->data_type())
                          && a->info()->data_type() == DataType::QASYMM8;
    _b_offset = _convert_to_qasymm8 ? -128 : b->info()->quantization_info().uniform().offset;

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
    const unsigned int m                       = reinterpret_input_as_3d ? (a->info()->dimension(1) * a->info()->dimension(2)) : a->info()->dimension(1);
    const unsigned int n                       = b->info()->dimension(0);
    const unsigned int k                       = a->info()->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->info()->dimension(3) : a->info()->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();

    // Check if we need to reshape the matrix A and matrix B
    _is_gemm_reshaped = is_gemm_reshaped(m, n, k, a->info()->data_type(), _reshape_b_only_on_first_run);

    if(_convert_to_qasymm8)
    {
        // Set data type for converted weights
        TensorInfo weights_info(*b->info());
        weights_info.set_data_type(DataType::QASYMM8);
        _qasymm8_weights.allocator()->init(weights_info);
        _weights_to_qasymm8->configure(compile_context, b, &_qasymm8_weights, ConvertPolicy::WRAP, 0);
    }

    const ICLTensor *matrix_b = _convert_to_qasymm8 ? &_qasymm8_weights : b;
    if(_is_gemm_reshaped)
    {
        matrix_b = &_tmp_b;

        if(!_reshape_b_only_on_first_run)
        {
            _memory_group.manage(&_tmp_b);
        }

        // Pick up the GEMM configuration
        // Datatype is DataType::QASYMM8 or DataType::QASYMM8_SIGNED doesn't matter, since it only affect the shape configuration
        std::tie(lhs_info, rhs_info) = CLGEMMReshapedOnlyRHSKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

        // Configure reshape RHS kernel
        _mtx_b_reshape_kernel->configure(compile_context, _convert_to_qasymm8 ? &_qasymm8_weights : b, &_tmp_b, rhs_info);
    }

    // Using default reduction info
    const GEMMLowpReductionKernelInfo reduction_info {};

    // Initialize matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0)
    {
        TensorInfo info_vector_sum_col(compute_reductionA_shape(*b->info()), 1, DataType::S32);
        _vector_sum_col.allocator()->init(info_vector_sum_col);
        if(!_reshape_b_only_on_first_run)
        {
            _memory_group.manage(&_vector_sum_col);
        }

        // Configure Matrix B reduction kernel
        _mtx_b_reduction_kernel->configure(compile_context, _convert_to_qasymm8 ? &_qasymm8_weights : b, &_vector_sum_col, reduction_info);
    }

    // Initialize Matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        TensorInfo info_vector_sum_row(compute_reductionB_shape(*a->info()), 1, DataType::S32);
        _vector_sum_row.allocator()->init(info_vector_sum_row);
        _memory_group.manage(&_vector_sum_row);

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

        _gemm_output_stage_multipliers.allocator()->init(TensorInfo(TensorShape(num_filters), 1, DataType::S32));
        _gemm_output_stage_shifts.allocator()->init(TensorInfo(TensorShape(num_filters), 1, DataType::S32));

        GEMMLowpOutputStageInfo gemmlowp_output_stage = gemm_info.gemmlowp_output_stage();
        gemmlowp_output_stage.output_data_type        = _matrix_a->info()->data_type();

        gemm_kernel_info.output_stage = gemmlowp_output_stage;

        if(_is_gemm_reshaped && gemmlowp_output_stage.type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
        {
            // Configure and tune matrix multiply kernel with fused output stage
            _mm_reshaped_only_rhs_kernel->configure(compile_context, _matrix_a, matrix_b, output, gemm_kernel_info, _a_offset == 0 ? nullptr : &_vector_sum_col,
                                                    _b_offset == 0 ? nullptr : &_vector_sum_row, c, &_gemm_output_stage_multipliers, &_gemm_output_stage_shifts);
        }
        else
        {
            _run_output_stage = true;

            _memory_group.manage(&_mm_result_s32);

            if(_is_gemm_reshaped)
            {
                _mm_reshaped_only_rhs_kernel->configure(compile_context, _matrix_a, matrix_b, &_mm_result_s32, gemm_kernel_info);
            }
            else
            {
                // Pick up the GEMM configuration
                std::tie(lhs_info, rhs_info) = CLGEMMNativeKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

                // Configure matrix multiply kernel
                _mm_native_kernel->configure(compile_context, _matrix_a, matrix_b, &_mm_result_s32, lhs_info, rhs_info, GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d));

                _offset_contribution_output_stage_kernel->configure(compile_context, &_mm_result_s32, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, c, output,
                                                                    a->info()->dimension(0),
                                                                    _a_offset, _b_offset, gemmlowp_output_stage, &_gemm_output_stage_multipliers, &_gemm_output_stage_shifts);
                _mm_result_s32.allocator()->allocate();
            }
        }

        _gemm_output_stage_multipliers.allocator()->allocate();
        _gemm_output_stage_shifts.allocator()->allocate();
        // Compute GEMM output multipliers and shifts for output stage
        _gemm_output_stage_multipliers.map();
        _gemm_output_stage_shifts.map();
        std::memcpy(_gemm_output_stage_multipliers.ptr_to_element(Coordinates(0)), gemm_info.gemmlowp_output_stage().gemmlowp_multipliers.data(), num_filters * sizeof(int32_t));
        std::memcpy(_gemm_output_stage_shifts.ptr_to_element(Coordinates(0)), gemm_info.gemmlowp_output_stage().gemmlowp_shifts.data(), num_filters * sizeof(int32_t));
        _gemm_output_stage_multipliers.unmap();
        _gemm_output_stage_shifts.unmap();
    }
    else
    {
        _run_offset_contribution = true;
        if(_is_gemm_reshaped)
        {
            // Configure and tune matrix multiply kernel
            _mm_reshaped_only_rhs_kernel->configure(compile_context, _matrix_a, matrix_b, output, gemm_kernel_info);
        }
        else
        {
            // Pick up the GEMM configuration
            std::tie(lhs_info, rhs_info) = CLGEMMNativeKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

            // Configure matrix multiply kernel
            _mm_native_kernel->configure(compile_context, _matrix_a, matrix_b, output, lhs_info, rhs_info, GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d));
        }

        // Configure offset contribution kernel
        _offset_contribution_kernel->configure(compile_context, output, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, c, a->info()->dimension(0), _a_offset,
                                               _b_offset);
    }

    // Allocate tensors
    if(_is_gemm_reshaped)
    {
        if(!_reshape_b_only_on_first_run)
        {
            _tmp_b.allocator()->allocate();
        }
    }

    if(_a_offset != 0 && !_reshape_b_only_on_first_run)
    {
        _vector_sum_col.allocator()->allocate();
    }

    if(_b_offset != 0)
    {
        _vector_sum_row.allocator()->allocate();
    }
}

Status CLGEMMLowpMatrixMultiplyCore::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info)
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

    bool reshape_matrix_b = is_gemm_reshaped(m, n, k, a->data_type(), gemm_info.reshape_b_only_on_first_run());

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d);

    bool convert_to_qasymm8 = is_data_type_quantized_per_channel(b->data_type()) && is_data_type_quantized_symmetric(b->data_type())
                              && is_data_type_quantized_asymmetric(a->data_type());
    TensorInfo weights_info(*b);
    if(convert_to_qasymm8)
    {
        b_offset = -128;
        weights_info.set_data_type(DataType::QASYMM8);
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthConvertLayerKernel::validate(b, &weights_info, ConvertPolicy::WRAP, 0));
    }
    const ITensorInfo *matrix_b_info = &weights_info;
    if(reshape_matrix_b)
    {
        matrix_b_info = &tmp_b_info;

        // Pick up the GEMM configuration
        std::tie(lhs_info, rhs_info) = CLGEMMReshapedOnlyRHSKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

        // Validate reshape RHS kernel
        auto_init_if_empty(tmp_b_info, weights_info.clone()->set_tensor_shape(compute_rhs_reshaped_shape(weights_info, rhs_info)));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeRHSMatrixKernel::validate(&weights_info, &tmp_b_info, rhs_info));
    }

    TensorInfo info_vector_sum_col{};
    TensorInfo info_vector_sum_row{};

    const GEMMLowpReductionKernelInfo reduction_info;
    // Validate matrix B reduction kernel only if _a_offset is not equal to 0
    if(a_offset != 0)
    {
        info_vector_sum_col = TensorInfo(compute_reductionA_shape(weights_info), 1, DataType::S32);

        // Configure Matrix B reduction kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixBReductionKernel::validate(&weights_info, &info_vector_sum_col, reduction_info));
    }

    // Validate Matrix A reduction kernel only if _b_offset is not equal to 0
    if(b_offset != 0)
    {
        info_vector_sum_row = TensorInfo(compute_reductionB_shape(*a), 1, DataType::S32);

        // Configure matrix A reduction kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixAReductionKernel::validate(a, &info_vector_sum_row, reduction_info));
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
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel::validate(matrix_a_info, matrix_b_info, output, gemm_kernel_info,
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
                ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info, gemm_kernel_info));
            }
            else
            {
                // Output tensor auto inizialitation if not yet initialized
                auto_init_if_empty(mm_result_s32_info, a->clone()->set_tensor_shape(compute_mm_shape(*matrix_a_info, *matrix_b_info, false, reshape_info)).set_data_type(DataType::S32));

                // Pick up the GEMM configuration
                std::tie(lhs_info, rhs_info) = CLGEMMNativeKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

                // Validate matrix multiply
                ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyNativeKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info, lhs_info, rhs_info, reshape_info));
            }

            // Validate offset contribution kernel
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpOffsetContributionOutputStageKernel::validate(&mm_result_s32_info,
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
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel::validate(matrix_a_info, matrix_b_info, output, gemm_kernel_info));
        }
        else
        {
            // Pick up the GEMM configuration
            std::tie(lhs_info, rhs_info) = CLGEMMNativeKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

            // Validate matrix multiply
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyNativeKernel::validate(matrix_a_info, matrix_b_info, output, lhs_info, rhs_info, reshape_info));
        }

        if(output->total_size() != 0)
        {
            // Validate offset contribution kernel
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpOffsetContributionKernel::validate(output,
                                                                                     a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                                     b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                                     c,
                                                                                     a_offset, b_offset));
        }
    }

    return Status{};
}

void CLGEMMLowpMatrixMultiplyCore::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_is_gemm_reshaped)
    {
        if(!_reshape_b_only_on_first_run)
        {
            // Run reshape matrix B
            CLScheduler::get().enqueue(*_mtx_b_reshape_kernel, false);
        }
    }

    // Run matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0 && !_reshape_b_only_on_first_run)
    {
        CLScheduler::get().enqueue(*_mtx_b_reduction_kernel, false);
    }

    // Run matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        CLScheduler::get().enqueue(*_mtx_a_reduction_kernel, false);
    }

    // Run matrix multiply
    if(_is_gemm_reshaped)
    {
        CLScheduler::get().enqueue(*_mm_reshaped_only_rhs_kernel, false);
    }
    else
    {
        CLScheduler::get().enqueue(*_mm_native_kernel, false);
    }
    if(_run_output_stage)
    {
        // Run offset contribution/output stage kernel
        CLScheduler::get().enqueue(*_offset_contribution_output_stage_kernel, true);
    }
    if(_run_offset_contribution)
    {
        // Run offset contribution kernel
        CLScheduler::get().enqueue(*_offset_contribution_kernel, true);
    }
}

void CLGEMMLowpMatrixMultiplyCore::prepare()
{
    if(!_is_prepared)
    {
        if(_convert_to_qasymm8)
        {
            _qasymm8_weights.allocator()->allocate();
            CLScheduler::get().enqueue(*_weights_to_qasymm8, false);
        }

        if(_is_gemm_reshaped && _reshape_b_only_on_first_run)
        {
            ARM_COMPUTE_ERROR_ON(!_original_b->is_used());

            // Run reshape kernel and mark original weights tensor as unused
            _tmp_b.allocator()->allocate();
            CLScheduler::get().enqueue(*_mtx_b_reshape_kernel, false);
            _original_b->mark_as_unused();
        }

        // Run matrix B reduction kernel only if _a_offset is not equal to 0
        if(_a_offset != 0 && _reshape_b_only_on_first_run)
        {
            _vector_sum_col.allocator()->allocate();
            CLScheduler::get().enqueue(*_mtx_b_reduction_kernel, false);
        }

        CLScheduler::get().queue().finish();
        _is_prepared = true;
    }
}
} // namespace arm_compute
