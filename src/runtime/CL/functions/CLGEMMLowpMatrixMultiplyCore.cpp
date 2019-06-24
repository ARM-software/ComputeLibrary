/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/CL/gemm/native/CLGEMMNativeKernelConfiguration.h"
#include "arm_compute/core/CL/gemm/reshaped_only_rhs/CLGEMMReshapedOnlyRHSKernelConfiguration.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::cl_gemm;

namespace
{
inline bool is_gemm_reshaped(bool reshape_b_only_on_first_run, GPUTarget gpu_target)
{
    return (get_arch_from_target(gpu_target) != GPUTarget::MIDGARD) && (reshape_b_only_on_first_run);
}
} // namespace

CLGEMMLowpMatrixMultiplyCore::CLGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _mm_midgard_kernel(),
      _mm_native_kernel(),
      _mm_reshaped_only_rhs_kernel(),
      _mtx_b_reshape_kernel(),
      _mtx_a_reduction_kernel(),
      _mtx_b_reduction_kernel(),
      _offset_contribution_kernel(),
      _offset_contribution_output_stage_kernel(),
      _vector_sum_col(),
      _vector_sum_row(),
      _tmp_b(),
      _mm_result_s32(),
      _original_b(nullptr),
      _a_offset(0),
      _b_offset(0),
      _is_gemm_reshaped(true),
      _is_midgard(false),
      _reshape_b_only_on_first_run(false),
      _is_prepared(false),
      _fuse_output_stage(false)
{
}

void CLGEMMLowpMatrixMultiplyCore::configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLGEMMLowpMatrixMultiplyCore::validate(a->info(), b->info(), c != nullptr ? c->info() : nullptr, output->info(), gemm_info));

    _is_prepared                 = false;
    _original_b                  = b;
    _reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();
    _a_offset                    = a->info()->quantization_info().uniform().offset;
    _b_offset                    = b->info()->quantization_info().uniform().offset;

    // Get the GPU target
    const GPUTarget gpu_target = CLScheduler::get().target();

    // Set the target for the kernels
    _mm_midgard_kernel.set_target(gpu_target);
    _mm_native_kernel.set_target(gpu_target);
    _mm_reshaped_only_rhs_kernel.set_target(gpu_target);

    const ICLTensor *matrix_a = a;
    const ICLTensor *matrix_b = b;
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
    _is_gemm_reshaped = is_gemm_reshaped(_reshape_b_only_on_first_run, gpu_target);
    _is_midgard       = gpu_target == GPUTarget::MIDGARD;

    if(_is_gemm_reshaped)
    {
        matrix_b = &_tmp_b;

        if(!_reshape_b_only_on_first_run)
        {
            _memory_group.manage(&_tmp_b);
        }

        // Pick up the GEMM configuration
        std::tie(lhs_info, rhs_info) = CLGEMMReshapedOnlyRHSKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

        // Configure reshape RHS kernel
        _mtx_b_reshape_kernel.configure(b, &_tmp_b, rhs_info);
    }

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
        _mtx_b_reduction_kernel.configure(b, &_vector_sum_col);
    }

    // Initialize Matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        TensorInfo info_vector_sum_row(compute_reductionB_shape(*a->info()), 1, DataType::S32);
        _vector_sum_row.allocator()->init(info_vector_sum_row);
        _memory_group.manage(&_vector_sum_row);

        // Configure matrix A reduction kernel
        _mtx_a_reduction_kernel.configure(a, &_vector_sum_row);
    }

    // If GEMMLowpOutputStage != NONE, fuse the offset contribution with the output stage
    if(gemm_info.gemmlowp_output_stage().type != GEMMLowpOutputStageType::NONE)
    {
        _fuse_output_stage = true;

        _memory_group.manage(&_mm_result_s32);

        if(_is_gemm_reshaped)
        {
            // Configure and tune matrix multiply kernel
            _mm_reshaped_only_rhs_kernel.configure(matrix_a, matrix_b, &_mm_result_s32, lhs_info, rhs_info, GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d));
        }
        else
        {
            if(_is_midgard)
            {
                // Configure matrix multiply kernel
                _mm_midgard_kernel.configure(matrix_a, matrix_b, &_mm_result_s32, GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d));
            }
            else
            {
                // Pick up the GEMM configuration
                std::tie(lhs_info, rhs_info) = CLGEMMNativeKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

                // Configure matrix multiply kernel
                _mm_native_kernel.configure(matrix_a, matrix_b, &_mm_result_s32, lhs_info, rhs_info, GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d));
            }
        }

        // Configure offset contribution kernel
        _offset_contribution_output_stage_kernel.configure(&_mm_result_s32, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, c, output, a->info()->dimension(0),
                                                           _a_offset, _b_offset, gemm_info.gemmlowp_output_stage());

        _mm_result_s32.allocator()->allocate();
    }
    else
    {
        if(_is_gemm_reshaped)
        {
            // Configure and tune matrix multiply kernel
            _mm_reshaped_only_rhs_kernel.configure(matrix_a, matrix_b, output, lhs_info, rhs_info, GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d));
        }
        else
        {
            if(_is_midgard)
            {
                // Configure matrix multiply kernel
                _mm_midgard_kernel.configure(matrix_a, matrix_b, output, GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d));
            }
            else
            {
                // Pick up the GEMM configuration
                std::tie(lhs_info, rhs_info) = CLGEMMNativeKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

                // Configure matrix multiply kernel
                _mm_native_kernel.configure(matrix_a, matrix_b, output, lhs_info, rhs_info, GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d));
            }
        }

        // Configure offset contribution kernel
        _offset_contribution_kernel.configure(output, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, c, a->info()->dimension(0), _a_offset, _b_offset);
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
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    int32_t a_offset = a->quantization_info().uniform().offset;
    int32_t b_offset = b->quantization_info().uniform().offset;

    const ITensorInfo *matrix_a_info = a;
    const ITensorInfo *matrix_b_info = b;

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
    const bool         is_midgard              = gpu_target == GPUTarget::MIDGARD;

    bool reshape_matrix_b = is_gemm_reshaped(gemm_info.reshape_b_only_on_first_run(), CLScheduler::get().target());

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d);

    if(reshape_matrix_b)
    {
        matrix_b_info = &tmp_b_info;

        // Pick up the GEMM configuration
        std::tie(lhs_info, rhs_info) = CLGEMMReshapedOnlyRHSKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

        // Validate reshape RHS kernel
        auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeRHSMatrixKernel::validate(b, &tmp_b_info, rhs_info));
    }

    TensorInfo info_vector_sum_col{};
    TensorInfo info_vector_sum_row{};

    // Validate matrix B reduction kernel only if _a_offset is not equal to 0
    if(a_offset != 0)
    {
        info_vector_sum_col = TensorInfo(compute_reductionA_shape(*b), 1, DataType::S32);

        // Configure Matrix B reduction kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixBReductionKernel::validate(b, &info_vector_sum_col));
    }

    // Validate Matrix A reduction kernel only if _b_offset is not equal to 0
    if(b_offset != 0)
    {
        info_vector_sum_row = TensorInfo(compute_reductionB_shape(*a), 1, DataType::S32);

        // Configure matrix A reduction kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixAReductionKernel::validate(a, &info_vector_sum_row));
    }

    if(gemm_info.gemmlowp_output_stage().type != GEMMLowpOutputStageType::NONE)
    {
        TensorInfo mm_result_s32_info{};

        if(reshape_matrix_b)
        {
            // Output tensor auto inizialitation if not yet initialized
            auto_init_if_empty(mm_result_s32_info, a->clone()->set_tensor_shape(compute_mm_shape(*matrix_a_info, *matrix_b_info, reshape_info)).set_data_type(DataType::S32));

            // Validate matrix multiply
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info, lhs_info, rhs_info, reshape_info));
        }
        else
        {
            // Output tensor auto inizialitation if not yet initialized
            auto_init_if_empty(mm_result_s32_info, a->clone()->set_tensor_shape(compute_mm_shape(*matrix_a_info, *matrix_b_info, false, reshape_info)).set_data_type(DataType::S32));

            if(is_midgard)
            {
                // Validate matrix multiply
                ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info, reshape_info));
            }
            else
            {
                // Pick up the GEMM configuration
                std::tie(lhs_info, rhs_info) = CLGEMMNativeKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

                // Validate matrix multiply
                ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyNativeKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info, lhs_info, rhs_info, reshape_info));
            }
        }

        // Validate offset contribution kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpOffsetContributionOutputStageKernel::validate(&mm_result_s32_info,
                                                                                            a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                                            b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                                            c,
                                                                                            output,
                                                                                            a_offset, b_offset,
                                                                                            gemm_info.gemmlowp_output_stage()));
    }
    else
    {
        if(reshape_matrix_b)
        {
            // Validate matrix multiply
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel::validate(matrix_a_info, matrix_b_info, output, lhs_info, rhs_info, reshape_info));
        }
        else
        {
            if(is_midgard)
            {
                // Validate matrix multiply
                ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, output, reshape_info));
            }
            else
            {
                // Pick up the GEMM configuration
                std::tie(lhs_info, rhs_info) = CLGEMMNativeKernelConfigurationFactory::create(gpu_target)->configure(m, n, k, batch_size, DataType::QASYMM8);

                // Validate matrix multiply
                ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyNativeKernel::validate(matrix_a_info, matrix_b_info, output, lhs_info, rhs_info, reshape_info));
            }
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
            CLScheduler::get().enqueue(_mtx_b_reshape_kernel, false);
        }
    }

    // Run matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0 && !_reshape_b_only_on_first_run)
    {
        CLScheduler::get().enqueue(_mtx_b_reduction_kernel, false);
    }

    // Run matrix multiply
    if(_is_gemm_reshaped)
    {
        CLScheduler::get().enqueue(_mm_reshaped_only_rhs_kernel, false);
    }
    else
    {
        if(_is_midgard)
        {
            CLScheduler::get().enqueue(_mm_midgard_kernel, false);
        }
        else
        {
            CLScheduler::get().enqueue(_mm_native_kernel, false);
        }
    }

    // Run matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        CLScheduler::get().enqueue(_mtx_a_reduction_kernel, false);
    }

    if(_fuse_output_stage)
    {
        // Run offset contribution/output stage kernel
        CLScheduler::get().enqueue(_offset_contribution_output_stage_kernel, true);
    }
    else
    {
        // Run offset contribution kernel
        CLScheduler::get().enqueue(_offset_contribution_kernel, true);
    }
}

void CLGEMMLowpMatrixMultiplyCore::prepare()
{
    if(!_is_prepared)
    {
        if(_is_gemm_reshaped && _reshape_b_only_on_first_run)
        {
            ARM_COMPUTE_ERROR_ON(!_original_b->is_used());

            // Run reshape kernel and mark original weights tensor as unused
            _tmp_b.allocator()->allocate();
            CLScheduler::get().enqueue(_mtx_b_reshape_kernel, false);
            _original_b->mark_as_unused();
        }

        // Run matrix B reduction kernel only if _a_offset is not equal to 0
        if(_a_offset != 0 && _reshape_b_only_on_first_run)
        {
            _vector_sum_col.allocator()->allocate();
            CLScheduler::get().enqueue(_mtx_b_reduction_kernel, false);
        }

        CLScheduler::get().queue().finish();
        _is_prepared = true;
    }
}
} // namespace arm_compute