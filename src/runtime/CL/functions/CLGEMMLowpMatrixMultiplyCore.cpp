/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
inline bool is_interleaved_transposed(int m, int n, int k, bool reshape_b_only_on_first_run, GPUTarget gpu_target)
{
    bool flag = true;

    if(gpu_target_is_in(gpu_target,
                        GPUTarget::G71, GPUTarget::G72,
                        GPUTarget::G51, GPUTarget::G51BIG, GPUTarget::G51LIT))
    {
        // COMPMID-852
        if(k > 256 && m > 4 && reshape_b_only_on_first_run)
        {
            flag = ((0.72f + n * 0.10766f) < (n * 0.1284f));
        }
        else
        {
            flag = false;
        }
    }
    else
    {
        flag = m > 1;
    }

    return flag;
}
} // namespace

CLGEMMLowpMatrixMultiplyCore::CLGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _mm_kernel(),
      _mtx_a_reshape_kernel(),
      _mtx_b_reshape_kernel(),
      _mtx_a_reduction_kernel(),
      _mtx_b_reduction_kernel(),
      _offset_contribution_kernel(),
      _offset_contribution_output_stage_kernel(),
      _vector_sum_col(),
      _vector_sum_row(),
      _tmp_a(),
      _tmp_b(),
      _mm_result_s32(),
      _original_b(nullptr),
      _a_offset(0),
      _b_offset(0),
      _is_interleaved_transposed(true),
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
    _a_offset                    = a->info()->quantization_info().offset;
    _b_offset                    = b->info()->quantization_info().offset;

    // Get the GPU target
    const GPUTarget gpu_target = CLScheduler::get().target();

    // Set the target for the kernels
    _mtx_a_reshape_kernel.set_target(gpu_target);
    _mm_kernel.set_target(gpu_target);

    const ICLTensor *matrix_a = a;
    const ICLTensor *matrix_b = b;
    GEMMRHSMatrixInfo rhs_info;

    // Arguments used by GEMMReshapeInfo
    // If we pass the matrix A and matrix B reshaped to CLGEMMMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to CLGEMMReshapeInfo
    // in order to know how the matrices have been reshaped
    bool          reinterpret_input_as_3d   = gemm_info.reinterpret_input_as_3d();
    const bool    unroll_block              = dot8_supported(CLKernelLibrary::get().get_device());
    const int     m                         = reinterpret_input_as_3d ? (a->info()->dimension(1) * a->info()->dimension(2)) : a->info()->dimension(1);
    const int     n                         = b->info()->dimension(0);
    const int     k                         = a->info()->dimension(0);
    const int     depth_output_gemm3d       = gemm_info.depth_output_gemm3d();
    constexpr int mult_transpose1xW_width   = 1;
    constexpr int mult_interleave4x4_height = 1;
    rhs_info.n0                             = 16 / b->info()->element_size();
    rhs_info.k0                             = 1;
    rhs_info.h0                             = mult_transpose1xW_width;
    rhs_info.interleave                     = false;
    rhs_info.transpose                      = false;

    // Check if we need to reshape the matrix A and matrix B
    _is_interleaved_transposed = is_interleaved_transposed(m, n, k, _reshape_b_only_on_first_run, gpu_target);

    if(_is_interleaved_transposed)
    {
        // if _is_interleaved_transposed is set, force reinterpret_input_as_3d to be false as the output of CLGEMMInterleaveKernel will be 2D
        reinterpret_input_as_3d = false;

        matrix_a = &_tmp_a;
        matrix_b = &_tmp_b;

        _memory_group.manage(&_tmp_a);
        if(!_reshape_b_only_on_first_run)
        {
            _memory_group.manage(&_tmp_b);
        }

        // Configure interleave kernel
        _mtx_a_reshape_kernel.configure(a, &_tmp_a, mult_interleave4x4_height, gemm_info.reinterpret_input_as_3d(), unroll_block);

        // Configure transpose kernel
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

        // Configure matrix multiply kernel
        _mm_kernel.configure(matrix_a, matrix_b, &_mm_result_s32, _is_interleaved_transposed, GEMMReshapeInfo(m, n, k,
                                                                                                              mult_transpose1xW_width, mult_interleave4x4_height,
                                                                                                              depth_output_gemm3d, reinterpret_input_as_3d));

        // Configure offset contribution kernel
        _offset_contribution_output_stage_kernel.configure(&_mm_result_s32, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, c, output, a->info()->dimension(0),
                                                           _a_offset, _b_offset, gemm_info.gemmlowp_output_stage());

        _mm_result_s32.allocator()->allocate();
    }
    else
    {
        // Configure matrix multiply kernel
        _mm_kernel.configure(matrix_a, matrix_b, output, _is_interleaved_transposed, GEMMReshapeInfo(m, n, k,
                                                                                                     mult_transpose1xW_width, mult_interleave4x4_height,
                                                                                                     depth_output_gemm3d, reinterpret_input_as_3d));

        // Configure offset contribution kernel
        _offset_contribution_kernel.configure(output, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, c, a->info()->dimension(0), _a_offset, _b_offset);
    }

    // Allocate tensors
    if(_is_interleaved_transposed)
    {
        _tmp_a.allocator()->allocate();
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

    int32_t a_offset = a->quantization_info().offset;
    int32_t b_offset = b->quantization_info().offset;

    const ITensorInfo *matrix_a_info = a;
    const ITensorInfo *matrix_b_info = b;

    TensorInfo        tmp_a_info{};
    TensorInfo        tmp_b_info{};
    GEMMRHSMatrixInfo rhs_info;

    bool          reinterpret_input_as_3d   = gemm_info.reinterpret_input_as_3d();
    const int     m                         = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const int     n                         = b->dimension(0);
    const int     k                         = a->dimension(0);
    constexpr int mult_transpose1xW_width   = 1;
    constexpr int mult_interleave4x4_height = 1;
    const int     depth_output_gemm3d       = gemm_info.depth_output_gemm3d();
    rhs_info.n0                             = 16 / b->element_size();
    rhs_info.k0                             = 1;
    rhs_info.h0                             = mult_transpose1xW_width;
    rhs_info.interleave                     = false;
    rhs_info.transpose                      = false;

    bool reshape_matrices = is_interleaved_transposed(m, n, k, gemm_info.reshape_b_only_on_first_run(), CLScheduler::get().target());

    // if reshape_matrices is set, force reinterpret_input_as_3d to be false as the output of CLGEMMInterleaveKernel will be 2D
    if(reshape_matrices)
    {
        reinterpret_input_as_3d = false;
    }

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, depth_output_gemm3d, reinterpret_input_as_3d);

    if(reshape_matrices)
    {
        matrix_a_info = &tmp_a_info;
        matrix_b_info = &tmp_b_info;

        // Validate interleave kernel
        auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_interleaved_shape(*a, mult_interleave4x4_height, gemm_info.reinterpret_input_as_3d())));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMInterleave4x4Kernel::validate(a, &tmp_a_info, mult_interleave4x4_height, gemm_info.reinterpret_input_as_3d()));

        // Validate transpose kernel

        auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeRHSMatrixKernel::validate(b, &tmp_b_info, rhs_info));
    }

    TensorInfo info_vector_sum_col, info_vector_sum_row;

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

        // Output tensor auto inizialitation if not yet initialized
        auto_init_if_empty(mm_result_s32_info, a->clone()->set_tensor_shape(compute_mm_shape(*matrix_a_info, *matrix_b_info, reshape_matrices, reshape_info)).set_data_type(DataType::S32));

        // Validate matrix multiply
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info, reshape_matrices, reshape_info));

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
        // Validate matrix multiply
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, output, reshape_matrices, reshape_info));

        // Validate offset contribution kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpOffsetContributionKernel::validate(output,
                                                                                 a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                                 b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                                 c,
                                                                                 a_offset, b_offset));
    }

    return Status{};
}

void CLGEMMLowpMatrixMultiplyCore::run()
{
    prepare();

    _memory_group.acquire();

    if(_is_interleaved_transposed)
    {
        // Run reshape matrix A
        CLScheduler::get().enqueue(_mtx_a_reshape_kernel, false);

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
    CLScheduler::get().enqueue(_mm_kernel, false);

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

    _memory_group.release();
}

void CLGEMMLowpMatrixMultiplyCore::prepare()
{
    if(!_is_prepared)
    {
        if(_is_interleaved_transposed && _reshape_b_only_on_first_run)
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
