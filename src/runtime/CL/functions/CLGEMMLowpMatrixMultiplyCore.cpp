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
                        GPUTarget::G71, GPUTarget::G72, GPUTarget::G76,
                        GPUTarget::G51, GPUTarget::G51BIG, GPUTarget::G51LIT,
                        GPUTarget::G52, GPUTarget::G52LIT))
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
      _vector_sum_col(),
      _vector_sum_row(),
      _tmp_a(),
      _tmp_b(),
      _original_b(nullptr),
      _a_offset(0),
      _b_offset(0),
      _is_interleaved_transposed(true),
      _reshape_b_only_on_first_run(false),
      _is_prepared(false)
{
}

void CLGEMMLowpMatrixMultiplyCore::configure(const ICLTensor *a, const ICLTensor *b, ICLTensor *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
    ARM_COMPUTE_UNUSED(gemm_info);
    ARM_COMPUTE_ERROR_THROW_ON(CLGEMMLowpMatrixMultiplyCore::validate(a->info(), b->info(), output->info(), gemm_info));

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

    // Arguments used by GEMMReshapeInfo
    // If we pass the matrix A and matrix B reshaped to CLGEMMMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to CLGEMMReshapeInfo
    // in order to know how the matrices have been reshaped
    const int     m                         = a->info()->dimension(1);
    const int     n                         = b->info()->dimension(0);
    const int     k                         = a->info()->dimension(0);
    constexpr int mult_transpose1xW_width   = 1;
    constexpr int mult_interleave4x4_height = 1;

    // Check if we need to reshape the matrix A and matrix B
    _is_interleaved_transposed = is_interleaved_transposed(m, n, k, _reshape_b_only_on_first_run, gpu_target);

    if(_is_interleaved_transposed)
    {
        matrix_a = &_tmp_a;
        matrix_b = &_tmp_b;

        _memory_group.manage(&_tmp_a);
        if(!_reshape_b_only_on_first_run)
        {
            _memory_group.manage(&_tmp_b);
        }

        // Configure interleave kernel
        _mtx_a_reshape_kernel.configure(a, &_tmp_a, mult_interleave4x4_height);

        // Configure transpose kernel
        _mtx_b_reshape_kernel.configure(b, &_tmp_b, mult_transpose1xW_width);
    }

    // Configure matrix multiply kernel
    _mm_kernel.configure(matrix_a, matrix_b, output, _is_interleaved_transposed, GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height));

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

    // Configure offset contribution kernel
    _offset_contribution_kernel.configure(output, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, a->info()->dimension(0), _a_offset, _b_offset);

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

Status CLGEMMLowpMatrixMultiplyCore::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((a)->dimension(0) != (b)->dimension(1),
                                    "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((a)->dimension(1) != (output)->dimension(1),
                                    "The output matrix must have the same number of rows as the matrix A");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((b)->dimension(0) != (output)->dimension(0),
                                    "The output matrix must have the same number of columns as the matrix B");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    int32_t a_offset = a->quantization_info().offset;
    int32_t b_offset = b->quantization_info().offset;

    const int             m                         = a->dimension(1);
    const int             n                         = b->dimension(0);
    const int             k                         = a->dimension(0);
    constexpr int         mult_transpose1xW_width   = 1;
    constexpr int         mult_interleave4x4_height = 1;
    const int             depth_output_gemm3d       = gemm_info.depth_output_gemm3d();
    const GEMMReshapeInfo reshape_info(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, depth_output_gemm3d);

    bool reshape_matrices = is_interleaved_transposed(m, n, k, gemm_info.reshape_b_only_on_first_run(), CLScheduler::get().target());

    if(reshape_matrices)
    {
        TensorInfo info_a(compute_interleaved_shape(*a, mult_interleave4x4_height, gemm_info.reinterpret_input_as_3d()), 1, a->data_type());
        TensorInfo info_b(compute_transpose1xW_with_element_size_shape(*b, mult_transpose1xW_width), 1, b->data_type());

        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMInterleave4x4Kernel::validate(a, &info_a, mult_interleave4x4_height, gemm_info.reinterpret_input_as_3d()));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMTranspose1xWKernel::validate(b, &info_b, mult_transpose1xW_width));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyKernel::validate(&info_a, &info_b, output, reshape_matrices, reshape_info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyKernel::validate(a, b, output, reshape_matrices, reshape_info));
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

    // Validate offset contribution kernel
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpOffsetContributionKernel::validate(output,
                                                                             a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                             b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                             a_offset, b_offset));

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

    // Run offset contribution kernel
    CLScheduler::get().enqueue(_offset_contribution_kernel, true);

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
