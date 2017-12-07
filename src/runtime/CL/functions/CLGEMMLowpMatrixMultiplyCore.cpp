/*
 * Copyright (c) 2017 ARM Limited.
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

CLGEMMLowpMatrixMultiplyCore::CLGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _mm_kernel(), _mtx_a_reshape_kernel(), _mtx_b_reshape_kernel(), _mtx_a_reduction_kernel(), _mtx_b_reduction_kernel(), _offset_contribution_kernel(),
      _vector_sum_col(), _vector_sum_row(), _tmp_a(), _tmp_b(), _a_offset(0), _b_offset(0), _is_interleaved_transposed(true), _is_first_run(true), _reshape_b_only_on_first_run(false)
{
}

void CLGEMMLowpMatrixMultiplyCore::configure(const ICLTensor *a, const ICLTensor *b, ICLTensor *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
    ARM_COMPUTE_UNUSED(gemm_info);
    ARM_COMPUTE_ERROR_THROW_ON(CLGEMMLowpMatrixMultiplyCore::validate(a->info(), b->info(), output->info(), gemm_info));

    _reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();
    _a_offset                    = a->info()->quantization_info().offset;
    _b_offset                    = b->info()->quantization_info().offset;

    // If the input tensor has less than 16 rows, we run a special version of GEMMLowp without reshaping the input tensors
    _is_interleaved_transposed = a->info()->dimension(1) > 16;

    const ICLTensor *matrix_a = a;
    const ICLTensor *matrix_b = b;

    if(_is_interleaved_transposed)
    {
        matrix_a = &_tmp_a;
        matrix_b = &_tmp_b;

        TensorInfo info_a(compute_interleaved_shape(*a->info()), 1, a->info()->data_type());
        TensorInfo info_b(compute_transpose1xW_shape(*b->info()), 1, b->info()->data_type());
        _tmp_a.allocator()->init(info_a);
        _tmp_b.allocator()->init(info_b);
        _memory_group.manage(&_tmp_a);
        _memory_group.manage(&_tmp_b);

        // Configure interleave kernel
        _mtx_a_reshape_kernel.configure(a, &_tmp_a);

        // Configure transpose kernel
        _mtx_b_reshape_kernel.configure(b, &_tmp_b);
    }

    // Configure matrix multiply kernel
    _mm_kernel.configure(matrix_a, matrix_b, output, _is_interleaved_transposed);

    // Initialize matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0)
    {
        TensorInfo info_vector_sum_col(compute_reductionA_shape(*b->info()), 1, DataType::S32);
        _vector_sum_col.allocator()->init(info_vector_sum_col);
        _memory_group.manage(&_vector_sum_col);

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
        _tmp_b.allocator()->allocate();
    }

    if(_a_offset != 0)
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

    int32_t a_offset                  = a->quantization_info().offset;
    int32_t b_offset                  = b->quantization_info().offset;
    bool    is_interleaved_transposed = a->dimension(1) > 16;

    if(is_interleaved_transposed)
    {
        TensorInfo info_a(compute_interleaved_shape(*a), 1, a->data_type());
        TensorInfo info_b(compute_transpose1xW_shape(*b), 1, b->data_type());

        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMInterleave4x4Kernel::validate(a, &info_a));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMTranspose1xWKernel::validate(b, &info_b));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyKernel::validate(&info_a, &info_b, output));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyKernel::validate(a, b, output));
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
    _memory_group.acquire();

    if(_is_interleaved_transposed)
    {
        // Run reshape matrix A
        CLScheduler::get().enqueue(_mtx_a_reshape_kernel, false);

        if(_is_first_run || !_reshape_b_only_on_first_run)
        {
            // Run reshape matrix B
            CLScheduler::get().enqueue(_mtx_b_reshape_kernel, false);
        }
    }

    // Note: if _reshape_b_only_on_first_run = true, the reduction kernel can be executed only once
    if(_is_first_run || !_reshape_b_only_on_first_run)
    {
        // Run matrix B reduction kernel only if _a_offset is not equal to 0
        if(_a_offset != 0)
        {
            CLScheduler::get().enqueue(_mtx_b_reduction_kernel, false);
        }
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

    _is_first_run = false;
}
