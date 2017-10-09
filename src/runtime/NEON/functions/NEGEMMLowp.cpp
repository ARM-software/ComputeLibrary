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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowp.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMLowpAArch64V8P4Kernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

NEGEMMLowp::NEGEMMLowp(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _mm_func(), _mtx_a_reduction_kernel(), _mtx_b_reduction_kernel(), _finalize_kernel(), _vector_sum_col(), _vector_sum_row(), _mm_output(), _a_offset(0),
      _b_offset(0)
{
}

void NEGEMMLowp::configure(const ITensor *a, const ITensor *b, ITensor *output, int32_t a_offset, int32_t b_offset, int32_t c_offset, int32_t output_mult_int, int32_t shift)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN((a), 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(a, b, output);
    ARM_COMPUTE_ERROR_ON_MSG((a)->info()->dimension(0) != (b)->info()->dimension(1), "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_ERROR_ON_MSG((a)->info()->dimension(1) != (output)->info()->dimension(1), "The output matrix must have the same number of rows as the matrix A");
    ARM_COMPUTE_ERROR_ON_MSG((b)->info()->dimension(0) != (output)->info()->dimension(0), "The output matrix must have the same number of columns as the matrix B");

    _a_offset = a_offset;
    _b_offset = b_offset;

    // Initialize matrix multiply output tensor
    const TensorShape &shape_mm_output = output->info()->tensor_shape();
    TensorInfo         info_mm_output(shape_mm_output, 1, DataType::S32);
    _mm_output.allocator()->init(info_mm_output);
    _memory_group.manage(&_mm_output);

    // Initialize Matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0)
    {
        TensorShape shape_vector_sum_col = b->info()->tensor_shape();
        shape_vector_sum_col.remove_dimension(1);
        TensorInfo info_vector_sum_col(shape_vector_sum_col, 1, DataType::S32);
        _vector_sum_col.allocator()->init(info_vector_sum_col);
        _memory_group.manage(&_vector_sum_col);

        // Configure Matrix B reduction kernel
        _mtx_b_reduction_kernel.configure(b, &_vector_sum_col, a->info()->dimension(0), false);
    }

    // Initialize Matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        TensorShape shape_vector_sum_row = a->info()->tensor_shape();
        shape_vector_sum_row.set(Window::DimX, a->info()->dimension(1));
        shape_vector_sum_row.remove_dimension(1);
        TensorInfo info_vector_sum_row(shape_vector_sum_row, 1, DataType::S32);
        _vector_sum_row.allocator()->init(info_vector_sum_row);
        _memory_group.manage(&_vector_sum_row);

        // Configure Matrix A reduction kernel
        _mtx_a_reduction_kernel.configure(a, &_vector_sum_row, a->info()->dimension(0), false);
    }

    // Configure matrix multiply function
    _mm_func.configure(a, b, &_mm_output);

    // Configure finalize kernel
    _finalize_kernel.configure(_a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, &_mm_output, output, a->info()->dimension(0), a_offset, b_offset, c_offset,
                               output_mult_int, shift);

    // Allocate tensors
    _mm_output.allocator()->allocate();

    if(_a_offset != 0)
    {
        _vector_sum_col.allocator()->allocate();
    }

    if(_b_offset != 0)
    {
        _vector_sum_row.allocator()->allocate();
    }
}

void NEGEMMLowp::run()
{
    _memory_group.acquire();

    // Run matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        NEScheduler::get().schedule(&_mtx_a_reduction_kernel, Window::DimX);
    }

    // Run matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0)
    {
        NEScheduler::get().schedule(&_mtx_b_reduction_kernel, Window::DimX);
    }

    // Run matrix multiply core function
    _mm_func.run();

    // Run finalise kernel
    NEScheduler::get().schedule(&_finalize_kernel, Window::DimY);

    _memory_group.release();
}