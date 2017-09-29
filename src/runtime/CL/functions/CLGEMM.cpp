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
#include "arm_compute/runtime/CL/functions/CLGEMM.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixAdditionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/ITensorAllocator.h"

using namespace arm_compute;

CLGEMM::CLGEMM(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _interleave_kernel(), _transpose_kernel(), _mm_kernel(), _ma_kernel(), _tmp_a(), _tmp_b(), _is_interleaved_transposed(false), _run_addition(false)
{
}

void CLGEMM::configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(a, b, output);

    if(c != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(a, c);
        ARM_COMPUTE_ERROR_ON_MSG(a->info()->dimension(1) != c->info()->dimension(1), "The C matrix must have the same number of rows as the matrix A");
        ARM_COMPUTE_ERROR_ON_MSG(b->info()->dimension(0) != c->info()->dimension(0), "The C matrix must have the same number of columns as the matrix C");
        ARM_COMPUTE_ERROR_ON_MSG(c->info()->dimension(0) != output->info()->dimension(0), "The C matrix must have the same number of rows as the output matrix");
        ARM_COMPUTE_ERROR_ON_MSG(c->info()->dimension(1) != output->info()->dimension(1), "The C matrix must have the same number of columns as the output matrix");
    }

    ARM_COMPUTE_ERROR_ON_MSG(a->info()->dimension(0) != b->info()->dimension(1), "The product AB is defined only if the number of columns in A is equal to the number of rows in B");

    // If the input tensor has less than 16 rows, we run a special version of GEMM without reshaping the input tensors
    _is_interleaved_transposed = a->info()->dimension(1) > 16;

    const ICLTensor *matrix_a = a;
    const ICLTensor *matrix_b = b;

    if(_is_interleaved_transposed)
    {
        matrix_a = &_tmp_a;
        matrix_b = &_tmp_b;

        TensorShape shape_tmp_a = a->info()->tensor_shape();
        TensorShape shape_tmp_b = b->info()->tensor_shape();

        shape_tmp_a.set(0, a->info()->dimension(0) * 4);
        shape_tmp_a.set(1, std::ceil(a->info()->dimension(1) / 4.0f));

        const unsigned int transpose_w = max_cl_vector_width / data_size_from_type(b->info()->data_type());
        shape_tmp_b.set(0, b->info()->dimension(1) * transpose_w);
        shape_tmp_b.set(1, std::ceil(b->info()->dimension(0) / static_cast<float>(transpose_w)));

        TensorInfo info_a(shape_tmp_a, 1, a->info()->data_type(), a->info()->fixed_point_position());
        _tmp_a.allocator()->init(info_a);

        TensorInfo info_b(shape_tmp_b, 1, b->info()->data_type(), b->info()->fixed_point_position());
        _tmp_b.allocator()->init(info_b);

        // Manage intermediate buffers
        _memory_group.manage(&_tmp_a);
        _memory_group.manage(&_tmp_b);

        // Configure interleave kernel
        _interleave_kernel.configure(a, &_tmp_a);

        // Configure transpose kernel
        _transpose_kernel.configure(b, &_tmp_b);

        // Configure matrix multiply kernel
        _mm_kernel.set_target(CLScheduler::get().target());
    }

    _mm_kernel.configure(matrix_a, matrix_b, output, alpha, _is_interleaved_transposed);

    if(_is_interleaved_transposed)
    {
        // Allocate intermediate tensors
        _tmp_a.allocator()->allocate();
        _tmp_b.allocator()->allocate();
    }

    // Configure matrix addition kernel
    if(beta != 0 && c != nullptr)
    {
        _ma_kernel.configure(c, output, beta);
        _run_addition = true;
    }
}

void CLGEMM::run()
{
    _memory_group.acquire();

    if(_is_interleaved_transposed)
    {
        // Run interleave kernel
        CLScheduler::get().enqueue(_interleave_kernel, false);

        // Run transpose kernel
        CLScheduler::get().enqueue(_transpose_kernel, false);
    }

    // Run matrix multiply kernel
    CLScheduler::get().enqueue(_mm_kernel, !_run_addition);

    // Run matrix addition kernel
    if(_run_addition)
    {
        CLScheduler::get().enqueue(_ma_kernel);
    }

    _memory_group.release();
}
