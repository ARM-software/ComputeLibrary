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
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/arm32/NEGEMMAArch32Kernel.h"
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMAArch64Kernel.h"
#include "arm_compute/core/NEON/kernels/arm64/NEHGEMMAArch64FP16Kernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
#include "arm_compute/core/NEON/kernels/assembly/gemm_interleaved.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a32_sgemm_8x6.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_hgemm_24x8.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_sgemm_12x8.hpp"
} // namespace arm_compute

#include <cmath>

namespace arm_compute
{
NEGEMM::NEGEMM(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _interleave_kernel(), _transpose_kernel(), _mm_kernel(), _mm_optimised_kernel(nullptr), _ma_kernel(), _tmp_a(), _tmp_b(), _workspace(),
      _run_vector_matrix_multiplication(false), _run_addition(false), _is_first_run(true), _reshape_b_only_on_first_run(false)
{
}

void NEGEMM::configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::F32, DataType::F16, DataType::QS8, DataType::QS16);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(a, b, d);
    ARM_COMPUTE_ERROR_ON_MSG(a->info()->dimension(0) != b->info()->dimension(1), "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    if(c != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(c, 1, DataType::F32, DataType::F16, DataType::QS8, DataType::QS16);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(a, c);
        ARM_COMPUTE_ERROR_ON_MSG(a->info()->dimension(1) != c->info()->dimension(1), "The C matrix must have the same number of rows as the matrix A");
        ARM_COMPUTE_ERROR_ON_MSG(b->info()->dimension(0) != c->info()->dimension(0), "The C matrix must have the same number of columns as the matrix B");
        ARM_COMPUTE_ERROR_ON_MSG(c->info()->dimension(0) != d->info()->dimension(0), "The C matrix must have the same number of rows as the output matrix");
        ARM_COMPUTE_ERROR_ON_MSG(c->info()->dimension(1) != d->info()->dimension(1), "The C matrix must have the same number of columns as the output matrix");
    }

    // Check if we need to reshape the matrix B only on the first run
    _reshape_b_only_on_first_run      = gemm_info.reshape_b_only_on_first_run();
    _run_vector_matrix_multiplication = a->info()->dimension(1) < 2;

    // Check if the first input tensor is a vector.
    // If so, all the kernels for reshaping the tensors can be skipped
    if(_run_vector_matrix_multiplication)
    {
        // Configure the matrix multiply kernel
        _mm_kernel.configure(a, b, d, alpha);

        // Configure matrix addition kernel
        if(beta != 0 && c != nullptr)
        {
            _ma_kernel.configure(c, d, beta);
            _run_addition = true;
        }
    }
    else
    {
#if defined(__arm__)
        if(NEScheduler::get().cpu_info().CPU == CPUTarget::ARMV7 && a->info()->data_type() == DataType::F32 && (c == nullptr || beta == 0.f))
        {
            _mm_optimised_kernel = support::cpp14::make_unique<NEGEMMAArch32Kernel>();
        }
#elif defined(__aarch64__)
        if(NEScheduler::get().cpu_info().CPU >= CPUTarget::ARMV8 && a->info()->data_type() == DataType::F32 && (c == nullptr || beta == 0.f))
        {
            _mm_optimised_kernel = support::cpp14::make_unique<NEGEMMAArch64Kernel>();
        }
        else if(a->info()->data_type() == DataType::F16 && (c == nullptr || beta == 0.f))
        {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            _mm_optimised_kernel = support::cpp14::make_unique<NEHGEMMAArch64FP16Kernel>();
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            ARM_COMPUTE_ERROR("Recompile the library with arch=arm64-v8.2-a to enable support for FP16.");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        }
#endif /* defined(__arm__) || defined(__aarch64__) */

#if defined(__arm__) || defined(__aarch64__)
        if(_mm_optimised_kernel != nullptr)
        {
            struct CPUInfo ci = NEScheduler::get().cpu_info();

            const int M = d->info()->tensor_shape().y();
            const int N = d->info()->tensor_shape().x();
            const int K = a->info()->tensor_shape().x();

            size_t workbench_size = 0;

#if defined(__arm__)
            workbench_size = GemmInterleaved<sgemm_8x6, sgemm_8x6::operand_type, sgemm_8x6::result_type>(&ci, M, N, K, false, false).get_working_size();
#elif defined(__aarch64__)
            if(a->info()->data_type() == DataType::F32)
            {
                workbench_size = GemmInterleaved<sgemm_12x8, sgemm_12x8::operand_type, sgemm_12x8::result_type>(&ci, M, N, K, false, false).get_working_size();
            }
            else if(a->info()->data_type() == DataType::F16)
            {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                workbench_size = GemmInterleaved<hgemm_24x8, hgemm_24x8::operand_type, hgemm_24x8::result_type>(&ci, M, N, K, false, false).get_working_size();
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                ARM_COMPUTE_ERROR("Recompile the library with arch=arm64-v8.2-a to enable support for FP16.");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            }
#endif /* defined(__arm__) || defined(__aarch64__) */

            constexpr size_t alignment = 4096;
            ARM_COMPUTE_ERROR_ON_MSG(workbench_size == 0, "size cannot be 0");
            _workspace.allocator()->init(TensorInfo(TensorShape{ (workbench_size + alignment - 1) * NEScheduler::get().num_threads() }, 1, DataType::S8));
            _memory_group.manage(&_workspace);

            // Configure matrix multiplication kernel
            _mm_optimised_kernel->configure(a, b, d, &_workspace, alpha, 0.f, false /* is_transposed_0 */, false /* is_transposed_1 */);
            _workspace.allocator()->allocate();
        }
        else
#endif /* defined(__arm__) || defined(__aarch64__) */
        {
            TensorShape shape_tmp_a = a->info()->tensor_shape();
            TensorShape shape_tmp_b = b->info()->tensor_shape();

            shape_tmp_a.set(0, a->info()->dimension(0) * 4);
            shape_tmp_a.set(1, std::ceil(a->info()->dimension(1) / 4.0f));

            const unsigned int transpose_w = 16 / data_size_from_type(b->info()->data_type());
            shape_tmp_b.set(0, b->info()->dimension(1) * transpose_w);
            shape_tmp_b.set(1, std::ceil(b->info()->dimension(0) / static_cast<float>(transpose_w)));

            TensorInfo info_a(shape_tmp_a, 1, a->info()->data_type(), a->info()->fixed_point_position());
            TensorInfo info_b(shape_tmp_b, 1, b->info()->data_type(), a->info()->fixed_point_position());

            _tmp_a.allocator()->init(info_a);
            _tmp_b.allocator()->init(info_b);

            // Manage intermediate buffers
            _memory_group.manage(&_tmp_a);
            _memory_group.manage(&_tmp_b);

            // Configure interleave kernel
            _interleave_kernel.configure(a, &_tmp_a);

            // Configure transpose kernel
            _transpose_kernel.configure(b, &_tmp_b);

            // Configure matrix multiplication kernel
            _mm_kernel.configure(&_tmp_a, &_tmp_b, d, alpha);

            // Allocate once the all configure methods have been called
            _tmp_a.allocator()->allocate();
            _tmp_b.allocator()->allocate();

            // Configure matrix addition kernel
            if(beta != 0 && c != nullptr)
            {
                _ma_kernel.configure(c, d, beta);
                _run_addition = true;
            }
        }
    }
}

void NEGEMM::run()
{
    _memory_group.acquire();

    if(_mm_optimised_kernel != nullptr)
    {
        NEScheduler::get().schedule(_mm_optimised_kernel.get(), Window::DimY);
        _memory_group.release();
    }
    else
    {
        if(!_run_vector_matrix_multiplication)
        {
            // Run interleave kernel
            NEScheduler::get().schedule(&_interleave_kernel, Window::DimY);

            if(_is_first_run)
            {
                // Run transpose kernel
                NEScheduler::get().schedule(&_transpose_kernel, Window::DimY);

                _is_first_run = false;
            }
            else if(!_reshape_b_only_on_first_run)
            {
                // Run transpose kernel
                NEScheduler::get().schedule(&_transpose_kernel, Window::DimY);
            }
        }

        NEScheduler::get().schedule(&_mm_kernel, _run_vector_matrix_multiplication ? Window::DimX : Window::DimY);

        _memory_group.release();

        // Run matrix addition kernel
        if(_run_addition)
        {
            NEScheduler::get().schedule(&_ma_kernel, Window::DimY);
        }
    }
}
} // namespace arm_compute
