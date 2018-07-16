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
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMAssemblyDispatch.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

#include <cmath>

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
NEGEMM::NEGEMM(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _interleave_kernel(), _transpose_kernel(), _mm_kernel(), _asm_glue(memory_manager), _ma_kernel(), _tmp_a(), _tmp_b(), _original_b(nullptr),
      _run_vector_matrix_multiplication(false), _run_addition(false), _reshape_b_only_on_first_run(false), _is_prepared(false)
{
}

void NEGEMM::configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMM::validate(a->info(), b->info(), (c != nullptr) ? c->info() : nullptr, d->info(), alpha, beta, gemm_info));

    // Check if we need to reshape the matrix B only on the first run
    _is_prepared                      = false;
    _reshape_b_only_on_first_run      = gemm_info.reshape_b_only_on_first_run();
    _run_vector_matrix_multiplication = a->info()->dimension(1) < 2;
    _original_b                       = b;

    bool run_optimised = c == nullptr && bool(NEGEMMAssemblyDispatch::validate(a->info(), b->info(), d->info(), alpha, beta, _reshape_b_only_on_first_run));
    if(run_optimised)
    {
        _asm_glue.configure(a, b, d, alpha, beta, _reshape_b_only_on_first_run);
        run_optimised = _asm_glue.is_configured();
    }

    // Check if the first input tensor is a vector.
    // If so, all the kernels for reshaping the tensors can be skipped
    if(_run_vector_matrix_multiplication)
    {
        if(!run_optimised)
        {
            // Configure the matrix multiply kernel
            _mm_kernel.configure(a, b, d, alpha, false);
        }

        // Configure matrix addition kernel
        if(beta != 0 && c != nullptr)
        {
            _ma_kernel.configure(c, d, beta);
            _run_addition = true;
        }
    }
    else
    {
        if(!run_optimised)
        {
            TensorShape shape_tmp_a = a->info()->tensor_shape();
            TensorShape shape_tmp_b = b->info()->tensor_shape();

            shape_tmp_a.set(0, a->info()->dimension(0) * 4);
            shape_tmp_a.set(1, std::ceil(a->info()->dimension(1) / 4.0f));

            const unsigned int transpose_w = 16 / data_size_from_type(b->info()->data_type());
            shape_tmp_b.set(0, b->info()->dimension(1) * transpose_w);
            shape_tmp_b.set(1, std::ceil(b->info()->dimension(0) / static_cast<float>(transpose_w)));

            TensorInfo info_a(shape_tmp_a, 1, a->info()->data_type());
            TensorInfo info_b(shape_tmp_b, 1, b->info()->data_type());

            _tmp_a.allocator()->init(info_a);
            _tmp_b.allocator()->init(info_b);

            // Manage intermediate buffers
            _memory_group.manage(&_tmp_a);
            if(!_reshape_b_only_on_first_run)
            {
                _memory_group.manage(&_tmp_b);
            }

            int m = a->info()->dimension(1);
            int n = b->info()->dimension(0);
            int k = a->info()->dimension(0);

            // Configure interleave kernel
            _interleave_kernel.configure(a, &_tmp_a);

            // Configure transpose kernel
            _transpose_kernel.configure(b, &_tmp_b);

            // Configure matrix multiplication kernel
            _mm_kernel.configure(&_tmp_a, &_tmp_b, d, alpha, true, GEMMReshapeInfo(m, n, k));

            // Allocate once the all configure methods have been called
            _tmp_a.allocator()->allocate();
            if(!_reshape_b_only_on_first_run)
            {
                _tmp_b.allocator()->allocate();
            }

            // Configure matrix addition kernel
            if(beta != 0 && c != nullptr)
            {
                _ma_kernel.configure(c, d, beta);
                _run_addition = true;
            }
        }
    }
}

Status NEGEMM::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(a);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->dimension(0) != b->dimension(1), "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    if(c != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(c, 1, DataType::F32, DataType::F16);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, c);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->dimension(1) != c->dimension(1), "The C matrix must have the same number of rows as the matrix A");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(b->dimension(0) != c->dimension(0), "The C matrix must have the same number of columns as the matrix B");
    }

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(b->dimension(0) != output->dimension(0));
        ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1));
    }

    // Check if the first input tensor is a vector.
    const bool run_vector_matrix_multiplication = a->dimension(1) < 2;
    // Check if we need to reshape the matrix A and matrix B
    const bool run_interleave_transpose = !run_vector_matrix_multiplication && !(gemm_info.reshape_b_only_on_first_run());
    // Check if we need to run the optimized assembly kernel
    const bool run_optimised = c == nullptr && bool(NEGEMMAssemblyDispatch::validate(a, b, output, alpha, beta, true));

    const ITensorInfo *matrix_a_info = a;
    const ITensorInfo *matrix_b_info = b;

    TensorInfo tmp_a_info{};
    TensorInfo tmp_b_info{};
    TensorInfo tmp_output_info = *output->clone();

    // Arguments used by GEMMReshapeInfo
    // If we pass the matrix A and matrix B reshaped to NEGEMMMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to NEGEMMReshapeInfo
    // in order to know how the matrices have been reshaped
    const int m                         = a->dimension(1);
    const int n                         = b->dimension(0);
    const int k                         = a->dimension(0);
    int       mult_transpose1xW_width   = 1;
    int       mult_interleave4x4_height = 1;

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, gemm_info.depth_output_gemm3d());

    // Initialize shapes
    if(run_interleave_transpose)
    {
        matrix_a_info = &tmp_a_info;
        auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_interleaved_shape(*a, mult_interleave4x4_height)));

        matrix_b_info = &tmp_b_info;
        auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_transpose1xW_with_element_size_shape(*b, mult_transpose1xW_width)));

        auto_init_if_empty(tmp_output_info, matrix_a_info->clone()->set_tensor_shape(compute_mm_shape(*matrix_a_info, *matrix_b_info, run_interleave_transpose, reshape_info)));
    }

    // Validate kernels
    if(run_optimised && run_interleave_transpose)
    {
        /* Interleave */
        TensorShape tensor_shape0{ matrix_a_info->tensor_shape() };
        tensor_shape0.set(0, k);
        tensor_shape0.set(1, m);

        const TensorInfo tensor_info0          = matrix_a_info->clone()->set_tensor_shape(tensor_shape0);
        const TensorInfo tensor_info_reshaped0 = matrix_a_info->clone()->set_tensor_shape(compute_interleaved_shape(tensor_info0, mult_interleave4x4_height));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(matrix_a_info, &tensor_info_reshaped0);

        if(n != 0) /* Transpose */
        {
            TensorShape tensor_shape1{ matrix_b_info->tensor_shape() };
            tensor_shape1.set(0, n);
            tensor_shape1.set(1, k);

            const TensorInfo tensor_info1          = matrix_b_info->clone()->set_tensor_shape(tensor_shape1);
            const TensorInfo tensor_info_reshaped1 = matrix_b_info->clone()->set_tensor_shape(compute_transpose1xW_with_element_size_shape(tensor_info1, mult_transpose1xW_width));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(matrix_b_info, &tensor_info_reshaped1);
        }

        if(output->total_size() != 0)
        {
            if(n != 0)
            {
                ARM_COMPUTE_RETURN_ERROR_ON(tmp_output_info.dimension(0) != static_cast<size_t>(n));
            }
            ARM_COMPUTE_RETURN_ERROR_ON(tmp_output_info.dimension(1) != static_cast<size_t>(m));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(matrix_a_info, &tmp_output_info);
        }
    }
    else if(run_vector_matrix_multiplication)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixMultiplyKernel::validate(a, b, output, alpha, false, reshape_info));

        if(beta != 0 && c != nullptr)
        {
            // Validate matrix addition kernel
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(c, output);
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(c, output);
        }
    }
    else
    {
        if(run_interleave_transpose)
        {
            // Validate interleave kernel
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMInterleave4x4Kernel::validate(a, matrix_a_info));

            // Validate transpose kernel
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMTranspose1xWKernel::validate(b, matrix_b_info));
        }

        // Validate matrix multiply
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, &tmp_output_info, alpha, run_interleave_transpose, reshape_info));

        if(beta != 0 && c != nullptr)
        {
            // Validate matrix addition kernel
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(c, &tmp_output_info);
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(c, &tmp_output_info);
        }
    }

    return Status{};
}

void NEGEMM::run()
{
    prepare();

    if(_asm_glue.is_configured())
    {
        _memory_group.acquire();
        _asm_glue.run();
        _memory_group.release();
    }
    else
    {
        _memory_group.acquire();

        if(!_run_vector_matrix_multiplication)
        {
            // Run interleave kernel
            NEScheduler::get().schedule(&_interleave_kernel, Window::DimY);

            if(!_reshape_b_only_on_first_run)
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

void NEGEMM::prepare()
{
    if(!_is_prepared)
    {
        if(_asm_glue.is_configured())
        {
            ARM_COMPUTE_ERROR_ON(!_original_b->is_used());

            _asm_glue.prepare();
        }
        else if(_reshape_b_only_on_first_run && !_run_vector_matrix_multiplication && !_asm_glue.is_configured())
        {
            ARM_COMPUTE_ERROR_ON(!_original_b->is_used());

            _tmp_b.allocator()->allocate();
            NEScheduler::get().schedule(&_transpose_kernel, Window::DimY);
            _original_b->mark_as_unused();
        }

        _is_prepared = true;
    }
}
} // namespace arm_compute
