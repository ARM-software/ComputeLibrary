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

    bool run_optimised = c == nullptr && bool(NEGEMMAssemblyDispatch::validate(a->info(), b->info(), c != nullptr ? c->info() : nullptr, d->info(), alpha, beta, gemm_info));

    if(run_optimised)
    {
        if(MEMInfo::get_policy() == MemoryPolicy::MINIMIZE)
        {
            GEMMInfo gemm_info_ntb = gemm_info;
            gemm_info_ntb.set_pretranpose_B(false);
            _asm_glue.configure(a, b, c, d, alpha, beta, gemm_info_ntb);
        }
        else
        {
            _asm_glue.configure(a, b, c, d, alpha, beta, gemm_info);
        }
        ARM_COMPUTE_ERROR_ON(!_asm_glue.is_configured());
    }
    else
    {
        if(_run_vector_matrix_multiplication)
        {
            // Configure the matrix multiply kernel
            _mm_kernel.configure(a, b, d, alpha, false);
        }
        else
        {
            TensorShape shape_tmp_a = a->info()->tensor_shape();
            TensorShape shape_tmp_b = b->info()->tensor_shape();

            shape_tmp_a.set(0, a->info()->dimension(0) * 4);
            shape_tmp_a.set(1, std::ceil(a->info()->dimension(1) / 4.0f));

            const unsigned int transpose_w = 16 / data_size_from_type(b->info()->data_type());
            shape_tmp_b.set(0, b->info()->dimension(1) * transpose_w);
            shape_tmp_b.set(1, std::ceil(b->info()->dimension(0) / static_cast<float>(transpose_w)));

            TensorInfo info_a = a->info()->clone()->set_tensor_shape(shape_tmp_a).set_is_resizable(true);
            TensorInfo info_b = b->info()->clone()->set_tensor_shape(shape_tmp_b).set_is_resizable(true);

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
        }

        // Configure matrix addition kernel
        if(beta != 0 && c != nullptr)
        {
            _ma_kernel.configure(c, d, beta);
            _run_addition = true;
        }
    }
}

Status NEGEMM::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(a);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->dimension(0) != b->dimension(1), "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    if(c != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.depth_output_gemm3d() != 0);
        ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.reinterpret_input_as_3d());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, c);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->dimension(1) != c->dimension(1), "The C matrix must have the same number of rows as the matrix A");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(b->dimension(0) != c->dimension(0), "The C matrix must have the same number of columns as the matrix B");
    }

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(b->dimension(0) != output->dimension(0));
        if(gemm_info.depth_output_gemm3d() != 0)
        {
            if(gemm_info.reinterpret_input_as_3d())
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

    // Check if we need to run the optimized assembly kernel
    const bool run_optimised = c == nullptr && bool(NEGEMMAssemblyDispatch::validate(a, b, c, output, alpha, beta, gemm_info));

    if(!run_optimised)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.reinterpret_input_as_3d(), "NEGEMM cannot reinterpret the input tensor as 3D");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.depth_output_gemm3d() != 0, "NEGEMM cannot reinterpret the output tensor as 3D");

        // Check if the first input tensor is a vector.
        const bool run_vector_matrix_multiplication = a->dimension(1) < 2;
        // Check if we need to reshape the matrix A and matrix B
        const bool run_interleave_transpose = !run_vector_matrix_multiplication && !(gemm_info.reshape_b_only_on_first_run());

        // Arguments used by GEMMReshapeInfo
        // If we pass the matrix A and matrix B reshaped to NEGEMMMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to NEGEMMReshapeInfo
        // in order to know how the matrices have been reshaped
        const int m                         = a->dimension(1);
        const int n                         = b->dimension(0);
        const int k                         = a->dimension(0);
        int       mult_transpose1xW_width   = 1;
        int       mult_interleave4x4_height = 1;

        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, gemm_info.depth_output_gemm3d());

        const ITensorInfo *matrix_a_info = a;
        const ITensorInfo *matrix_b_info = b;

        TensorInfo tmp_a_info{};
        TensorInfo tmp_b_info{};
        TensorInfo tmp_output_info = *output->clone();

        if(run_interleave_transpose)
        {
            matrix_a_info = &tmp_a_info;
            matrix_b_info = &tmp_b_info;

            // Validate interleave kernel
            auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_interleaved_shape(*a, mult_interleave4x4_height, gemm_info.reinterpret_input_as_3d())));
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMInterleave4x4Kernel::validate(a, &tmp_a_info));

            // Validate transpose kernel
            auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_transpose1xW_with_element_size_shape(*b, mult_transpose1xW_width)));
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMTranspose1xWKernel::validate(b, &tmp_b_info));
        }

        // Validate matrix multiply
        auto_init_if_empty(tmp_output_info, matrix_a_info->clone()->set_tensor_shape(compute_mm_shape(*matrix_a_info, *matrix_b_info, run_interleave_transpose, reshape_info)));
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, &tmp_output_info, alpha, run_interleave_transpose, reshape_info));
    }

    // Validate matrix addition kernel
    if(beta != 0 && c != nullptr)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixAdditionKernel::validate(c, output, beta));
    }

    return Status{};
}

void NEGEMM::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_asm_glue.is_configured())
    {
        _asm_glue.run();
    }
    else
    {
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
