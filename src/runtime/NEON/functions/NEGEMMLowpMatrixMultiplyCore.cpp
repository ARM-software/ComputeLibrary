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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEGEMMAssemblyBaseKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMLowpAArch64V8P4Kernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
#include "arm_compute/core/NEON/kernels/assembly/gemm_interleaved.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_u8_12x8.hpp"
} // namespace arm_compute

using namespace arm_compute;

NEGEMMLowpMatrixMultiplyCore::NEGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _mm_kernel(nullptr), _mtx_a_reshape_kernel(nullptr), _mtx_b_reshape_kernel(nullptr), _mtx_a_reduction_kernel(), _mtx_b_reduction_kernel(),
      _offset_contribution_kernel(), _vector_sum_col(), _vector_sum_row(), _tmp_a(), _tmp_b(), _workspace(), _a_offset(0), _b_offset(0), _run_vector_matrix_multiplication(false), _dot_product_path(false)
{
}

void NEGEMMLowpMatrixMultiplyCore::configure(const ITensor *a, const ITensor *b, ITensor *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
    ARM_COMPUTE_UNUSED(gemm_info);
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMLowpMatrixMultiplyCore::validate(a->info(), b->info(), output->info(), gemm_info));

    _a_offset                         = a->info()->quantization_info().offset;
    _b_offset                         = b->info()->quantization_info().offset;
    _run_vector_matrix_multiplication = a->info()->dimension(1) < 2;

#ifdef ARM_COMPUTE_AARCH64_V8_2
    // Check for DOT product instruction
    const struct CPUInfo ci              = NEScheduler::get().cpu_info();
    const int            cpu_has_dotprod = static_cast<int>(ci.CPU) & static_cast<int>(CPUTarget::DOT);

    if(cpu_has_dotprod != 0)
    {
        _dot_product_path = true;

        // Configure matrix multiply kernel
        struct CPUInfo ci = NEScheduler::get().cpu_info();
        const int      M  = output->info()->tensor_shape().y();
        const int      N  = output->info()->tensor_shape().x();
        const int      K  = a->info()->tensor_shape().x();

        const size_t     workbench_size = GemmInterleaved<gemm_u8_12x8, gemm_u8_12x8::operand_type, gemm_u8_12x8::result_type>(&ci, M, N, K, false, false).get_working_size();
        constexpr size_t alignment      = 4096;
        _workspace.allocator()->init(TensorInfo(TensorShape{ (workbench_size + alignment - 1) * NEScheduler::get().num_threads() }, 1, DataType::U8));
        _memory_group.manage(&_workspace);

        // Configure matrix multiplication kernel
        auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpAArch64V8P4Kernel>();
        k->configure(a, b, output, &_workspace, 1.f, 1.f, false, false);
        _mm_kernel = std::move(k);
    }
    else
#endif /* ARM_COMPUTE_AARCH64_V8_2 */
    {
        if(_run_vector_matrix_multiplication)
        {
            // Configure matrix multiply kernel
            {
                auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpMatrixMultiplyKernel>();
                k->configure(a, b, output);
                _mm_kernel = std::move(k);
            }
        }
        else
        {
            // The interleaved output matrix will have the following shape: [ a_height * 4, ceil(a_width / 4.0f) ]
            TensorShape shape_tmp_a = a->info()->tensor_shape();
            shape_tmp_a.set(0, a->info()->dimension(0) * 4);
            shape_tmp_a.set(1, std::ceil(a->info()->dimension(1) / 4.f));

            // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
            TensorShape shape_tmp_b = b->info()->tensor_shape();
            shape_tmp_b.set(0, b->info()->dimension(1) * 16);
            shape_tmp_b.set(1, std::ceil(b->info()->dimension(0) / 16.f));

            TensorInfo info_a(shape_tmp_a, 1, a->info()->data_type());
            TensorInfo info_b(shape_tmp_b, 1, b->info()->data_type());
            _tmp_a.allocator()->init(info_a);
            _tmp_b.allocator()->init(info_b);
            _memory_group.manage(&_tmp_a);
            _memory_group.manage(&_tmp_b);

            // Configure interleave kernel
            {
                auto k = arm_compute::support::cpp14::make_unique<NEGEMMInterleave4x4Kernel>();
                k->configure(a, &_tmp_a);
                _mtx_a_reshape_kernel = std::move(k);
            }

            // Configure transpose kernel
            {
                auto k = arm_compute::support::cpp14::make_unique<NEGEMMTranspose1xWKernel>();
                k->configure(b, &_tmp_b);
                _mtx_b_reshape_kernel = std::move(k);
            }

            // Configure matrix multiply kernel
            {
                auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpMatrixMultiplyKernel>();
                k->configure(&_tmp_a, &_tmp_b, output);
                _mm_kernel = std::move(k);
            }
        }
    }

    // Initialize matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0)
    {
        TensorShape shape_vector_sum_col = b->info()->tensor_shape();
        if(b->info()->num_dimensions() > 1)
        {
            shape_vector_sum_col.remove_dimension(1);
        }
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
        if(a->info()->num_dimensions() > 1)
        {
            shape_vector_sum_row.remove_dimension(1);
        }
        TensorInfo info_vector_sum_row(shape_vector_sum_row, 1, DataType::S32);
        _vector_sum_row.allocator()->init(info_vector_sum_row);
        _memory_group.manage(&_vector_sum_row);

        // Configure matrix A reduction kernel
        _mtx_a_reduction_kernel.configure(a, &_vector_sum_row, a->info()->dimension(0), false);
    }

    // Configure offset contribution kernel
    _offset_contribution_kernel.configure(output, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, a->info()->dimension(0), _a_offset, _b_offset);

    // Allocate tensors
    if(!_dot_product_path && !_run_vector_matrix_multiplication)
    {
        _tmp_a.allocator()->allocate();
        _tmp_b.allocator()->allocate();
    }
    else
    {
        _workspace.allocator()->allocate();
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

Status NEGEMMLowpMatrixMultiplyCore::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *output, const GEMMInfo &gemm_info)
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
    ARM_COMPUTE_UNUSED(gemm_info);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    int32_t a_offset                         = a->quantization_info().offset;
    int32_t b_offset                         = b->quantization_info().offset;
    bool    run_vector_matrix_multiplication = a->dimension(1) < 2;

#ifdef ARM_COMPUTE_AARCH64_V8_2
    // Check for DOT product instruction
    const struct CPUInfo ci              = NEScheduler::get().cpu_info();
    const int            cpu_has_dotprod = static_cast<int>(ci.CPU) & static_cast<int>(CPUTarget::DOT);

    if(cpu_has_dotprod != 0)
    {
        // Validate matrix multiply kernel
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpAArch64V8P4Kernel::validate(a, b, output));
    }
    else
#endif /* ARM_COMPUTE_AARCH64_V8_2 */
    {
        if(!run_vector_matrix_multiplication)
        {
            // The interleaved output matrix will have the following shape: [ a_height * 4, ceil(a_width / 4.0f) ]
            TensorShape shape_tmp_a = a->tensor_shape();
            shape_tmp_a.set(0, a->dimension(0) * 4);
            shape_tmp_a.set(1, std::ceil(a->dimension(1) / 4.f));

            // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
            TensorShape shape_tmp_b = b->tensor_shape();
            shape_tmp_b.set(0, b->dimension(1) * 16);
            shape_tmp_b.set(1, std::ceil(b->dimension(0) / 16.f));

            TensorInfo info_a(shape_tmp_a, 1, a->data_type());
            TensorInfo info_b(shape_tmp_b, 1, b->data_type());

            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMInterleave4x4Kernel::validate(a, &info_a));
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMTranspose1xWKernel::validate(b, &info_b));
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixMultiplyKernel::validate(&info_a, &info_b, output));
        }
        else
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixMultiplyKernel::validate(a, b, output));
        }
    }

    TensorInfo info_vector_sum_col, info_vector_sum_row;

    // Validate matrix B reduction kernel only if _a_offset is not equal to 0
    if(a_offset != 0)
    {
        TensorShape shape_vector_sum_col = b->tensor_shape();
        shape_vector_sum_col.remove_dimension(1);
        info_vector_sum_col = TensorInfo(shape_vector_sum_col, 1, DataType::S32);

        // Configure Matrix B reduction kernel
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixBReductionKernel::validate(b, &info_vector_sum_col, a->dimension(0), false));
    }

    // Validate Matrix A reduction kernel only if _b_offset is not equal to 0
    if(b_offset != 0)
    {
        TensorShape shape_vector_sum_row = a->tensor_shape();
        shape_vector_sum_row.set(Window::DimX, a->dimension(1));
        shape_vector_sum_row.remove_dimension(1);
        info_vector_sum_row = TensorInfo(shape_vector_sum_row, 1, DataType::S32);

        // Configure matrix A reduction kernel
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(a, &info_vector_sum_row, a->dimension(0), false));
    }

    // Validate offset contribution kernel
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpOffsetContributionKernel::validate(output,
                                                                             a_offset == 0 ? nullptr : &info_vector_sum_col,
                                                                             b_offset == 0 ? nullptr : &info_vector_sum_row,
                                                                             a_offset, b_offset));

    return Status{};
}

void NEGEMMLowpMatrixMultiplyCore::run()
{
    _memory_group.acquire();

    // Do not reshape if we run the vector-by-matrix case and we do not have the optimized gemm with dot product instruction
    if(!_run_vector_matrix_multiplication && !_dot_product_path)
    {
        if(_mtx_a_reshape_kernel)
        {
            NEScheduler::get().schedule(_mtx_a_reshape_kernel.get(), Window::DimY);
        }

        if(_mtx_b_reshape_kernel)
        {
            NEScheduler::get().schedule(_mtx_b_reshape_kernel.get(), Window::DimY);
        }
    }

    NEScheduler::get().schedule(_mm_kernel.get(), Window::DimY);

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

    // Run offset contribution kernel
    NEScheduler::get().schedule(&_offset_contribution_kernel, Window::DimY);

    _memory_group.release();
}
