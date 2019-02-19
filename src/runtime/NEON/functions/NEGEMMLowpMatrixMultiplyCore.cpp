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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

NEGEMMLowpMatrixMultiplyCore::NEGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _asm_glue(memory_manager), _mm_kernel(nullptr), _mtx_a_reshape_kernel(nullptr), _mtx_b_reshape_kernel(nullptr), _mtx_a_reduction_kernel(), _mtx_b_reduction_kernel(),
      _offset_contribution_kernel(), _vector_sum_col(), _vector_sum_row(), _tmp_a(), _tmp_b(), _original_b(nullptr), _a_offset(0), _b_offset(0), _run_vector_matrix_multiplication(false),
      _dot_product_path(false), _reshape_b_only_on_first_run(false), _is_prepared(false)
{
}

void NEGEMMLowpMatrixMultiplyCore::configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMLowpMatrixMultiplyCore::validate(a->info(), b->info(), c != nullptr ? c->info() : nullptr, output->info(), gemm_info));

    // Clear state
    _mtx_a_reshape_kernel = nullptr;
    _mtx_b_reshape_kernel = nullptr;

    // Set internal variables
    _a_offset                         = a->info()->quantization_info().offset;
    _b_offset                         = b->info()->quantization_info().offset;
    _run_vector_matrix_multiplication = a->info()->dimension(1) < 2;
    _reshape_b_only_on_first_run      = gemm_info.reshape_b_only_on_first_run();
    _is_prepared                      = false;
    _original_b                       = b;

#ifdef __aarch64__
    switch(a->info()->data_type())
    {
        case DataType::QASYMM8:
        case DataType::U8:
        case DataType::S8:
        {
            _asm_glue.configure(a, b, output, 1.f, 0.f, _reshape_b_only_on_first_run);
            _dot_product_path = _asm_glue.is_configured();
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Datatype not supported");
            break;
        }
    }
#endif /* __aarch64__ */
    if(!_dot_product_path)
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
            TensorInfo info_a = a->info()->clone()->set_tensor_shape(compute_interleaved_shape(*a->info())).set_is_resizable(true);
            // The transpose1xW output matrix will have the following shape: [ b_height * 16, ceil(b_width / 16.0f) ]
            TensorInfo info_b = b->info()->clone()->set_tensor_shape(compute_transpose1xW_shape(*b->info())).set_is_resizable(true);
            _tmp_a.allocator()->init(info_a);
            _tmp_b.allocator()->init(info_b);
            _memory_group.manage(&_tmp_a);
            if(!_reshape_b_only_on_first_run)
            {
                _memory_group.manage(&_tmp_b);
            }

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
        TensorInfo info_vector_sum_col(compute_reductionA_shape(*b->info()), 1, DataType::S32);

        _vector_sum_col.allocator()->init(info_vector_sum_col);
        if(!_reshape_b_only_on_first_run)
        {
            _memory_group.manage(&_vector_sum_col);
        }

        // Configure Matrix B reduction kernel
        _mtx_b_reduction_kernel.configure(b, &_vector_sum_col, a->info()->dimension(0), false);
    }

    // Initialize Matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        TensorInfo info_vector_sum_row(compute_reductionB_shape(*a->info()), 1, DataType::S32);

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

Status NEGEMMLowpMatrixMultiplyCore::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(c != nullptr, "Bias addition not supported in NEGEMMLowpMatrixMultiplyCore");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((a)->dimension(0) != (b)->dimension(1),
                                    "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

    int32_t    a_offset                    = a->quantization_info().offset;
    int32_t    b_offset                    = b->quantization_info().offset;
    const bool reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();

    // Check if we need to run the optimized assembly kernel
    const bool run_optimised = bool(NEGEMMAssemblyDispatch::validate(a, b, output, 1.f, 0.f, reshape_b_only_on_first_run));

    if(run_optimised)
    {
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
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.reinterpret_input_as_3d(), "NEGEMM cannot reinterpret the input tensor as 3D");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.depth_output_gemm3d() != 0, "NEGEMM cannot reinterpret the output tensor as 3D");

        const bool run_vector_matrix_multiplication = a->dimension(1) < 2;
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

            TensorInfo info_a = a->clone()->set_tensor_shape(shape_tmp_a).set_is_resizable(true);
            TensorInfo info_b = b->clone()->set_tensor_shape(shape_tmp_b).set_is_resizable(true);

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
        info_vector_sum_col = TensorInfo(compute_reductionA_shape(*b), 1, DataType::S32);

        // Configure Matrix B reduction kernel
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixBReductionKernel::validate(b, &info_vector_sum_col, a->dimension(0), false));
    }

    // Validate Matrix A reduction kernel only if _b_offset is not equal to 0
    if(b_offset != 0)
    {
        info_vector_sum_row = TensorInfo(compute_reductionB_shape(*a), 1, DataType::S32);

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
    prepare();

    _memory_group.acquire();

    // Reshape inputs
    if(_mtx_a_reshape_kernel)
    {
        NEScheduler::get().schedule(_mtx_a_reshape_kernel.get(), Window::DimY);
    }
    if(_mtx_b_reshape_kernel && !_reshape_b_only_on_first_run)
    {
        NEScheduler::get().schedule(_mtx_b_reshape_kernel.get(), Window::DimY);
    }

    // Run GEMM
    if(_asm_glue.is_configured())
    {
        _asm_glue.run();
    }
    else
    {
        NEScheduler::get().schedule(_mm_kernel.get(), Window::DimY);
    }

    // Run matrix A reduction kernel only if _b_offset is not equal to 0
    if(_b_offset != 0)
    {
        NEScheduler::get().schedule(&_mtx_a_reduction_kernel, Window::DimX);
    }

    // Run matrix B reduction kernel only if _a_offset is not equal to 0
    if(_a_offset != 0 && !_reshape_b_only_on_first_run)
    {
        NEScheduler::get().schedule(&_mtx_b_reduction_kernel, Window::DimX);
    }

    // Run offset contribution kernel
    NEScheduler::get().schedule(&_offset_contribution_kernel, Window::DimY);

    _memory_group.release();
}

void NEGEMMLowpMatrixMultiplyCore::prepare()
{
    if(!_is_prepared)
    {
        // Run assembly reshape
        if(_asm_glue.is_configured() && _reshape_b_only_on_first_run)
        {
            ARM_COMPUTE_ERROR_ON(!_original_b->is_used());

            _asm_glue.prepare();
            _original_b->mark_as_unused();
        }
        // Run non-assembly reshape
        else if(_mtx_b_reshape_kernel && _reshape_b_only_on_first_run)
        {
            ARM_COMPUTE_ERROR_ON(!_original_b->is_used());

            // Run reshape kernel and mark original weights tensor as unused
            _tmp_b.allocator()->allocate();
            NEScheduler::get().schedule(_mtx_b_reshape_kernel.get(), Window::DimY);
            _original_b->mark_as_unused();
        }

        // Run matrix B reduction kernel only if _a_offset is not equal to 0
        if(_a_offset != 0 && _reshape_b_only_on_first_run)
        {
            _vector_sum_col.allocator()->allocate();
            NEScheduler::get().schedule(&_mtx_b_reduction_kernel, Window::DimX);
        }

        _is_prepared = true;
    }
}
