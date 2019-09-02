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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpAssemblyMatrixMultiplyCore.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

NEGEMMLowpAssemblyMatrixMultiplyCore::NEGEMMLowpAssemblyMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _asm_glue(memory_manager), _mm_kernel(nullptr), _mtx_a_reshape_kernel(nullptr), _mtx_b_reshape_kernel(nullptr), _tmp_a(), _tmp_b()
{
}

void NEGEMMLowpAssemblyMatrixMultiplyCore::configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::U8, DataType::S8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32, DataType::S32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
    ARM_COMPUTE_ERROR_ON_MSG((a)->info()->dimension(0) != (b)->info()->dimension(1), "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_ERROR_ON_MSG((a)->info()->dimension(1) != (output)->info()->dimension(1), "The output matrix must have the same number of rows as the matrix A");
    ARM_COMPUTE_ERROR_ON_MSG((b)->info()->dimension(0) != (output)->info()->dimension(0), "The output matrix must have the same number of columns as the matrix B");

    bool run_optimised = false;
    switch(a->info()->data_type())
    {
        case DataType::S8:
        case DataType::QASYMM8:
        case DataType::U8:
        {
            _asm_glue.configure(a, b, c, output, 1.f, 0.f, GEMMInfo(false, false, true));
            run_optimised = _asm_glue.is_configured();
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Datatype not supported");
            break;
        }
    }
    if(!run_optimised)
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

        // Allocate tensors
        _tmp_a.allocator()->allocate();
        _tmp_b.allocator()->allocate();
    }
}

void NEGEMMLowpAssemblyMatrixMultiplyCore::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);
    if(_mtx_a_reshape_kernel)
    {
        NEScheduler::get().schedule(_mtx_a_reshape_kernel.get(), Window::DimY);
    }

    if(_mtx_b_reshape_kernel)
    {
        NEScheduler::get().schedule(_mtx_b_reshape_kernel.get(), Window::DimY);
    }

    if(_asm_glue.is_configured())
    {
        _asm_glue.run();
    }
    else
    {
        NEScheduler::get().schedule(_mm_kernel.get(), Window::DimY);
    }
}
