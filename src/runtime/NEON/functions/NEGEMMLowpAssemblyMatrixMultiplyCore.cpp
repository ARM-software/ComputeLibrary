/* Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMAssemblyBaseKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMLowpAArch64A53Kernel.h"
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMLowpAArch64Kernel.h"
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
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_s16_12x8.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_s8_12x8.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_s8_4x4.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_u16_12x8.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_u8_4x4.hpp"
} // namespace arm_compute

using namespace arm_compute;

NEGEMMLowpAssemblyMatrixMultiplyCore::NEGEMMLowpAssemblyMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _mm_kernel(nullptr), _mtx_a_reshape_kernel(nullptr), _mtx_b_reshape_kernel(nullptr), _tmp_a(), _tmp_b(), _workspace()
{
}

void NEGEMMLowpAssemblyMatrixMultiplyCore::configure(const ITensor *a, const ITensor *b, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::U8, DataType::S8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32, DataType::S32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
    ARM_COMPUTE_ERROR_ON_MSG((a)->info()->dimension(0) != (b)->info()->dimension(1), "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_ERROR_ON_MSG((a)->info()->dimension(1) != (output)->info()->dimension(1), "The output matrix must have the same number of rows as the matrix A");
    ARM_COMPUTE_ERROR_ON_MSG((b)->info()->dimension(0) != (output)->info()->dimension(0), "The output matrix must have the same number of columns as the matrix B");

#ifdef __aarch64__
    const int            M                   = output->info()->tensor_shape().y();
    const int            N                   = output->info()->tensor_shape().x();
    const int            K                   = a->info()->tensor_shape().x();
    constexpr size_t     workspace_alignment = 4096;
    const struct CPUInfo ci                  = NEScheduler::get().cpu_info();
#endif /* __aarch64__ */

#ifdef ARM_COMPUTE_AARCH64_V8_2
    if(ci.CPU == CPUTarget::A75_DOT || ci.CPU == CPUTarget::A55_DOT)
    {
        // Configure matrix multiply kernel
        GemmInterleaved<gemm_s8_12x8, int8_t, int32_t> gemm(&ci, M, N, K, false, false);
        _workspace.allocator()->init(TensorInfo(TensorShape{ (gemm.get_working_size() + workspace_alignment - 1) * NEScheduler::get().num_threads() }, 1, DataType::U8));
        _memory_group.manage(&_workspace);

        // Configure matrix multiplication kernel
        auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpAArch64V8P4Kernel>();
        k->configure(a, b, output, &_workspace, 1.f, 1.f);
        _mm_kernel = std::move(k);
        _workspace.allocator()->allocate();
    }
    else
#elif defined(ARM_COMPUTE_AARCH64_V8A)
    if(ci.CPU == CPUTarget::A53)
    {
        switch(a->info()->data_type())
        {
            case DataType::S8:
            {
                // Configure matrix multiply kernel
                GemmInterleaved<gemm_s16_12x8, int8_t, int32_t> gemm(&ci, M, N, K, false, false);
                _workspace.allocator()->init(TensorInfo(TensorShape{ (gemm.get_working_size() + workspace_alignment - 1) * NEScheduler::get().num_threads() }, 1, DataType::U8));
            }
            break;
            case DataType::U8:
            {
                // Configure matrix multiply kernel
                GemmInterleaved<gemm_u16_12x8, uint8_t, uint32_t> gemm(&ci, M, N, K, false, false);
                _workspace.allocator()->init(TensorInfo(TensorShape{ (gemm.get_working_size() + workspace_alignment - 1) * NEScheduler::get().num_threads() }, 1, DataType::U8));
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Datatype not supported");
        }

        _memory_group.manage(&_workspace);
        // Configure matrix multiplication kernel
        auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpAArch64A53Kernel>();
        k->configure(a, b, output, &_workspace, 1.f, 1.f);
        _mm_kernel = std::move(k);
        _workspace.allocator()->allocate();
    }
    else if(1) // Generic v8a kernel
    {
        switch(a->info()->data_type())
        {
            case DataType::S8:
            {
                // Configure matrix multiply kernel
                GemmInterleaved<gemm_s8_4x4, int8_t, int32_t> gemm(&ci, M, N, K, false, false);
                _workspace.allocator()->init(TensorInfo(TensorShape{ (gemm.get_working_size() + workspace_alignment - 1) * NEScheduler::get().num_threads() }, 1, DataType::U8));
            }
            break;
            case DataType::U8:
            {
                // Configure matrix multiply kernel
                GemmInterleaved<gemm_u8_4x4, uint8_t, uint32_t> gemm(&ci, M, N, K, false, false);
                _workspace.allocator()->init(TensorInfo(TensorShape{ (gemm.get_working_size() + workspace_alignment - 1) * NEScheduler::get().num_threads() }, 1, DataType::U8));
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Datatype not supported");
        }
        _memory_group.manage(&_workspace);
        // Configure matrix multiplication kernel
        auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpAArch64Kernel>();
        k->configure(a, b, output, &_workspace, 1.f, 1.f);
        _mm_kernel = std::move(k);
        _workspace.allocator()->allocate();
    }
    else
#endif /* ARM_COMPUTE_AARCH64_V8_2 */
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
    _memory_group.acquire();
    if(_mtx_a_reshape_kernel)
    {
        NEScheduler::get().schedule(_mtx_a_reshape_kernel.get(), Window::DimY);
    }

    if(_mtx_b_reshape_kernel)
    {
        NEScheduler::get().schedule(_mtx_b_reshape_kernel.get(), Window::DimY);
    }

    NEScheduler::get().schedule(_mm_kernel.get(), Window::DimY);

    _memory_group.release();
}
