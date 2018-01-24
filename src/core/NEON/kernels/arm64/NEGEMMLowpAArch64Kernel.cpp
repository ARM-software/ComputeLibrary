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
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMLowpAArch64Kernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
#include "arm_compute/core/NEON/kernels/assembly/gemm_interleaved.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_s8_4x4.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_u8_4x4.hpp"
} // namespace arm_compute

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

// Enable only if compiled for AArch64-V8A targets
#ifdef ARM_COMPUTE_AARCH64_V8A

namespace arm_compute
{
NEGEMMLowpAArch64Kernel::NEGEMMLowpAArch64Kernel()
    : _func(nullptr)
{
}

void gemm_interleaved_s8(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha, float beta, bool is_transposed_0, bool is_transposed_1, const Window &window,
                         const ThreadInfo &info)
{
    const int lda = input0->info()->strides_in_bytes().y();
    const int ldb = input1->info()->strides_in_bytes().y();
    const int ldc = output->info()->strides_in_bytes().y() / sizeof(int32_t);

    const auto in1_ptr = reinterpret_cast<const int8_t *>(input1->buffer());

    const int M = std::min(output->info()->tensor_shape().y(), static_cast<size_t>(window.y().end())) - window.y().start();
    const int N = output->info()->tensor_shape().x();
    const int K = input0->info()->tensor_shape().x();

    // Only iterate over batches
    Window win(window);
    win.set(0, Window::Dimension(0, 1, 1));
    win.set(1, Window::Dimension(0, 1, 1));

    Iterator in0(input0, window);
    Iterator out(output, window);

    GemmInterleaved<gemm_s8_4x4, int8_t, int32_t> gemm(&info.cpu_info, M, N, K, is_transposed_0, is_transposed_1);

    constexpr size_t alignment      = 4096;
    const size_t     offset         = (gemm.get_working_size() + alignment - 1) * info.thread_id;
    void            *_workspace     = workspace->buffer() + offset;
    size_t           workspace_size = workspace->info()->total_size();

    if(support::cpp11::align(alignment, gemm.get_working_size(), _workspace, workspace_size) == nullptr)
    {
        ARM_COMPUTE_ERROR("Not enough space to align buffer!");
    }

    execute_window_loop(win, [&](const Coordinates & id)
    {
        gemm.execute(reinterpret_cast<const int8_t *>(in0.ptr()), lda,
                     reinterpret_cast<const int8_t *>(in1_ptr), ldb,
                     reinterpret_cast<int32_t *>(out.ptr()), ldc,
                     alpha, beta, _workspace);
    },
    in0, out);
}

void gemm_interleaved_u8(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha, float beta, bool is_transposed_0, bool is_transposed_1, const Window &window,
                         const ThreadInfo &info)
{
    const int lda = input0->info()->strides_in_bytes().y();
    const int ldb = input1->info()->strides_in_bytes().y();
    const int ldc = output->info()->strides_in_bytes().y() / sizeof(uint32_t);

    const auto in1_ptr = reinterpret_cast<const uint8_t *>(input1->buffer());

    const int M = std::min(output->info()->tensor_shape().y(), static_cast<size_t>(window.y().end())) - window.y().start();
    const int N = output->info()->tensor_shape().x();
    const int K = input0->info()->tensor_shape().x();

    // Only iterate over batches
    Window win(window);
    win.set(0, Window::Dimension(0, 1, 1));
    win.set(1, Window::Dimension(0, 1, 1));

    Iterator in0(input0, window);
    Iterator out(output, window);

    GemmInterleaved<gemm_u8_4x4, uint8_t, uint32_t> gemm(&info.cpu_info, M, N, K, is_transposed_0, is_transposed_1);

    constexpr size_t alignment      = 4096;
    const size_t     offset         = (gemm.get_working_size() + alignment - 1) * info.thread_id;
    void            *_workspace     = workspace->buffer() + offset;
    size_t           workspace_size = workspace->info()->total_size();

    if(support::cpp11::align(alignment, gemm.get_working_size(), _workspace, workspace_size) == nullptr)
    {
        ARM_COMPUTE_ERROR("Not enough space to align buffer!");
    }

    execute_window_loop(win, [&](const Coordinates & id)
    {
        gemm.execute(reinterpret_cast<const uint8_t *>(in0.ptr()), lda,
                     reinterpret_cast<const uint8_t *>(in1_ptr), ldb,
                     reinterpret_cast<uint32_t *>(out.ptr()), ldc,
                     alpha, beta, _workspace);
    },
    in0, out);
}

void NEGEMMLowpAArch64Kernel::internal_configure(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha, float beta, bool is_transposed_0,
                                                 bool is_transposed_1)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::S8, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32, DataType::U32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);

    _input0          = input0;
    _input1          = input1;
    _output          = output;
    _workspace       = workspace;
    _alpha           = alpha;
    _beta            = beta;
    _is_transposed_0 = is_transposed_0;
    _is_transposed_1 = is_transposed_1;

    switch(input0->info()->data_type())
    {
        case DataType::S8:
            _func = &gemm_interleaved_s8;
            break;
        case DataType::U8:
            _func = &gemm_interleaved_u8;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    // Configure kernel window
    Window win = calculate_max_window(*output->info());

    AccessWindowRectangle output_access(output->info(), 0, 0, 4, 4);

    const int input0_access_end = ceil_to_multiple(input0->info()->tensor_shape().x(), 4);
    const int input1_access_end = ceil_to_multiple(input1->info()->tensor_shape().x(), 4);

    update_window_and_padding(win,
                              AccessWindowStatic(input0->info(), 0, 0, input0_access_end, input0->info()->tensor_shape().y()),
                              AccessWindowStatic(input1->info(), 0, 0, input1_access_end, input1->info()->tensor_shape().y()),
                              output_access);

    INEKernel::configure(win);
}

void NEGEMMLowpAArch64Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input0, _input1, _output, _workspace, _alpha, _beta, _is_transposed_0, _is_transposed_1, window, info);
}
} // namespace arm_compute
#endif /* ARM_COMPUTE_AARCH64_V8A */
