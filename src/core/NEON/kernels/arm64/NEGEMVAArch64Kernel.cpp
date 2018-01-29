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
#include "arm_compute/core/NEON/kernels/arm64/NEGEMVAArch64Kernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/AccessWindowTranspose.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#pragma GCC diagnostic ignored "-Weffc++"
#include "arm_compute/core/NEON/kernels/assembly/gemv_transposed.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_sgemv_trans.hpp"
#pragma GCC diagnostic pop
} // namespace arm_compute

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace arm_compute
{
void NEGEMVAArch64Kernel::internal_configure(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha, float beta, bool is_transposed_0, bool is_transposed_1)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input0, input1, output);

    _input0          = input0;
    _input1          = input1;
    _output          = output;
    _workspace       = workspace;
    _alpha           = alpha;
    _beta            = beta;
    _is_transposed_0 = is_transposed_0;
    _is_transposed_1 = is_transposed_1;

    // Configure kernel window
    Window win = calculate_max_window(*output->info());

    AccessWindowRectangle output_access(output->info(), 0, 0, 12, 8);

    const int input0_access_end = ceil_to_multiple(input0->info()->tensor_shape().x(), 8);
    const int input1_access_end = ceil_to_multiple(input1->info()->tensor_shape().x(), 12);

    update_window_and_padding(win,
                              AccessWindowStatic(input0->info(), 0, 0, input0_access_end, input0->info()->tensor_shape().y()),
                              AccessWindowStatic(input1->info(), 0, 0, input1_access_end, input1->info()->tensor_shape().y()),
                              output_access);

    INEKernel::configure(win);
}

void NEGEMVAArch64Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const int lda = _input0->info()->strides_in_bytes().y() / sizeof(sgemv_trans::operand_type);
    const int ldb = _input1->info()->strides_in_bytes().y() / sizeof(sgemv_trans::operand_type);
    const int ldc = _output->info()->strides_in_bytes().y() / sizeof(sgemv_trans::result_type);

    const auto in1_ptr = reinterpret_cast<const sgemv_trans::operand_type *>(_input1->buffer());

    const int N = _output->info()->tensor_shape().x();
    const int K = _input0->info()->tensor_shape().x();

    // Only iterate over batches
    Window win(window);
    win.set(0, Window::Dimension(0, 1, 1));
    win.set(1, Window::Dimension(0, 1, 1));

    Iterator in0(_input0, window);
    Iterator out(_output, window);

    GemvTransposed<sgemv_trans, sgemv_trans::operand_type, sgemv_trans::result_type> gemm(&info.cpu_info, N, K);
    constexpr size_t alignment      = 4096;
    const size_t     offset         = (gemm.get_working_size() + alignment - 1) * info.thread_id;
    void            *workspace      = _workspace->buffer() + offset;
    size_t           workspace_size = _workspace->info()->total_size();

    if(support::cpp11::align(alignment, gemm.get_working_size(), workspace, workspace_size) == nullptr)
    {
        ARM_COMPUTE_ERROR("Not enough space to align buffer!");
    }

    execute_window_loop(win, [&](const Coordinates & id)
    {
        gemm.execute(reinterpret_cast<const sgemv_trans::operand_type *>(in0.ptr()), lda,
                     reinterpret_cast<const sgemv_trans::operand_type *>(in1_ptr), ldb,
                     reinterpret_cast<sgemv_trans::result_type *>(out.ptr()), ldc,
                     _alpha, _beta, workspace);
    },
    in0, out);
}
} // namespace arm_compute
