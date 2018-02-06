/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMAArch64NativeKernel.h"

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
#include "arm_compute/core/NEON/kernels/winograd/gemm.hpp"
} // namespace arm_compute

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace arm_compute
{
void NEGEMMAArch64NativeKernel::internal_configure(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha, float beta, bool is_transposed_0,
                                                   bool is_transposed_1)
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
    Window win = calculate_max_window(*output->info(), Steps(16U, 4U));

    const int input0_access_end_x = ceil_to_multiple(input0->info()->tensor_shape().x(), 4);
    const int input0_access_end_y = ceil_to_multiple(input0->info()->tensor_shape().y(), 4);
    const int input1_access_end_x = ceil_to_multiple(input1->info()->tensor_shape().x(), 16);

    AccessWindowStatic    input0_access(input0->info(), 0, 0, input0_access_end_x, input0_access_end_y);
    AccessWindowStatic    input1_access(input1->info(), 0, 0, input1_access_end_x, input1->info()->tensor_shape().y());
    AccessWindowRectangle output_access(output->info(), 0, 0, 16, 4);
    update_window_and_padding(win, input0_access, input1_access, output_access);

    INEKernel::configure(win);
}

void NEGEMMAArch64NativeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_UNUSED(info);

    const auto in1_ptr = reinterpret_cast<const float *>(_input1->buffer());

    // Calculate row strides for each matrix
    const int lda = _input0->info()->strides_in_bytes().y() / sizeof(float);
    const int ldb = _input1->info()->strides_in_bytes().y() / sizeof(float);
    const int ldc = _output->info()->strides_in_bytes().y() / sizeof(float);

    // Calculate matrix sizes
    const int M = std::min(_input0->info()->tensor_shape().y(), static_cast<size_t>(window.y().end())) - window.y().start();
    const int K = _input0->info()->tensor_shape().x();
    const int N = _input1->info()->tensor_shape().x();

    // Create window (Only iterate over batches)
    Window win(window);
    win.set(0, Window::Dimension(0, 1, 1));
    win.set(1, Window::Dimension(0, 1, 1));

    // Create Iterators
    Iterator in0(_input0, window);
    Iterator out(_output, window);

    // Execute GEMM
    execute_window_loop(win, [&](const Coordinates & id)
    {
        BlockedGemm<4, 16, float, float>(reinterpret_cast<const float *>(in0.ptr()),
                                         reinterpret_cast<const float *>(in1_ptr),
                                         reinterpret_cast<float *>(out.ptr()),
                                         M, K, N,
                                         lda, ldb, ldc);
    },
    in0, out);
}
} // namespace arm_compute
