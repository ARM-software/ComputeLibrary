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
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMLowpAArch64V8P4Kernel.h"

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
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_s8_12x8.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_gemm_u8_12x8.hpp"
} // namespace arm_compute

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

// Enable only if compiled for AArch64-V8.2-A targets
#ifdef ARM_COMPUTE_AARCH64_V8_2

namespace
{
using namespace arm_compute;

Status validate_arguments(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::QASYMM8, DataType::U8, DataType::S8);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input0, ITensorInfo *input1, ITensorInfo *output)
{
    // Configure kernel window
    Window win = calculate_max_window(*output);

    AccessWindowRectangle output_access(output, 0, 0, 12, 8);

    const int input0_access_end = ceil_to_multiple(input0->tensor_shape().x(), 8);
    const int input1_access_end = ceil_to_multiple(input1->tensor_shape().x(), 12);

    bool window_changed = update_window_and_padding(win,
                                                    AccessWindowStatic(input0, 0, 0, input0_access_end, input0->tensor_shape().y()),
                                                    AccessWindowStatic(input1, 0, 0, input1_access_end, input1->tensor_shape().y()),
                                                    output_access);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

template <typename strategy, typename To, typename Tr>
void *align_workspace(GemmInterleaved<strategy, To, Tr> &gemm, const ThreadInfo &info, ITensor *ws)
{
    constexpr size_t alignment      = 4096;
    const size_t     offset         = (gemm.get_working_size() + alignment - 1) * info.thread_id;
    void            *workspace      = ws->buffer() + offset;
    size_t           workspace_size = ws->info()->total_size();

    if(support::cpp11::align(alignment, gemm.get_working_size(), workspace, workspace_size) == nullptr)
    {
        ARM_COMPUTE_ERROR("Not enough space to align buffer!");
    }
    return workspace;
}

template <typename strategy>
void execute_gemm(const Window &win, Iterator &in0, Iterator &in1, Iterator &out,
                  const ThreadInfo &info, ITensor *ws, int M, int N, int K, bool is_transposed_0, bool is_transposed_1,
                  int lda, int ldb, int ldc, float alpha, float beta)
{
    GemmInterleaved<strategy, typename strategy::operand_type, typename strategy::result_type> gemm(&info.cpu_info, M, N, K, is_transposed_0, is_transposed_1);
    void *workspace = align_workspace(gemm, info, ws);
    execute_window_loop(win, [&](const Coordinates & id)
    {
        gemm.execute(reinterpret_cast<const typename strategy::operand_type *>(in0.ptr()), lda,
                     reinterpret_cast<const typename strategy::operand_type *>(in1.ptr()), ldb,
                     reinterpret_cast<typename strategy::result_type *>(out.ptr()), ldc,
                     alpha, beta, workspace);
    },
    in0, out);
}
} // namespace

namespace arm_compute
{
void NEGEMMLowpAArch64V8P4Kernel::internal_configure(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha, float beta, bool is_transposed_0,
                                                     bool is_transposed_1)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input0, input1, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input0->info(), input1->info(), output->info()));

    _input0          = input0;
    _input1          = input1;
    _output          = output;
    _workspace       = workspace;
    _alpha           = alpha;
    _beta            = beta;
    _is_transposed_0 = is_transposed_0;
    _is_transposed_1 = is_transposed_1;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input0->info(), input1->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEGEMMLowpAArch64V8P4Kernel::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input0, input1, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input0->clone().get(), input1->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEGEMMLowpAArch64V8P4Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const int lda = _input0->info()->strides_in_bytes().y();
    const int ldb = _input1->info()->strides_in_bytes().y();
    const int ldc = _output->info()->strides_in_bytes().y() / sizeof(uint32_t);

    const int M = std::min(_output->info()->tensor_shape().y(), static_cast<size_t>(window.y().end())) - window.y().start();
    const int N = _output->info()->tensor_shape().x();
    const int K = _input0->info()->tensor_shape().x();

    // Only iterate over batches
    Window win(window);
    win.set(0, Window::Dimension(0, 1, 1));
    win.set(1, Window::Dimension(0, 1, 1));

    Iterator in0(_input0, window);
    Iterator in1(_input1, window);
    Iterator out(_output, window);

    switch(_input0->info()->data_type())
    {
        case DataType::QASYMM8:
        case DataType::U8:
        {
            execute_gemm<gemm_u8_12x8>(win, in0, in1, out, info, _workspace, M, N, K, _is_transposed_0, _is_transposed_1, lda, ldb, ldc, _alpha, _beta);
            break;
        }
        case DataType::S8:
        {
            execute_gemm<gemm_s8_12x8>(win, in0, in1, out, info, _workspace, M, N, K, _is_transposed_0, _is_transposed_1, lda, ldb, ldc, _alpha, _beta);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported.");
            break;
        }
    }
}
} // namespace arm_compute
#endif /* ARM_COMPUTE_AARCH64_V8_2 */
