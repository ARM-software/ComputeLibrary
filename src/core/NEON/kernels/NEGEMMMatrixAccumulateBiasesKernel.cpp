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
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAccumulateBiasesKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

NEGEMMMatrixAccumulateBiasesKernel::NEGEMMMatrixAccumulateBiasesKernel()
    : _accum(nullptr), _biases(nullptr)
{
}

void NEGEMMMatrixAccumulateBiasesKernel::configure(ITensor *accum, const ITensor *biases)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(accum, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(biases, accum);
    ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() != 1);

    _biases = biases;
    _accum  = accum;

    constexpr unsigned int num_elems_processed_per_iteration = 4;

    // Configure kernel window
    Window win = calculate_max_window(*accum->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowStatic output_access(biases->info(), 0, 0, biases->info()->dimension(0), biases->info()->dimension(1));

    update_window_and_padding(win,
                              AccessWindowHorizontal(accum->info(), 0, num_elems_processed_per_iteration),
                              output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), accum->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGEMMMatrixAccumulateBiasesKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window win_biases;
    win_biases.set(Window::DimX, Window::Dimension(window.x().start(), window.x().end(), window.x().step()));
    win_biases.set(Window::DimY, Window::Dimension(0, 1, 1));

    Iterator in0_out(_accum, window);
    Iterator in1(_biases, win_biases);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float32x4_t accum  = vld1q_f32(reinterpret_cast<const float *>(in0_out.ptr()));
        const float32x4_t biases = vld1q_f32(reinterpret_cast<const float *>(in1.ptr()));

        vst1q_f32(reinterpret_cast<float *>(in0_out.ptr()), vaddq_f32(accum, biases));
    },
    in0_out, in1);
}
