/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#include "arm_compute/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/WindowIterator.h"

using namespace arm_compute;

INEGEMMWrapperKernel::INEGEMMWrapperKernel()
    : _a(nullptr), _b(nullptr), _c(nullptr), _params(), _gemm_info(), _window3d(), _window_shape()
{
}

INEGEMMWrapperKernel::Params INEGEMMWrapperKernel::extract_parameters(const ITensor *a, const ITensor *b, const ITensor *c, const GEMMInfo &gemm_info)
{
    Params p;

    ARM_COMPUTE_ERROR_ON_NULLPTR(a);
    ARM_COMPUTE_ERROR_ON_NULLPTR(b);
    ARM_COMPUTE_ERROR_ON_NULLPTR(c);

    // Initalize params
    p.M       = c->info()->tensor_shape().y();
    p.N       = c->info()->tensor_shape().x();
    p.K       = a->info()->tensor_shape().x();
    p.multis  = b->info()->tensor_shape().z();
    p.batches = c->info()->tensor_shape().total_size_upper(2) / p.multis; //COMPMID-1423: Agree on and document the layout of gemm inputs/outputs

    // Update M in case of GEMM3D for output
    if(gemm_info.depth_output_gemm3d() != 0)
    {
        p.M       = c->info()->tensor_shape().y() * c->info()->tensor_shape().z();
        p.batches = c->info()->tensor_shape().total_size_upper(3) / p.multis;
    }

    return p;
}

void INEGEMMWrapperKernel::configure(const ITensor *a, const ITensor *b, ITensor *c, float alpha, float beta, const GEMMInfo &gemm_info)
{
    _gemm_info = gemm_info;
    _params    = extract_parameters(a, b, c, gemm_info);
    _a         = a;
    _b         = b;
    _c         = c;

    _window3d     = configure_internal(alpha, beta);
    _window_shape = _window3d.shape();

    // Convert the 3D window into a 1D window in order to allow the scheduler to arbitrary split it.
    Window collapsed;
    collapsed.set(0, Window::Dimension(0, _window3d.num_iterations_total()));

    INEKernel::configure(collapsed);
}

void INEGEMMWrapperKernel::run(const Window &window, const ThreadInfo &info)
{
    const Coordinates start_offset = index2coords(_window_shape, window.x().start());
    const Coordinates end_offset   = index2coords(_window_shape, window.x().end() - 1);

    run_internal(_window3d, start_offset, end_offset, info);
}
