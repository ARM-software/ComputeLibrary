/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/tuners/MidgardTuner.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernels.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace tuners
{
namespace
{
void tune_gemm_kernel(CLGEMMMatrixMultiplyKernel &k)
{
    cl::NDRange     lws_hint   = k.lws_hint();
    const GPUTarget gpu_target = k.get_target();

    switch(gpu_target)
    {
        case GPUTarget::MIDGARD:
        case GPUTarget::T600:
        case GPUTarget::T700:
        case GPUTarget::T800:
            if(k._output->info()->dimension(1) == 196)
            {
                lws_hint = cl::NDRange(1, 7);
            }
            else
            {
                lws_hint = cl::NDRange(8, 8);
            }
            break;
        default:
            lws_hint = cl::NullRange;
    }

    k.set_lws_hint(lws_hint);
}
} // namespace

void MidgardTuner::tune_kernel_static(ICLKernel &kernel)
{
    if(dynamic_cast<CLGEMMMatrixMultiplyKernel *>(&kernel) != nullptr)
    {
        tune_gemm_kernel(*utils::cast::polymorphic_downcast<CLGEMMMatrixMultiplyKernel *>(&kernel));
    }
}

void MidgardTuner::tune_kernel_dynamic(ICLKernel &kernel)
{
    ARM_COMPUTE_UNUSED(kernel);
}

void MidgardTuner::tune_kernel_dynamic(ICLKernel &kernel, ITensorPack &tensors)
{
    ARM_COMPUTE_UNUSED(kernel, tensors);
}
} // namespace tuners
} // namespace arm_compute
