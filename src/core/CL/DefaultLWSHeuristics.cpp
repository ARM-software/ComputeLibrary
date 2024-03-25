/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#include "src/core/CL/DefaultLWSHeuristics.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"

namespace
{
cl::NDRange get_gemm_lws(size_t gws_x, size_t gws_y, size_t gws_z)
{
    ARM_COMPUTE_UNUSED(gws_y);

    if (gws_z != 1)
    {
        return cl::NDRange(4, 4, 2);
    }
    else
    {
        if (gws_x > 256)
        {
            return cl::NDRange(2, 16, 1);
        }
        else
        {
            return cl::NDRange(32, 4, 1);
        }
    }
}

cl::NDRange get_winograd_lws(size_t gws_x, size_t gws_y, size_t gws_z)
{
    ARM_COMPUTE_UNUSED(gws_x, gws_y, gws_z);

    return cl::NDRange(4, 2, 1);
}

cl::NDRange get_direct_lws(size_t gws_x, size_t gws_y, size_t gws_z)
{
    ARM_COMPUTE_UNUSED(gws_z);

    if (gws_x < gws_y)
    {
        if (gws_x < 4)
        {
            return cl::NDRange(std::min(gws_x, static_cast<size_t>(2u)), 32, 1);
        }
        else
        {
            return cl::NDRange(std::min(gws_x, static_cast<size_t>(4u)), 8, 1);
        }
    }
    else
    {
        return cl::NDRange(8, 4, 1);
    }
}

cl::NDRange get_dwc_lws(size_t gws_x, size_t gws_y, size_t gws_z)
{
    ARM_COMPUTE_UNUSED(gws_y);
    ARM_COMPUTE_UNUSED(gws_z);

    if (gws_x < 32)
    {
        return cl::NDRange(gws_x, 4, 4);
    }
    else
    {
        return cl::NDRange(8, 4, 2);
    }
}
} // namespace

namespace arm_compute
{
cl::NDRange get_default_lws_for_type(CLKernelType kernel_type, cl::NDRange gws)
{
    const size_t gws_x = gws[0];
    const size_t gws_y = gws[1];
    const size_t gws_z = gws[2];

    switch (kernel_type)
    {
        case CLKernelType::GEMM:
        {
            return get_gemm_lws(gws_x, gws_y, gws_z);
        }
        case CLKernelType::DIRECT:
        {
            return get_direct_lws(gws_x, gws_y, gws_z);
        }
        case CLKernelType::WINOGRAD:
        {
            return get_winograd_lws(gws_x, gws_y, gws_z);
        }
        case CLKernelType::DEPTHWISE:
        {
            return get_dwc_lws(gws_x, gws_y, gws_z);
        }
        default:
        {
            return CLKernelLibrary::get().default_ndrange();
        }
    }
}
} // namespace arm_compute
