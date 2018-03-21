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
#ifndef __ARM_COMPUTE_CL_TYPES_H__
#define __ARM_COMPUTE_CL_TYPES_H__

#include <string>

namespace arm_compute
{
/** Default string for the CLKernel configuration id */
static const std::string default_config_id = "no_config_id";

/** Available GPU Targets */
enum class GPUTarget
{
    UNKNOWN       = 0x000,
    GPU_ARCH_MASK = 0xF00,
    MIDGARD       = 0x100,
    BIFROST       = 0x200,
    T600          = 0x110,
    T700          = 0x120,
    T800          = 0x130,
    G71           = 0x210,
    G72           = 0x220,
    G51           = 0x230,
    G51BIG        = 0x231,
    G51LIT        = 0x232,
    TNOX          = 0x240,
    TTRX          = 0x250,
    TBOX          = 0x260
};

/** Available OpenCL Version */
enum class CLVersion
{
    CL10,   /* the OpenCL 1.0 */
    CL11,   /* the OpenCL 1.1 */
    CL12,   /* the OpenCL 1.2 */
    CL20,   /* the OpenCL 2.0 and above */
    UNKNOWN /* unkown version */
};
}
#endif /* __ARM_COMPUTE_CL_TYPES_H__ */
