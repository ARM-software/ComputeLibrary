/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_COMMON_TYPES_H_
#define SRC_COMMON_TYPES_H_

#include "arm_compute/AclDescriptors.h"
#include "arm_compute/AclTypes.h"

namespace arm_compute
{
enum class StatusCode
{
    Success            = AclSuccess,
    RuntimeError       = AclRuntimeError,
    OutOfMemory        = AclOutOfMemory,
    Unimplemented      = AclUnimplemented,
    UnsupportedTarget  = AclUnsupportedTarget,
    InvalidTarget      = AclInvalidTarget,
    InvalidArgument    = AclInvalidArgument,
    UnsupportedConfig  = AclUnsupportedConfig,
    InvalidObjectState = AclInvalidObjectState,
};

enum class Target
{
    Cpu    = AclTarget::AclCpu,
    GpuOcl = AclTarget::AclGpuOcl,
};

enum class ExecutionMode
{
    FastRerun = AclPreferFastRerun,
    FastStart = AclPreferFastStart,
};

enum class ImportMemoryType
{
    HostPtr = AclImportMemoryType::AclHostPtr
};
} // namespace arm_compute
#endif /* SRC_COMMON_TYPES_H_ */
