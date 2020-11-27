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
#ifndef ARM_COMPUTE_CL_VALIDATE_H
#define ARM_COMPUTE_CL_VALIDATE_H

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Validate.h"

namespace arm_compute
{
#define ARM_COMPUTE_ERROR_ON_F16_UNSUPPORTED(tensor) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_unsupported_fp16(__func__, __FILE__, __LINE__, tensor, CLKernelLibrary::get().fp16_supported()))

#define ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(tensor) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_fp16(__func__, __FILE__, __LINE__, tensor, CLKernelLibrary::get().fp16_supported()))

/** Return an error if int64_base_atomics extension is not supported by the device.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 *
 * @return Status
 */
inline arm_compute::Status error_on_unsupported_int64_base_atomics(const char *function, const char *file, const int line)
{
    if(!CLKernelLibrary::get().int64_base_atomics_supported())
    {
        return ARM_COMPUTE_CREATE_ERROR_LOC(arm_compute::ErrorCode::UNSUPPORTED_EXTENSION_USE, function, file, line, "Atomic functions are not supported");
    }
    return arm_compute::Status{};
}

#define ARM_COMPUTE_ERROR_ON_INT64_BASE_ATOMICS_UNSUPPORTED() \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_unsupported_int64_base_atomics(__func__, __FILE__, __LINE__));

#define ARM_COMPUTE_RETURN_ERROR_ON_INT64_BASE_ATOMICS_UNSUPPORTED() \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_int64_base_atomics(__func__, __FILE__, __LINE__));

} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_VALIDATE_H */
