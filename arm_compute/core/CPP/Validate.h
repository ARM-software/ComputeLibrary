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
#ifndef __ARM_COMPUTE_CPP_VALIDATE_H__
#define __ARM_COMPUTE_CPP_VALIDATE_H__

#include "arm_compute/core/Validate.h"

namespace arm_compute
{
/** Return an error if the data type of the passed tensor info is FP16 and FP16 support is not compiled in.
 *
 * @param[in] function    Function in which the error occurred.
 * @param[in] file        Name of the file where the error occurred.
 * @param[in] line        Line on which the error occurred.
 * @param[in] tensor_info Tensor info to validate.
 *
 * @return Status
 */
inline arm_compute::Status error_on_unsupported_cpu_fp16(const char *function, const char *file, const int line,
                                                         const ITensorInfo *tensor_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(tensor_info->data_type() == DataType::F16,
                                        function, file, line, "This CPU architecture does not support F16 data type, you need v8.2 or above");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    return arm_compute::Status {};
}

/** Return an error if the data type of the passed tensor is FP16 and FP16 support is not compiled in.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor   Tensor to validate.
 *
 * @return Status
 */
inline arm_compute::Status error_on_unsupported_cpu_fp16(const char *function, const char *file, const int line,
                                                         const ITensor *tensor)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_cpu_fp16(function, file, line, tensor->info()));
    return arm_compute::Status{};
}

#define ARM_COMPUTE_ERROR_ON_CPU_F16_UNSUPPORTED(tensor) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_unsupported_cpu_fp16(__func__, __FILE__, __LINE__, tensor))

#define ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(tensor) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_cpu_fp16(__func__, __FILE__, __LINE__, tensor))
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CPP_VALIDATE_H__ */
