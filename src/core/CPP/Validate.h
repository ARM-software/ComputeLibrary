/*
 * Copyright (c) 2018-2021, 2025-2026 Arm Limited.
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
#ifndef ACL_SRC_CORE_CPP_VALIDATE_H
#define ACL_SRC_CORE_CPP_VALIDATE_H

#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/Validate.h"

#include <algorithm>

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
inline Status
error_on_unsupported_cpu_fp16(const char *function, const char *file, const int line, const ITensorInfo *tensor_info)
{
    bool fp16_kernels_enabled = false;
#if defined(ARM_COMPUTE_ENABLE_FP16) && defined(ENABLE_FP16_KERNELS)
    fp16_kernels_enabled = true;
#endif /* defined(ARM_COMPUTE_ENABLE_FP16) && defined(ENABLE_FP16_KERNELS) */

    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(
        (tensor_info->data_type() == DataType::F16) && (!CPUInfo::get().has_fp16() || !fp16_kernels_enabled), function,
        file, line, "This CPU architecture does not support F16 data type, you need v8.2 or above");
    return Status{};
}

/** Return an error if the tensor sizes are too large.
 *
 * @param[in] function     Function in which the error occurred.
 * @param[in] file         Name of the file where the error occurred.
 * @param[in] line         Line on which the error occurred.
 * @param[in] tensor_infos Tensor infos to validate.
 *
 * @return Status
 */
template <typename... Ts>
inline Status error_on_unsupported_size(const char *function, const char *file, const int line, Ts &&...tensor_infos)
{
    constexpr size_t max_size_in_elements =
        (1ULL << 31) - 1 - 16; // Subtract one SIMD register size, in case window step causes overflow

    constexpr size_t max_stride = (1U << 31) - 1;

    const ITensorInfo *tensor_array[] = {std::forward<Ts>(tensor_infos)...};

    for (const ITensorInfo *tensor_info : tensor_array)
    {
        if (tensor_info != nullptr && tensor_info->data_type() != DataType::UNKNOWN)
        {
            const TensorShape &tensor_shape   = tensor_info->tensor_shape();
            const size_t       total_elements = tensor_shape.total_size();

            ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(total_elements > max_size_in_elements, function, file, line,
                                                "Maximum supported number of tensor elements is 2^31-1-16");

            const size_t num_dimensions = tensor_info->num_dimensions();
            if (num_dimensions > 1)
            {
                const size_t last_dim          = num_dimensions - 1;
                const size_t penultimate_shape = tensor_shape.total_size_lower(last_dim);
                const size_t penultimate_stride =
                    penultimate_shape * tensor_info->num_channels() * tensor_info->element_size();

                ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(penultimate_stride > max_stride, function, file, line,
                                                    "Maximum supported penultimate tensor stride is 2^31-1");
            }
        }
    }

    return Status{};
}

/** Return an error if the data type of the passed tensor info is BFLOAT16 and BFLOAT16 support is not compiled in.
 *
 * @param[in] function    Function in which the error occurred.
 * @param[in] file        Name of the file where the error occurred.
 * @param[in] line        Line on which the error occurred.
 * @param[in] tensor_info Tensor info to validate.
 *
 * @return Status
 */
inline Status
error_on_unsupported_cpu_bf16(const char *function, const char *file, const int line, const ITensorInfo *tensor_info)
{
    bool bf16_kernels_enabled = false;
#if defined(ARM_COMPUTE_ENABLE_BF16)
    bf16_kernels_enabled = true;
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */

    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(
        (tensor_info->data_type() == DataType::BFLOAT16) && (!CPUInfo::get().has_bf16() || !bf16_kernels_enabled),
        function, file, line, "This CPU architecture does not support BFloat16 data type, you need v8.6 or above");
    return Status{};
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
inline Status
error_on_unsupported_cpu_fp16(const char *function, const char *file, const int line, const ITensor *tensor)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_cpu_fp16(function, file, line, tensor->info()));
    return Status{};
}

/** Return an error if the data type of the passed tensor is BFLOAT16 and BFLOAT16 support is not compiled in.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor   Tensor to validate.
 *
 * @return Status
 */
inline Status
error_on_unsupported_cpu_bf16(const char *function, const char *file, const int line, const ITensor *tensor)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_cpu_bf16(function, file, line, tensor->info()));
    return Status{};
}

#define ARM_COMPUTE_ERROR_ON_CPU_F16_UNSUPPORTED(tensor) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_unsupported_cpu_fp16(__func__, __FILE__, __LINE__, tensor))

#define ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(tensor) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_cpu_fp16(__func__, __FILE__, __LINE__, tensor))

#define ARM_COMPUTE_ERROR_ON_CPU_BF16_UNSUPPORTED(tensor) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_unsupported_cpu_bf16(__func__, __FILE__, __LINE__, tensor))

#define ARM_COMPUTE_RETURN_ERROR_ON_CPU_BF16_UNSUPPORTED(tensor) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_cpu_bf16(__func__, __FILE__, __LINE__, tensor))

#define ARM_COMPUTE_ERROR_ON_SIZE_UNSUPPORTED(...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_unsupported_size(__func__, __FILE__, __LINE__, __VA_ARGS__))

#define ARM_COMPUTE_RETURN_ERROR_ON_SIZE_UNSUPPORTED(...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_size(__func__, __FILE__, __LINE__, __VA_ARGS__))

} // namespace arm_compute
#endif // ACL_SRC_CORE_CPP_VALIDATE_H
