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
#ifndef UTILS_CORE_ASSEMBLY_UTILS_H
#define UTILS_CORE_ASSEMBLY_UTILS_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/kernels/assembly/common.hpp"
#include "src/cpu/kernels/assembly/arm_gemm.hpp"

namespace arm_compute
{
namespace assembly_utils
{
/** Performs a mapping between Compute Library ActivationLayerInfo and the assembly Activation structure.
 *
 * @param[in] act Compute Library activation info.
 *
 * @return Assembly activation info.
 */
arm_gemm::Activation map_to_arm_gemm_activation(const ActivationLayerInfo &act);

/** Performs a mapping between Compute Library PadStrideInfo and the assembly PaddingValues structure.
 *
 * @param[in] pad_stride_info Compute Library padding and strides info.
 *
 * @return Assembly padding values.
 */
arm_conv::PaddingValues map_to_arm_conv_padding(const PadStrideInfo &pad_stride_info);

/** Performs a mapping from Compute Library WeightFormat to the assembly WeightFormat enum
 *
 * @param[in] weight_format Compute Library WeightFormat enum value
 *
 * @return Assembly WeightFormat
 */
arm_gemm::WeightFormat map_to_arm_gemm_weight_format(const arm_compute::WeightFormat &weight_format);

/** Performs a mapping from Assembly WeightFormat to the Compute Library WeightFormat enum
 *
 * @param[in] weight_format Assembly WeightFormat enum value
 *
 * @return Compute Library WeightFormat
 */
arm_compute::WeightFormat map_to_arm_compute_weight_format(const arm_gemm::WeightFormat &weight_format);
} // namespace assembly
} // namespace arm_compute
#endif /* UTILS_CORE_ASSEMBLY_UTILS_H */
