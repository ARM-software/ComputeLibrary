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
#ifndef SRC_COMMON_CPUINFO_CPUMODEL_H
#define SRC_COMMON_CPUINFO_CPUMODEL_H

#include <cstdint>
#include <string>

#include "arm_compute/core/CPP/CPPTypes.h"

namespace arm_compute
{
namespace cpuinfo
{
using CpuModel = arm_compute::CPUModel;

/** Convert a CPU model value to a string
 *
 * @param model CpuModel value to be converted
 *
 * @return String representing the corresponding CpuModel
 */
std::string cpu_model_to_string(CpuModel model);

/** Extract the model type from the MIDR value
 *
 * @param[in] midr MIDR information
 *
 * @return CpuModel a mapped CPU model
 */
CpuModel midr_to_model(uint32_t midr);

/** Check if a model supports half-precision floating point arithmetic
 *
 * @note This is used in case of old kernel configurations where some capabilities are not exposed.
 *
 * @param[in] model Model to check for allowlisted capabilities
 */
bool model_supports_fp16(CpuModel model);

/** Check if a model supports dot product
 *
 * @note This is used in case of old kernel configurations where some capabilities are not exposed.
 *
 * @param[in] model Model to check for allowlisted capabilities
 */
bool model_supports_dot(CpuModel model);
} // namespace cpuinfo
} // namespace arm_compute
#endif /* SRC_COMMON_CPUINFO_CPUMODEL_H */
