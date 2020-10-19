/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_UTILS_HELPERS_TENSOR_INFO_H
#define ARM_COMPUTE_UTILS_HELPERS_TENSOR_INFO_H

#include "arm_compute/core/ITensorInfo.h"

namespace arm_compute
{
namespace helpers
{
namespace tensor_info
{
/** Checks if the quantization info of given tensors are different
 *
 * @param tensor_info_1 Tensor info of the first tensor
 * @param tensor_info_2 Tensor info of the second tensor
 * @param tensor_infos  Tensor infos of the rest tensors
 *
 * @return True if tensors have mismatching quantization info else false.
 */
template <typename... Ts>
inline bool tensors_have_different_quantization_info(const ITensorInfo *tensor_info_1, const ITensorInfo *tensor_info_2, Ts... tensor_infos)
{
    const QuantizationInfo first_quantization_info = tensor_info_1->quantization_info();

    const std::array < const ITensorInfo *, 1 + sizeof...(Ts) > tensor_infos_array{ { tensor_info_2, std::forward<Ts>(tensor_infos)... } };
    return std::any_of(tensor_infos_array.begin(), tensor_infos_array.end(), [&](const ITensorInfo * tensor_info)
    {
        return tensor_info->quantization_info() != first_quantization_info;
    });
}
} // namespace tensor_info
} // namespace helpers
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_HELPERS_TENSOR_INFO_H */
