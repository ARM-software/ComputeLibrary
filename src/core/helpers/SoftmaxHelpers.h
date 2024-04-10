/*
* Copyright (c) 2020 Arm Limited.
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
#ifndef SRC_CORE_HELPERS_SOFTMAXHELPERS_H
#define SRC_CORE_HELPERS_SOFTMAXHELPERS_H

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace softmax_helpers
{
/** Given a softmax axis, this function returns the permutation vector required to put the axis to the front
 *
 * @note This function assumes a tensor rank <= 4
 *
 * Axis selects the dimension on which softmax is performed.
 * E.g. For input of shape 4x5x6 and axis=1, softmax will be applied to 4x6=24 vectors of size 5.
 * Interally softmax kernels is always performed on the first dimension (front dimension), therefore permutation is
 * required to put the dimension specified by @p axis to the first dimension.
 *
 * @param[in] axis Axis on which to perform softmax. Supported: 1, 2, 3 (0 implies no permutation needed)
 *
 * @return the permutation vector
 */
PermutationVector get_permutation_vector_from_softmax_axis(size_t axis);
} // namespace softmax_helpers
} // namespace arm_compute

#endif /* SRC_CORE_HELPERS_SOFTMAXHELPERS_H */
