/*
 * Copyright (c) 2019-2023 Arm Limited.
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

#ifndef ARM_COMPUTE_NEGATHER_H
#define ARM_COMPUTE_NEGATHER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to run @ref NEGatherKernel */
class NEGather : public INESimpleFunctionNoBorder
{
public:
    /** Initialise the kernel's inputs and outputs
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in]  input   Source tensor. Supported tensor rank: up to 4. Data type supported: All
     * @param[in]  indices Indices tensor. Supported tensor rank: up to 3. Must be one of the following type: U32/S32. Each value must be in range [0, input.shape[@p axis]), otherwise the result will become unpredictable.
     *                     @note The "axis" must be in the range [0, input.rank -1] when indices is a vector, and must be 1 when indices is a 2D or 3D tensor.
     * @param[out] output  Destination tensor. Data type supported: Same as @p input
     * @param[in]  axis    (Optional) The axis in @p input to gather @p indices from. Defaults to 0
     *
     */
    void configure(const ITensor *input, const ITensor *indices, ITensor *output, int axis = 0);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref NEGather::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, int axis);
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NEGATHER_H */
