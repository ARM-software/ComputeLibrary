/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NETILE_H
#define ARM_COMPUTE_NETILE_H

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to run @ref NETileKernel */
class NETile : public INESimpleFunctionNoBorder
{
public:
    /** Set the source, destination of the kernel
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in]  input     Source tensor. Data type supported: All.
     * @param[out] output    Destination tensor. Same as @p input
     * @param[in]  multiples Contains the number of times the input tensor should be replicated on the given dimension.
     */
    void configure(const ITensor *input, ITensor *output, const Multiples &multiples);
    /** Static function to check if given info will lead to a valid configuration of @ref NETile
     *
     * @param[in] input     Source tensor info. Data type supported: All.
     * @param[in] output    Destination tensor info. Same as @p input
     * @param[in] multiples Contains the number of times the input tensor should be replicated on the given dimension.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Multiples &multiples);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NETILE_H */
