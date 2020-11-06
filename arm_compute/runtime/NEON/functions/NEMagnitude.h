/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEMAGNITUDE_H
#define ARM_COMPUTE_NEMAGNITUDE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEMagnitudePhaseKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEMagnitude : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEMagnitude() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMagnitude(const NEMagnitude &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMagnitude &operator=(const NEMagnitude &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEMagnitude(NEMagnitude &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEMagnitude &operator=(NEMagnitude &&) = delete;
    /** Default destructor */
    ~NEMagnitude();
    /** Initialise the kernel's inputs.
     *
     * @param[in]  input1   First tensor input. Data type supported: S16.
     * @param[in]  input2   Second tensor input. Data type supported: S16.
     * @param[out] output   Output tensor. Data type supported: S16.
     * @param[in]  mag_type (Optional) Magnitude calculation type. Default: L2NORM.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output, MagnitudeType mag_type = MagnitudeType::L2NORM);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEMAGNITUDE_H */
