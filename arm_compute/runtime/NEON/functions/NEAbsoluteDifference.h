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
#ifndef ARM_COMPUTE_NEABSOLUTEDIFFERENCE_H
#define ARM_COMPUTE_NEABSOLUTEDIFFERENCE_H

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEAbsoluteDifferenceKernel
 *
 * @note The image data type for the inputs must be U8 or S16
 * @note The function calculates the absolute difference also when the 2 inputs have different image data types
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEAbsoluteDifference : public INESimpleFunctionNoBorder
{
public:
    /** Default constructor */
    NEAbsoluteDifference() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAbsoluteDifference(const NEAbsoluteDifference &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAbsoluteDifference &operator=(const NEAbsoluteDifference &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAbsoluteDifference(NEAbsoluteDifference &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAbsoluteDifference &operator=(NEAbsoluteDifference &&) = delete;
    /** Default destructor */
    ~NEAbsoluteDifference();
    /** Set the inputs and output images
     *
     * @param[in]  input1 Source tensor. Data types supported: U8/S16.
     * @param[in]  input2 Source tensor. Data types supported: U8/S16.
     * @param[out] output Destination tensor. Data types supported: U8/S16.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output);
};
}
#endif /* ARM_COMPUTE_NEABSOLUTEDIFFERENCE_H */
