/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGEMMTRANSPOSE1XW_H
#define ARM_COMPUTE_NEGEMMTRANSPOSE1XW_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to execute NEGEMMTranspose1xWKernel. This function calls the following NEON kernels:
 *
 *  -# @ref NEGEMMTranspose1xWKernel
 *
 */
class NEGEMMTranspose1xW : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEGEMMTranspose1xW() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMTranspose1xW(const NEGEMMTranspose1xW &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMTranspose1xW &operator=(const NEGEMMTranspose1xW &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMTranspose1xW(NEGEMMTranspose1xW &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMTranspose1xW &operator=(NEGEMMTranspose1xW &&) = delete;
    /** Default destructor */
    ~NEGEMMTranspose1xW() = default;
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input  First input tensor. Data type supported: All
     * @param[out] output Output tensor. Data type supported: same as @p input
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMTranspose1xW
     *
     * @param[in] input  First input tensor. Data type supported: All
     * @param[in] output Output tensor. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGEMMTRANSPOSE1XW_H */
