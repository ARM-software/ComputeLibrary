/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NETRANSPOSEKERNEL_H
#define ARM_COMPUTE_NETRANSPOSEKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Neon kernel which transposes the elements of a matrix.
 *
 * [width, height, batch] -> [height, width, batch]
 *
 */
class NETransposeKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NETransposeKernel";
    }
    /** Default constructor */
    NETransposeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETransposeKernel(const NETransposeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETransposeKernel &operator=(const NETransposeKernel &) = delete;
    /** Allow instances of this class to be moved */
    NETransposeKernel(NETransposeKernel &&) = default;
    /** Allow instances of this class to be moved */
    NETransposeKernel &operator=(NETransposeKernel &&) = default;
    /** Default destructor */
    ~NETransposeKernel() = default;

    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor. Data types supported: All
     * @param[out] output Output tensor. Data type supported: Same as @p input
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NETransposeKernel
     *
     * @param[in] input  Input tensor. Data types supported: All
     * @param[in] output Output tensor. Data type supported: Same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the transpose functions
     *
     * @param[in]  input  An input tensor. Data types supported: All
     * @param[out] output The output tensor. Data type supported: same as @p input
     * @param[in]  window Region on which to execute the kernel.
     */
    using TransposeFunction = void(const ITensor *input, ITensor *output, const Window &window);
    /** Transpose function to use for the particular tensor types passed to configure() */
    TransposeFunction *_func;
    const ITensor     *_input;
    ITensor           *_output;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NETRANSPOSEKERNEL_H */
