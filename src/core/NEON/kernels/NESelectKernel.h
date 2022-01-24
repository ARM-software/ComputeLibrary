/*
 * Copyright (c) 2018-2020, 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_NESELECTKERNEL_H
#define ARM_COMPUTE_NESELECTKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for the select kernel
 *
 * Select is computed by:
 * @f[ output(i) = condition(i) ? x(i) : y(i) @f]
 *
 */
class NESelectKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NESelectKernel";
    }
    /** Default constructor */
    NESelectKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESelectKernel(const NESelectKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESelectKernel &operator=(const NESelectKernel &) = delete;
    /** Allow instances of this class to be moved */
    NESelectKernel(NESelectKernel &&) = default;
    /** Allow instances of this class to be moved */
    NESelectKernel &operator=(NESelectKernel &&) = default;
    /** Default destructor */
    ~NESelectKernel() = default;

    /** Common signature for all the specialised elementwise functions
     *
     * @param[in]  c      Condition input tensor. Data types supported: U8.
     * @param[in]  x      First input tensor. Data types supported: All.
     * @param[out] y      Second input tensor. Data types supported: Same as @p x
     * @param[in]  output Output tensor. Data types supported: Same as @p x
     */
    void configure(const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output);

    /** Validate the argument passed to the kernel
     *
     * @param[in] c      Condition input tensor. Data types supported: U8.
     * @param[in] x      First input tensor. Data types supported: All.
     * @param[in] y      Second input tensor. Data types supported: Same as @p x
     * @param[in] output Output tensor. Data types supported: Same as @p x.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *c, const ITensorInfo *x, const ITensorInfo *y, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:

    const ITensor *_c;             /**< Condition tensor */
    const ITensor *_x;             /**< Source tensor 1 */
    const ITensor *_y;             /**< Source tensor 2 */
    ITensor       *_output;        /**< Destination tensor */
    bool           _has_same_rank; /**< Flag that indicates if condition tensor and other inputs have the same rank */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NESELECTKERNEL_H */
