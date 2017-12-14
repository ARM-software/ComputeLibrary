/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEL2NORMALIZEKERNEL_H__
#define __ARM_COMPUTE_NEL2NORMALIZEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for performing a L2 normalize on a given axis given the square sum of it in this axis */
class NEL2NormalizeLayerKernel : public INEKernel
{
public:
    /** Default constructor */
    NEL2NormalizeLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEL2NormalizeLayerKernel(const NEL2NormalizeLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEL2NormalizeLayerKernel &operator=(const NEL2NormalizeLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEL2NormalizeLayerKernel(NEL2NormalizeLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEL2NormalizeLayerKernel &operator=(NEL2NormalizeLayerKernel &&) = default;
    /** Default destructor */
    ~NEL2NormalizeLayerKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input   Source tensor. Data types supported: F32.
     * @param[in]  sum     Sum values tensor. Data types supported: same as @p input.
     * @param[out] output  Destination tensor. Data types supported: same as @p input.
     * @param[in]  axis    Dimension along which to reduce. Supported reduction axis : 0
     * @param[in]  epsilon Lower bound value for the normalization.
     */
    void configure(const ITensor *input, const ITensor *sum, ITensor *output, unsigned int axis, float epsilon);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    const ITensor *_sum;
    ITensor       *_output;
    unsigned int   _axis;
    float          _epsilon;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEL2NORMALIZEKERNEL_H__ */
