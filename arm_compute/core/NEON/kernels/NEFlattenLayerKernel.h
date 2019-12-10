/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_NEFLATTENLAYERKERNEL_H
#define ARM_COMPUTE_NEFLATTENLAYERKERNEL_H

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the flatten layer kernel. */
class NEFlattenLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEFlattenLayerKernel";
    }
    /** Default constructor */
    NEFlattenLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFlattenLayerKernel(const NEFlattenLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFlattenLayerKernel &operator=(const NEFlattenLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEFlattenLayerKernel(NEFlattenLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEFlattenLayerKernel &operator=(NEFlattenLayerKernel &&) = default;
    /** Default destructor */
    ~NEFlattenLayerKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input  First input tensor to flatten with at least 3 dimensions.
     *                    The dimensions above the third will be interpreted as batches. Data types supported: All
     * @param[out] output Output tensor with shape [w*h*d, input_batches] where:
     *                    w = width input tensor, h = height input tensor and d = depth input tensor. Data type supported: same as @p input
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEFlattenLayerKernel
     *
     * @param[in]  input  First input tensor to flatten with at least 3 dimensions.
     *                    The dimensions above the third will be interpreted as batches. Data types supported: All
     * @param[out] output Output tensor with shape [w*h*d, input_batches] where:
     *                    w = width input tensor, h = height input tensor and d = depth input tensor. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    ITensor       *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEFLATTENLAYERKERNEL_H */
