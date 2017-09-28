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

#ifndef __ARM_COMPUTE_NEMINMAXLAYERKERNEL_H__
#define __ARM_COMPUTE_NEMINMAXLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "support/Mutex.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to perform min max search on a 3D tensor. */
class NEMinMaxLayerKernel : public INEKernel
{
public:
    /** Default constructor */
    NEMinMaxLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMinMaxLayerKernel(const NEMinMaxLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMinMaxLayerKernel &operator=(const NEMinMaxLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEMinMaxLayerKernel(NEMinMaxLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEMinMaxLayerKernel &operator=(NEMinMaxLayerKernel &&) = default;
    /** Default destructor */
    ~NEMinMaxLayerKernel() = default;

    /** Initialise the kernel's input and outputs.
     *
     * @note output[0] = minimum
     * @note output[1] = maximum
     *
     * @param[in]  input  Input tensor with at least 3 dimensions. The dimensions over the third will be interpreted as batches. Data type supported: F32.
     * @param[out] output Output tensor with shape [2, batches, ...] which stores the minimum and maximum value for each 3D input tensor.
     *                    The dimensions over the second must match the batched dimensions of the input tensor. Data types supported: F32
     */
    void configure(const ITensor *input, ITensor *output);
    /** Resets global minimum and maximum. */
    void reset();

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    void update_min_max(float *out_ptr, float min, float max);
    const ITensor *_input;
    ITensor       *_output;
    Mutex          _mtx;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEMINMAXLAYERKERNEL_H__ */
