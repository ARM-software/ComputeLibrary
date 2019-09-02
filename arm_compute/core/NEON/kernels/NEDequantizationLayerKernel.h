/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEDEQUANTIZATIONLAYERKERNEL_H__
#define __ARM_COMPUTE_NEDEQUANTIZATIONLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the dequantization layer kernel. */
class NEDequantizationLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDequantizationLayerKernel";
    }
    /** Default constructor */
    NEDequantizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDequantizationLayerKernel(const NEDequantizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDequantizationLayerKernel &operator=(const NEDequantizationLayerKernel &) = delete;
    /** Default Move Constructor. */
    NEDequantizationLayerKernel(NEDequantizationLayerKernel &&) = default;
    /** Default move assignment operator */
    NEDequantizationLayerKernel &operator=(NEDequantizationLayerKernel &&) = default;
    /** Default destructor */
    ~NEDequantizationLayerKernel() = default;
    /** Set input, output tensors.
     *
     * @param[in]  input  Source tensor. Data type supported: QASYMM8/QSYMM8/QSYMM16.
     * @param[out] output Destination tensor with the same dimensions of input. Data type supported: F16/F32.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEDequantizationLayerKernel
     *
     * @param[in] input  Input tensor info. Data types supported: QASYMM8/QSYMM8/QSYMM16.
     * @param[in] output Output tensor info. Data types supported: F16/F32.
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
#endif /*__ARM_COMPUTE_NEDEQUANTIZATIONLAYERKERNEL_H__ */
