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
#ifndef ARM_COMPUTE_CLQUANTIZATIONLAYERKERNEL_H
#define ARM_COMPUTE_CLQUANTIZATIONLAYERKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the quantization layer kernel.
 *
 * @note The implementation supports only 3D input tensors.
 */
class CLQuantizationLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLQuantizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLQuantizationLayerKernel(const CLQuantizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLQuantizationLayerKernel &operator=(const CLQuantizationLayerKernel &) = delete;
    /** Default Move Constructor. */
    CLQuantizationLayerKernel(CLQuantizationLayerKernel &&) = default;
    /** Default move assignment operator */
    CLQuantizationLayerKernel &operator=(CLQuantizationLayerKernel &&) = default;
    /** Default destructor */
    ~CLQuantizationLayerKernel() = default;
    /** Set the input, output.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
     * @param[out] output Destination tensor with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16.
     *
     * @note Output auto initialization is not supported by this kernel
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Set the input, output.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
     * @param[out] output          Destination tensor with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16.
     *
     * @note Output auto initialization is not supported by this kernel
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLQuantizationLayerKernel
     *
     * @param[in] input  Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
     * @param[in] output Destination tensor info with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLQUANTIZATIONLAYERKERNEL_H */
