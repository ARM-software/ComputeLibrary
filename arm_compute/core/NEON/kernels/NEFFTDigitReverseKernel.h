/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEFFTDIGITREVERSEKERNEL_H__
#define __ARM_COMPUTE_NEFFTDIGITREVERSEKERNEL_H__

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for the digit reverse operation kernel. */
class NEFFTDigitReverseKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEFFTDigitReverseKernel";
    }
    /** Constructor */
    NEFFTDigitReverseKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFTDigitReverseKernel(const NEFFTDigitReverseKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFTDigitReverseKernel &operator=(const NEFFTDigitReverseKernel &) = delete;
    /** Default Move Constructor. */
    NEFFTDigitReverseKernel(NEFFTDigitReverseKernel &&) = default;
    /** Default move assignment operator */
    NEFFTDigitReverseKernel &operator=(NEFFTDigitReverseKernel &&) = default;
    /** Default destructor */
    ~NEFFTDigitReverseKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: F32. Number of channels supported: 1 (real tensor) or 2 (complex tensor).
     * @param[out] output Destination tensor. Data type supported: same as @p input. Number of channels supported: 2 (complex tensor).
     * @param[in]  idx    Digit reverse index tensor. Data type supported: U32
     * @param[in]  config Kernel configuration.
     */
    void configure(const ITensor *input, ITensor *output, const ITensor *idx, const FFTDigitReverseKernelInfo &config);

    /** Static function to check if given info will lead to a valid configuration of @ref NEFFTDigitReverseKernel
     *
     * @param[in] input  Source tensor info. Data types supported: F32. Number of channels supported: 1 (real tensor) or 2 (complex tensor).
     * @param[in] output Destination tensor info. Data type supported: same as @p input. Number of channels supported: 2 (complex tensor).
     * @param[in] idx    Digit reverse index tensor info. Data type supported: U32
     * @param[in] config Kernel configuration
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *idx, const FFTDigitReverseKernelInfo &config);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    using NEFFTDigitReverseKernelFunctionPtr = void (NEFFTDigitReverseKernel::*)(const Window &window);

    template <bool is_input_complex, bool is_conj>
    void digit_reverse_kernel_axis_0(const Window &window);

    template <bool is_input_complex, bool is_conj>
    void digit_reverse_kernel_axis_1(const Window &window);

    NEFFTDigitReverseKernelFunctionPtr _func;
    const ITensor                     *_input;
    ITensor                           *_output;
    const ITensor                     *_idx;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEFFTDIGITREVERSEKERNEL_H__ */
