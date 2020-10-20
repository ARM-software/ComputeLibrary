/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEFFTSCALEKERNEL_H
#define ARM_COMPUTE_NEFFTSCALEKERNEL_H

#include "src/core/NEON/INEKernel.h"

#include "arm_compute/core/KernelDescriptors.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for the inverse fft scale kernel. */
class NEFFTScaleKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEFFTScaleKernel";
    }
    /** Constructor */
    NEFFTScaleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFTScaleKernel(const NEFFTScaleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFTScaleKernel &operator=(const NEFFTScaleKernel &) = delete;
    /** Default Move Constructor. */
    NEFFTScaleKernel(NEFFTScaleKernel &&) = default;
    /** Default move assignment operator */
    NEFFTScaleKernel &operator=(NEFFTScaleKernel &&) = default;
    /** Default destructor */
    ~NEFFTScaleKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in,out] input  Source tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[out]    output Destination tensor. Data type supported: same as @p input. Number of channels supported: 1 (real tensor) or 2 (complex tensor).
     * @param[in]     config Kernel configuration
     */
    void configure(ITensor *input, ITensor *output, const FFTScaleKernelInfo &config);
    /** Static function to check if given info will lead to a valid configuration of @ref NEFFTScaleKernel
     *
     * @param[in] input  Source tensor info. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[in] output Destination tensor info. Data type supported: same as @p input. Number of channels supported: 1 (real tensor) or 2 (complex tensor).
     * @param[in] config Kernel configuration
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const FFTScaleKernelInfo &config);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    ITensor *_input;
    ITensor *_output;
    float    _scale;
    bool     _run_in_place;
    bool     _is_conj;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEFFTSCALEKERNEL_H */
