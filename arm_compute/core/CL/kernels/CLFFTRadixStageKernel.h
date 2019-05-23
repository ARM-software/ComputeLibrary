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
#ifndef __ARM_COMPUTE_CLFFTRADIXSTAGEKERNEL_H__
#define __ARM_COMPUTE_CLFFTRADIXSTAGEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

#include "arm_compute/core/KernelDescriptors.h"

#include <set>

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Interface for the FFT radix stage kernel. */
class CLFFTRadixStageKernel : public ICLKernel
{
public:
    /** Constructor */
    CLFFTRadixStageKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFFTRadixStageKernel(const CLFFTRadixStageKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFFTRadixStageKernel &operator=(const CLFFTRadixStageKernel &) = delete;
    /** Default Move Constructor. */
    CLFFTRadixStageKernel(CLFFTRadixStageKernel &&) = default;
    /** Default move assignment operator */
    CLFFTRadixStageKernel &operator=(CLFFTRadixStageKernel &&) = default;
    /** Default destructor */
    ~CLFFTRadixStageKernel() = default;
    /** Set the input and output tensors.
     *
     * @note If the output tensor is nullptr, the FFT will be performed in-place
     *
     * @param[in,out] input  Source tensor. Data types supported: F32.
     * @param[out]    output Destination tensor. Can be nullptr. Data type supported: same as @p input
     * @param[in]     config FFT descriptor metadata.
     */
    void configure(ICLTensor *input, ICLTensor *output, const FFTRadixStageKernelInfo &config);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFFTRadixStageKernel
     *
     * @param[in] input  Source tensor info. Data types supported: F32.
     * @param[in] output Destination tensor info. Can be nullptr. Data type supported: same as @p input
     * @param[in] config FFT descriptor metadata.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const FFTRadixStageKernelInfo &config);
    /** Returns the radix that are support by the FFT kernel
     *
     * @return A set of supported radix
     */
    static std::set<unsigned int> supported_radix();

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    ICLTensor *_input;
    ICLTensor *_output;
    bool       _run_in_place;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLFFTRADIXSTAGEKERNEL_H__ */
