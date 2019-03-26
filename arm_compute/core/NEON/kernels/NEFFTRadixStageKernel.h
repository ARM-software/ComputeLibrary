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
#ifndef __ARM_COMPUTE_NEFFTRADIXSTAGEKERNEL_H__
#define __ARM_COMPUTE_NEFFTRADIXSTAGEKERNEL_H__

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/NEON/INEKernel.h"

#include <arm_neon.h>
#include <set>

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for the FFT kernel. */
class NEFFTRadixStageKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEFFTRadixStageKernel";
    }
    /** Constructor */
    NEFFTRadixStageKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFTRadixStageKernel(const NEFFTRadixStageKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFTRadixStageKernel &operator=(const NEFFTRadixStageKernel &) = delete;
    /** Default Move Constructor. */
    NEFFTRadixStageKernel(NEFFTRadixStageKernel &&) = default;
    /** Default move assignment operator */
    NEFFTRadixStageKernel &operator=(NEFFTRadixStageKernel &&) = default;
    /** Default destructor */
    ~NEFFTRadixStageKernel() = default;
    /** Set the input and output tensors.
     *
     * @note If the output tensor is nullptr, the FFT will be performed in-place
     *
     * @param[in,out] input  Source tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[out]    output Destination tensor. Data type supported: same as @p input. Number of channels supported: same as @p input.
     * @param[in]     config FFT descriptor metadata.
     */
    void configure(ITensor *input, ITensor *output, const FFTRadixStageKernelInfo &config);
    /** Static function to check if given info will lead to a valid configuration of @ref NEFFTRadixStageKernel
     *
     * @param[in] input  Source tensor info. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[in] output Destination tensor info. Data type supported: same as @p input. Number of channels supported: same as @p input.
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
    void run(const Window &window, const ThreadInfo &info) override;

private:
    ITensor     *_input;
    ITensor     *_output;
    bool         _run_in_place;
    unsigned int _Nx;
    unsigned int _axis;
    unsigned int _radix;

    void set_radix_stage_axis0(const FFTRadixStageKernelInfo &config);
    void set_radix_stage_axis1(const FFTRadixStageKernelInfo &config);

    using FFTFunctionPointerAxis0 = std::function<void(float *, float *, unsigned int, unsigned int, const float32x2_t &, unsigned int)>;
    using FFTFunctionPointerAxis1 = std::function<void(float *, float *, unsigned int, unsigned int, const float32x2_t &, unsigned int, unsigned int)>;

    FFTFunctionPointerAxis0 _func_0;
    FFTFunctionPointerAxis1 _func_1;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEFFTRADIXSTAGEKERNEL_H__ */
