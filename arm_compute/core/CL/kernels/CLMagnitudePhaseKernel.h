/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLMAGNITUDEPHASEKERNEL_H__
#define __ARM_COMPUTE_CLMAGNITUDEPHASEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Template interface for the kernel to compute magnitude and phase.
 *
 */
class CLMagnitudePhaseKernel : public ICLKernel
{
public:
    /** Default constructor. */
    CLMagnitudePhaseKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLMagnitudePhaseKernel(const CLMagnitudePhaseKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLMagnitudePhaseKernel &operator=(const CLMagnitudePhaseKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLMagnitudePhaseKernel(CLMagnitudePhaseKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLMagnitudePhaseKernel &operator=(CLMagnitudePhaseKernel &&) = default;
    /** Initialise the kernel's input, output.
     *
     * @note At least one of output1 or output2 must be set.
     *
     * @param[in]  gx         The input gradient X tensor. Data types supported: S16.
     * @param[in]  gy         The input gradient Y tensor. Data types supported: S16.
     * @param[out] magnitude  (Optional) The output tensor - Magnitude. Data types supported: S16.
     * @param[out] phase      (Optional) The output tensor - Phase. Data types supported: U8.
     * @param[in]  mag_type   (Optional) Magnitude calculation type. Default: L2NORM.
     * @param[in]  phase_type (Optional) Phase calculation type. Default: SIGNED.
     */
    void configure(const ICLTensor *gx, const ICLTensor *gy, ICLTensor *magnitude, ICLTensor *phase,
                   MagnitudeType mag_type = MagnitudeType::L2NORM, PhaseType phase_type = PhaseType::SIGNED);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_gx;        /**< Input gradient X. */
    const ICLTensor *_gy;        /**< Input gradient Y. */
    ICLTensor       *_magnitude; /**< Output - Magnitude. */
    ICLTensor       *_phase;     /**< Output - Phase. */
    bool             _run_mag;   /**< Calculate magnitude ? */
    bool             _run_phase; /**< Calculate phase ? */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLMAGNITUDEPHASEKERNEL_H__ */
