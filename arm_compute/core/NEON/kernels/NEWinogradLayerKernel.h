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
#ifndef __ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__
#define __ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

#include "arm_compute/core/NEON/kernels/winograd/winograd_shim_nchw.hpp"

namespace arm_compute
{
class ITensor;

class NEWinogradLayerKernel : public INEKernel
{
public:
    using Winograd3x3F32 = winograd_shim_nchw::Winograd2x2_3x3GEMM<float, float>;

    /** Constructor */
    NEWinogradLayerKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerKernel(const NEWinogradLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerKernel &operator=(const NEWinogradLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEWinogradLayerKernel(NEWinogradLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEWinogradLayerKernel &operator=(NEWinogradLayerKernel &&) = default;

    virtual ~NEWinogradLayerKernel() = default;

    /** Initialise the kernel
     *
     * @param[in,out] output    Output tensor to store the result of matrix multiplication.
     * @param[in]     convolver A pointer to the winograd convolver, this object must have been configured and is ready to execute 16 GEMMS .
     */
    void configure(ITensor *output, Winograd3x3F32 *convolver);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

protected:
    Winograd3x3F32 *_convolver;
    ITensor        *_output;
};

} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__*/
