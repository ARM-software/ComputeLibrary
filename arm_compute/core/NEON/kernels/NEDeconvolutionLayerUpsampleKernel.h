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
#ifndef __ARM_COMPUTE_NEDECONVOLUTIONLAYERKERNEL_H__
#define __ARM_COMPUTE_NEDECONVOLUTIONLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to perform scaling on a tensor */
class NEDeconvolutionLayerUpsampleKernel : public INEKernel
{
public:
    /** Default constructor */
    NEDeconvolutionLayerUpsampleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDeconvolutionLayerUpsampleKernel(const NEDeconvolutionLayerUpsampleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDeconvolutionLayerUpsampleKernel &operator=(const NEDeconvolutionLayerUpsampleKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDeconvolutionLayerUpsampleKernel(NEDeconvolutionLayerUpsampleKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDeconvolutionLayerUpsampleKernel &operator=(NEDeconvolutionLayerUpsampleKernel &&) = default;
    /** Default destructor */
    ~NEDeconvolutionLayerUpsampleKernel() = default;

    /** Initialise the kernel's inputs, output and interpolation policy
     *
     * @param[in]  input   Source tensor. Data types supported: F32.
     * @param[in]  offsets Offset to access the pixel with NEAREST interpolation or the top-left pixel with BILINEAR interpolation in the input tensor. Data type supported: S32.
     * @param[out] output  Destination tensor. Data types supported: F32. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     */
    void configure(const ITensor *input, const ITensor *offsets, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Function to perform scale using nearest interpolation on the given window */
    void scale_nearest(const Window &window);

    const ITensor *_offsets;
    const ITensor *_input;
    ITensor       *_output;
};
} // arm_compute
#endif /*__ARM_COMPUTE_NEDECONVOLUTIONLAYERKERNEL_H__ */
