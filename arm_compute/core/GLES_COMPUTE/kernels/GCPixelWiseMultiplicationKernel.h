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
#ifndef __ARM_COMPUTE_GCPIXELWISEMULTIPLICATIONKERNEL_H__
#define __ARM_COMPUTE_GCPIXELWISEMULTIPLICATIONKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the pixelwise multiplication kernel.
 *
 */
class GCPixelWiseMultiplicationKernel : public IGCKernel
{
public:
    /** Default constructor.*/
    GCPixelWiseMultiplicationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    GCPixelWiseMultiplicationKernel(const GCPixelWiseMultiplicationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    GCPixelWiseMultiplicationKernel &operator=(const GCPixelWiseMultiplicationKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCPixelWiseMultiplicationKernel(GCPixelWiseMultiplicationKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCPixelWiseMultiplicationKernel &operator=(GCPixelWiseMultiplicationKernel &&) = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input1 An input tensor. Data types supported: F32.
     * @param[in]  input2 An input tensor. Data types supported: same as @p input1.
     * @param[out] output The output tensor, Data types supported: same as @p input1.
     * @param[in]  scale  Scale to apply after multiplication.
     *                             Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     */
    void configure(const IGCTensor *input1, const IGCTensor *input2, IGCTensor *output, float scale);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input1;
    const IGCTensor *_input2;
    IGCTensor       *_output;
};
}

#endif /*__ARM_COMPUTE_GCPIXELWISEMULTIPLICATIONKERNEL_H__ */
