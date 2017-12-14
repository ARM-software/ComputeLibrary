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
#ifndef __ARM_COMPUTE_NEDERIVATIVEKERNEL_H__
#define __ARM_COMPUTE_NEDERIVATIVEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to run the derivative along the X/Y directions on a tensor.
 *
 */
class NEDerivativeKernel : public INEKernel
{
public:
    /** Default constructor */
    NEDerivativeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDerivativeKernel(const NEDerivativeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDerivativeKernel &operator=(const NEDerivativeKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDerivativeKernel(NEDerivativeKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDerivativeKernel &operator=(NEDerivativeKernel &&) = default;
    /** Initialise the kernel's sources, destination and border
     *
     * @note At least one of output_x or output_y must be set
     *
     * @param[in]  input            Source tensor. Data type supported: U8.
     * @param[out] output_x         (Optional) Destination tensor for the X gradient. Data type supported: S16.
     * @param[out] output_y         (Optional) Destination tensor for the Y gradient. Data type supported: S16.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output_x, ITensor *output_y, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Function to perform derivative along the X direction on the given window
     *
     * @param[in] window Region on which to execute the kernel
     */
    void derivative_x(const Window &window);
    /** Function to perform derivative along the Y direction on the given window
     *
     * @param[in] window Region on which to execute the kernel
     */
    void derivative_y(const Window &window);
    /** Function to perform derivative along the X and Y direction on the given window
     *
     * @param[in] window Region on which to execute the kernel
     */
    void derivative_xy(const Window &window);
    /** Common signature for all the specialised derivative functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using DerivativeFunction = void (NEDerivativeKernel::*)(const Window &window);
    /** Derivative function to use for the particular tensor types passed to configure() */
    DerivativeFunction _func;

private:
    const ITensor *_input;    /**< Input tensor */
    ITensor       *_output_x; /**< Output tensor - Derivate along the X direction */
    ITensor       *_output_y; /**< Output tensor - Derivate along the Y direction */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEDERIVATIVEKERNEL_H__ */
