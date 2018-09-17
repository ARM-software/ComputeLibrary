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
#ifndef __ARM_COMPUTE_CLDERIVATIVEKERNEL_H__
#define __ARM_COMPUTE_CLDERIVATIVEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the derivative kernel. */
class CLDerivativeKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLDerivativeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLDerivativeKernel(const CLDerivativeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLDerivativeKernel &operator=(const CLDerivativeKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLDerivativeKernel(CLDerivativeKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLDerivativeKernel &operator=(CLDerivativeKernel &&) = default;
    /** Default destructor */
    ~CLDerivativeKernel() = default;
    /** Initialise the kernel's sources, destination and border
     *
     * @note At least one of output_x or output_y must be set
     *
     * @param[in]  input            Source tensor. Data types supported: U8.
     * @param[out] output_x         (Optional) Destination tensor for the X gradient, Data types supported: S16.
     * @param[out] output_y         (Optional) Destination tensor for the Y gradient, Data types supported: S16.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output_x, ICLTensor *output_y, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_input;            /**< Input tensor */
    ICLTensor       *_output_x;         /**< Output tensor - Derivate along the X direction */
    ICLTensor       *_output_y;         /**< Output tensor - Derivate along the Y direction */
    bool             _run_derivative_x; /**< Do we need to run Derivative X ? */
    bool             _run_derivative_y; /**< Do we need to run Derivative Y ? */
};
}
#endif /*__ARM_COMPUTE_CLDERIVATIVEKERNEL_H__ */
