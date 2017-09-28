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
#ifndef __ARM_COMPUTE_CLNONLINEARFILTERKERNEL_H__
#define __ARM_COMPUTE_CLNONLINEARFILTERKERNEL_H__

#include "arm_compute/core/CL/ICLSimple2DKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to apply a non-linear filter */
class CLNonLinearFilterKernel : public ICLSimple2DKernel
{
public:
    /** Default constructor */
    CLNonLinearFilterKernel();
    /** Set the source, destination and border mode of the kernel
     *
     * @param[in]  input            Source tensor. Data types supported: U8
     * @param[out] output           Destination tensor. Data types supported: U8
     * @param[in]  function         Non linear function to perform
     * @param[in]  mask_size        Mask size. Supported sizes: 3, 5
     * @param[in]  pattern          Mask pattern
     * @param[in]  mask             The given mask. Will be used only if pattern is specified to PATTERN_OTHER
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output, NonLinearFilterFunction function,
                   unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask,
                   bool border_undefined);

    // Inherited methods overridden:
    BorderSize border_size() const override;

private:
    BorderSize _border_size; /**< Border size */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLNONLINEARFILTERKERNEL_H__ */
