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
#ifndef __ARM_COMPUTE_GCFILLBORDERKERNEL_H__
#define __ARM_COMPUTE_GCFILLBORDERKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for filling the border of a kernel */
class GCFillBorderKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCFillBorderKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCFillBorderKernel(const GCFillBorderKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCFillBorderKernel &operator=(const GCFillBorderKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCFillBorderKernel(GCFillBorderKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCFillBorderKernel &operator=(GCFillBorderKernel &&) = default;
    /** Default destructor */
    ~GCFillBorderKernel() = default;

    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in,out] tensor                Tensor to process Data types supported: F16/F32.
     * @param[in]     border_size           Size of the border to fill in elements.
     * @param[in]     border_mode           Border mode to use for the convolution.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(const IGCTensor *tensor, BorderSize border_size, BorderMode border_mode, const PixelValue &constant_border_value = PixelValue());

    /** Function to set the constant value on fill border kernel depending on type.
     *
     * @param[in] idx                   Index of the kernel argument to set.
     * @param[in] constant_border_value Constant value to use for borders if border_mode is set to CONSTANT.
     */
    template <class T>
    void set_constant_border(unsigned int idx, const PixelValue &constant_border_value);

    // Inherited methods overridden:
    void run(const Window &window) override;
    bool is_parallelisable() const override;

private:
    const IGCTensor *_tensor;
};
}
#endif /*__ARM_COMPUTE_GCFILLBORDERKERNEL_H__ */
