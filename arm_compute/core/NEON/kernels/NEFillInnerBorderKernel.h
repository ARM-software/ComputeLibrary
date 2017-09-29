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
#ifndef __ARM_COMPUTE_NEFILLINNERBORDERKERNEL_H__
#define __ARM_COMPUTE_NEFILLINNERBORDERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to fill the interior borders */
class NEFillInnerBorderKernel : public INEKernel
{
public:
    /** Default constructor */
    NEFillInnerBorderKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFillInnerBorderKernel(const NEFillInnerBorderKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFillInnerBorderKernel &operator=(const NEFillInnerBorderKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEFillInnerBorderKernel(NEFillInnerBorderKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEFillInnerBorderKernel &operator=(NEFillInnerBorderKernel &&) = default;
    /** Default destructor */
    ~NEFillInnerBorderKernel() = default;

    /** Initialise the function.
     *
     * @note This kernel fills the borders within the XY-planes.
     *
     * @param[in,out] input                 Tensor to process. Data types supported: U8/QS8/S16/S32/F32.
     * @param[in]     border_size           Size of the border to fill in elements.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    void configure(ITensor *input, BorderSize border_size, const PixelValue &constant_border_value = PixelValue());

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    template <typename T>
    void fill_value_single_channel(const Window &window);

    ITensor   *_tensor;
    BorderSize _border_size;
    PixelValue _constant_border_value;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEFILLINNERBORDERKERNEL_H__ */
