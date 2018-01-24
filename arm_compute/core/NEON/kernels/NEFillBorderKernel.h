/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEFILLBORDERKERNEL_H__
#define __ARM_COMPUTE_NEFILLBORDERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to fill borders */
class NEFillBorderKernel : public INEKernel
{
public:
    /** Default Constructor */
    NEFillBorderKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFillBorderKernel(const NEFillBorderKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFillBorderKernel &operator=(const NEFillBorderKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEFillBorderKernel(NEFillBorderKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEFillBorderKernel &operator=(NEFillBorderKernel &&) = default;
    /** Default destructor */
    ~NEFillBorderKernel() = default;

    /** Initialise the function.
     *
     * @note This kernel fills the borders within the XY-planes.
     *
     * @param[in,out] tensor                Tensor to process. Data types supported: U8/S8/QS8/QASYMM8/QS16/S16/S32/F32.
     * @param[in]     border_size           Size of the border to fill in elements.
     * @param[in]     border_mode           Border mode to use for the convolution.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    void configure(ITensor *tensor, BorderSize border_size, BorderMode border_mode, const PixelValue &constant_border_value = PixelValue());

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    template <typename T>
    void fill_replicate_single_channel(const Window &window);
    template <typename T>
    void fill_constant_value_single_channel(const Window &window);

    ITensor   *_tensor;
    BorderSize _border_size;
    BorderMode _mode;
    PixelValue _constant_border_value;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEFILLBORDERKERNEL_H__ */
