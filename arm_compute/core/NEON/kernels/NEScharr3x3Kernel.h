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
#ifndef __ARM_COMPUTE_NESCHARR3x3KERNEL_H__
#define __ARM_COMPUTE_NESCHARR3x3KERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to run a 3x3 Scharr filter on a tensor.
 *
* @f[
*      \mathbf{G}_x=\begin{vmatrix}
*      -3 & 0 & +3\\
*      -10& 0 & +10\\
*      -3 & 0 & +3
*      \end{vmatrix}
* @f]
*/
class NEScharr3x3Kernel : public INEKernel
{
public:
    /** Default constructor */
    NEScharr3x3Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEScharr3x3Kernel(const NEScharr3x3Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEScharr3x3Kernel &operator=(const NEScharr3x3Kernel &) = delete;
    /** Allow instances of this class to be moved */
    NEScharr3x3Kernel(NEScharr3x3Kernel &&) = default;
    /** Allow instances of this class to be moved */
    NEScharr3x3Kernel &operator=(NEScharr3x3Kernel &&) = default;
    /** Default destructor */
    ~NEScharr3x3Kernel() = default;

    /** Initialise the kernel's source, destination and border.
     *
     * @note At least one of output_x or output_y must be set.
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
    bool           _run_scharr_x; /**< Do we need to run Scharr X ? */
    bool           _run_scharr_y; /**< Do we need to run Scharr Y ? */
    const ITensor *_input;        /**< Input tensor */
    ITensor       *_output_x;     /**< Output tensor for scharr X */
    ITensor       *_output_y;     /**< Output tensor for scharr Y */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NESCHARR3x3KERNEL_H__ */
