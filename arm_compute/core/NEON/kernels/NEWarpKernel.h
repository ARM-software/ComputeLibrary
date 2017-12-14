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
#ifndef __ARM_COMPUTE_NEWARPKERNEL_H__
#define __ARM_COMPUTE_NEWARPKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Common interface for warp affine and warp perspective */
class INEWarpKernel : public INEKernel
{
public:
    /** Default constructor */
    INEWarpKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INEWarpKernel(const INEWarpKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INEWarpKernel &operator=(const INEWarpKernel &) = delete;
    /** Allow instances of this class to be moved */
    INEWarpKernel(INEWarpKernel &&) = default;
    /** Allow instances of this class to be moved */
    INEWarpKernel &operator=(INEWarpKernel &&) = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input                 Source tensor. Data type supported: U8.
     * @param[out] output                Destination tensor. Data type supported: U8.
     * @param[in]  matrix                The perspective or affine matrix to use. Must be 2x3 for affine and 3x3 for perspective of type float.
     * @param[in]  border_mode           Strategy to use for borders
     * @param[in]  constant_border_value Constant value used for filling the border.
     */
    virtual void configure(const ITensor *input, ITensor *output, const float *matrix, BorderMode border_mode, uint8_t constant_border_value);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

    // Inherited methods overridden:
    BorderSize border_size() const override;

protected:
    /** function to perform warp affine or warp perspective on the given window when border mode == UNDEFINED
     *
     * @param[in] window Region on which to execute the kernel
     */
    virtual void warp_undefined(const Window &window) = 0;
    /** function to perform warp affine or warp perspective on the given window when border mode == CONSTANT
     *
     * @param[in] window Region on which to execute the kernel
     */
    virtual void warp_constant(const Window &window) = 0;
    /** function to perform warp affine or warp perspective on the given window when border mode == REPLICATE
     *
     * @param[in] window Region on which to execute the kernel
     */
    virtual void warp_replicate(const Window &window) = 0;
    /** Common signature for all the specialised warp functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    void (INEWarpKernel::*_func)(const Window &window);

    const ITensor *_input;                 /**< Input Tensor */
    ITensor       *_output;                /**< Output Tensor */
    uint8_t        _constant_border_value; /**< Constant value used for filling the border. This value is used for those pixels out of the ROI when the border mode is CONSTANT */
    const float   *_matrix;                /**< The affine or perspective matrix. Must be 2x3 for warp affine or 3x3 for warp perspective of type float. */
};

/** Template interface for the kernel to compute warp affine
 *
 */
template <InterpolationPolicy interpolation>
class NEWarpAffineKernel : public INEWarpKernel
{
private:
    // Inherited methods overridden:
    void warp_undefined(const Window &window) override;
    void warp_constant(const Window &window) override;
    void warp_replicate(const Window &window) override;
};

/** Template interface for the kernel to compute warp perspective
 *
 */
template <InterpolationPolicy interpolation>
class NEWarpPerspectiveKernel : public INEWarpKernel
{
private:
    // Inherited methods overridden:
    void warp_undefined(const Window &window) override;
    void warp_constant(const Window &window) override;
    void warp_replicate(const Window &window) override;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEWARPKERNEL_H__ */
