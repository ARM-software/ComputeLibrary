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
#ifndef __ARM_COMPUTE_NEREMAPKERNEL_H__
#define __ARM_COMPUTE_NEREMAPKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to perform a remap on a tensor */
class NERemapKernel : public INEKernel
{
public:
    /** Default constructor */
    NERemapKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NERemapKernel(const NERemapKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NERemapKernel &operator=(const NERemapKernel &) = delete;
    /** Allow instances of this class to be moved */
    NERemapKernel(NERemapKernel &&) = default;
    /** Allow instances of this class to be moved */
    NERemapKernel &operator=(NERemapKernel &&) = default;
    /** Default destructor */
    ~NERemapKernel() = default;

    /** Initialize the kernel's input, output and border mode.
     *
     * @param[in]  input  Source tensor. Data type supported: U8.
     * @param[in]  map_x  Map for X coordinates. Data type supported: F32.
     * @param[in]  map_y  Map for Y coordinates. Data type supported: F32.
     * @param[out] output Destination tensor. Data types supported: U8. All but the lowest two dimensions must be the same size as in the input tensor, i.e. remapping is only performed within the XY-plane.
     * @param[in]  policy The interpolation type.
     */
    void configure(const ITensor *input, const ITensor *map_x, const ITensor *map_y, ITensor *output, InterpolationPolicy policy);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** function to perform nearest interpolation on the given window */
    void remap_nearest(const Window &window);
    /** function to perform bilinear interpolation on the given window */
    void remap_bilinear(const Window &window);
    /** Remap function to use for the particular interpolation type passed to configure() */
    void (NERemapKernel::*_func)(const Window &window);

    const ITensor *_input;  /**< Input image */
    ITensor       *_output; /**< Output image */
    const ITensor *_map_x;  /**< Input remap x coordinates */
    const ITensor *_map_y;  /**< Input remap y coordinates */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEREMAPKERNEL_H__ */
