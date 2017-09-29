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
#ifndef __ARM_COMPUTE_CLREMAPKERNEL_H__
#define __ARM_COMPUTE_CLREMAPKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform a remap on a tensor */
class CLRemapKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLRemapKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLRemapKernel(const CLRemapKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLRemapKernel &operator=(const CLRemapKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLRemapKernel(CLRemapKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLRemapKernel &operator=(CLRemapKernel &&) = default;
    /** Initialize the kernel's input, output and border mode.
     *
     * @param[in]  input            Source tensor. Data types supported: U8.
     * @param[in]  map_x            Map for X coordinates. Data types supported: F32.
     * @param[in]  map_y            Map for Y coordinates. Data types supported: F32.
     * @param[out] output           Destination tensor. Data types supported: U8. All but the lowest two dimensions must be the same size as in the input tensor, i.e. remapping is only performed within the XY-plane.
     * @param[in]  policy           The interpolation type.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output, InterpolationPolicy policy, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    const ICLTensor *_map_x;
    const ICLTensor *_map_y;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLREMAPKERNEL_H__ */
