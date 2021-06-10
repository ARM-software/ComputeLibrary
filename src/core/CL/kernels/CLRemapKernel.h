/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLREMAPKERNEL_H
#define ARM_COMPUTE_CLREMAPKERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform a remap on a tensor */
class CLRemapKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLRemapKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRemapKernel(const CLRemapKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRemapKernel &operator=(const CLRemapKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLRemapKernel(CLRemapKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLRemapKernel &operator=(CLRemapKernel &&) = default;
    /** Initialize the kernel's input, output and border mode.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: U8 (or F16 when layout is NHWC).
     * @param[in]  map_x           Map for X coordinates. Data types supported: F32.
     * @param[in]  map_y           Map for Y coordinates. Data types supported: F32.
     * @param[out] output          Destination tensor. Data types supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. remapping is only performed within the XY-plane.
     * @param[in]  info            RemapInfo struct:
     *                                   - policy                   Interpolation policy to use. Only NEAREST and BILINEAR are supported.
     *                                   - border_mode              Border mode to use on the input tensor. Only CONSTANT and UNDEFINED are supported.
     *                                   - constant_border_value    Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output, RemapInfo info);
    /** Checks if the kernel's input, output and border mode will lead to a valid configuration of @ref CLRemapKernel
     *
     * Similar to @ref CLRemapKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output, RemapInfo info)
     *
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *map_x, const ITensorInfo *map_y, const ITensorInfo *output, RemapInfo info);
    /** Function to set the constant value on fill border kernel depending on type.
     *
     * @param[in] idx                   Index of the kernel argument to set.
     * @param[in] constant_border_value Constant value to use for borders if border_mode is set to CONSTANT.
     */
    template <class T>
    void set_constant_border(unsigned int idx, const PixelValue &constant_border_value);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    const ICLTensor *_map_x;
    const ICLTensor *_map_y;
    DataLayout       _data_layout;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLREMAPKERNEL_H */
