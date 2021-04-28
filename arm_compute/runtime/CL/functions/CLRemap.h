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
#ifndef ARM_COMPUTE_CLREMAP_H
#define ARM_COMPUTE_CLREMAP_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;

/** Basic function to execute remap. This function calls the following OpenCL kernels:
 *
 * -# @ref CLFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref CLRemapKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class CLRemap : public ICLSimpleFunction
{
public:
    /** Initialise the function's sources, destination, interpolation policy and border mode.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0   |src1   |src2   |dst    |
     * |:------|:------|:------|:------|
     * |U8     |F32    |F32    |U 8    |
     *
     * @param[in,out] input                 Source tensor. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[in]     map_x                 Map for X coords. Data types supported: F32.
     * @param[in]     map_y                 Map for Y coords. Data types supported: F32.
     * @param[out]    output                Output tensor. Data types supported: U8.
     * @param[in]     policy                Interpolation policy to use. Only NEAREST and BILINEAR are supported.
     * @param[in]     border_mode           Border mode to use on the input tensor.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    void configure(ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output,
                   InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value = 0);
    /** Initialise the function's sources, destination, interpolation policy and border mode.
     *
     * @param[in]     compile_context       The compile context to be used.
     * @param[in,out] input                 Source tensor. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[in]     map_x                 Map for X coords. Data types supported: F32.
     * @param[in]     map_y                 Map for Y coords. Data types supported: F32.
     * @param[out]    output                Output tensor. Data types supported: U8.
     * @param[in]     policy                Interpolation policy to use. Only NEAREST and BILINEAR are supported.
     * @param[in]     border_mode           Border mode to use on the input tensor.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output,
                   InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value = 0);
};
}
#endif /*ARM_COMPUTE_CLREMAP_H */
