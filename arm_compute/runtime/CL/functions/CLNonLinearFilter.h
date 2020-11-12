/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLNONLINEARFILTER_H
#define ARM_COMPUTE_CLNONLINEARFILTER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;

/** Basic function to execute non linear filter. This function calls the following OpenCL kernels:
 *
 * -# @ref CLFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref CLNonLinearFilterKernel
 *
 * @note Supported mask dimensions squares of sizes 3, 5
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class CLNonLinearFilter : public ICLSimpleFunction
{
public:
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor. Data types supported: U8
     * @param[in]     function              Non linear function to perform
     * @param[in]     mask_size             Mask size. Supported sizes: 3, 5
     * @param[in]     pattern               Mask pattern
     * @param[in]     mask                  The given mask. Will be used only if pattern is specified to PATTERN_OTHER
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *input, ICLTensor *output, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask,
                   BorderMode border_mode, uint8_t constant_border_value = 0);
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in]     compile_context       The compile context to be used.
     * @param[in,out] input                 Source tensor. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor. Data types supported: U8
     * @param[in]     function              Non linear function to perform
     * @param[in]     mask_size             Mask size. Supported sizes: 3, 5
     * @param[in]     pattern               Mask pattern
     * @param[in]     mask                  The given mask. Will be used only if pattern is specified to PATTERN_OTHER
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask,
                   BorderMode border_mode, uint8_t constant_border_value = 0);
};
}
#endif /*ARM_COMPUTE_CLNONLINEARFILTER_H */
