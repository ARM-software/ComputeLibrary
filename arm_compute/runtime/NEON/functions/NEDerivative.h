/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEDERIVATIVE_H
#define ARM_COMPUTE_NEDERIVATIVE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class NEDerivativeKernel;
class NEFillBorderKernel;

/** Basic function to execute first order derivative operator. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEDerivativeKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEDerivative : public IFunction
{
public:
    /** Default constructor */
    NEDerivative();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDerivative(const NEDerivative &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDerivative &operator=(const NEDerivative &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEDerivative(NEDerivative &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEDerivative &operator=(NEDerivative &&) = delete;
    /** Default destructor */
    ~NEDerivative();
    /** Initialise the function's source, destinations and border mode.
     *
     * @note At least one of output_x or output_y must be not NULL.
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output_x              (optional) Destination tensor. Derivative along the X direction. Data type supported: S16.
     * @param[out]     output_y              (optional) Destination tensor. Derivative along the Y direction. Data type supported: S16.
     * @param[in]      border_mode           Border mode to use
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    void configure(ITensor *input, ITensor *output_x, ITensor *output_y, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEDerivativeKernel> _kernel;         /**< Derivative kernel */
    std::unique_ptr<NEFillBorderKernel> _border_handler; /**< Kernel to handle tensor borders */
};
}
#endif /* ARM_COMPUTE_NEDERIVATIVE_H */
