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
#ifndef __ARM_COMPUTE_CLSOBEL7X7_H__
#define __ARM_COMPUTE_CLSOBEL7X7_H__

#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLSobel7x7Kernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute sobel 7x7 filter. This function calls the following OpenCL kernels:
 *
 * -# @ref CLFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref CLSobel7x7HorKernel
 * -# @ref CLSobel7x7VertKernel
 *
 */
class CLSobel7x7 : public IFunction
{
public:
    /** Default Constructor. */
    CLSobel7x7(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialise the function's source, destinations and border mode.
     *
     * @note At least one of output_x or output_y must be not NULL.
     *
     * @param[in,out] input                 Source tensor. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output_x              (optional) Destination for the Sobel 7x7 convolution along the X axis. Data types supported: S32.
     * @param[out]    output_y              (optional) Destination for the Sobel 7x7 convolution along the Y axis. Data types supported: S32.
     * @param[in]     border_mode           Border mode to use for the convolution.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *input, ICLTensor *output_x, ICLTensor *output_y, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

protected:
    CLMemoryGroup        _memory_group;   /**< Function's memory group */
    CLSobel7x7HorKernel  _sobel_hor;      /**< Sobel Horizontal 7x7 kernel */
    CLSobel7x7VertKernel _sobel_vert;     /**< Sobel Vertical 7x7 kernel */
    CLFillBorderKernel   _border_handler; /**< Kernel to handle image borders */
    CLImage              _tmp_x;          /**< Temporary buffer for Sobel X */
    CLImage              _tmp_y;          /**< Temporary buffer for Sobel Y */
};
}
#endif /*__ARM_COMPUTE_CLSOBEL7X7_H__ */
