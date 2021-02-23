/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NESOBEL5x5_H
#define ARM_COMPUTE_NESOBEL5x5_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ITensor;
class NESobel5x5HorKernel;
class NESobel5x5VertKernel;
class NEFillBorderKernel;

/** Basic function to execute sobel 5x5 filter. This function calls the following Neon kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NESobel5x5HorKernel
 * -# @ref NESobel5x5VertKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NESobel5x5 : public IFunction
{
public:
    /** Default constructor */
    NESobel5x5(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel5x5(const NESobel5x5 &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel5x5 &operator=(const NESobel5x5 &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NESobel5x5(NESobel5x5 &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NESobel5x5 &operator=(NESobel5x5 &&) = delete;
    /** Default destructor */
    ~NESobel5x5();
    /** Initialise the function's source, destinations and border mode.
     *
     * @note At least one of output_x or output_y must be not NULL.
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output_x              (optional) Destination for the Sobel 5x5 convolution along the X axis. Data type supported: S16.
     * @param[out]     output_y              (optional) Destination for the Sobel 5x5 convolution along the Y axis. Data type supported: S16.
     * @param[in]      border_mode           Border mode to use for the convolution.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    void configure(ITensor *input, ITensor *output_x, ITensor *output_y, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

protected:
    MemoryGroup                           _memory_group;   /**< Function memory group */
    std::unique_ptr<NESobel5x5HorKernel>  _sobel_hor;      /**< Sobel Horizontal 5x5 kernel */
    std::unique_ptr<NESobel5x5VertKernel> _sobel_vert;     /**< Sobel Vertical 5x5 kernel */
    Tensor                                _tmp_x;          /**< Temporary buffer for Sobel X */
    Tensor                                _tmp_y;          /**< Temporary buffer for Sobel Y */
    std::unique_ptr<NEFillBorderKernel>   _border_handler; /**< Kernel to handle tensor borders */
};
}
#endif /*ARM_COMPUTE_NESOBEL5x5_H */
