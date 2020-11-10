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
#ifndef ARM_COMPUTE_NEGAUSSIAN5x5_H
#define ARM_COMPUTE_NEGAUSSIAN5x5_H

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
class NEGaussian5x5HorKernel;
class NEGaussian5x5VertKernel;
class NEFillBorderKernel;

/** Basic function to execute gaussian filter 5x5. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEGaussian5x5HorKernel
 * -# @ref NEGaussian5x5VertKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEGaussian5x5 : public IFunction
{
public:
    /** Default constructor
     */
    NEGaussian5x5(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussian5x5(const NEGaussian5x5 &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussian5x5 &operator=(const NEGaussian5x5 &) = delete;
    /** Allow instances of this class to be moved */
    NEGaussian5x5(NEGaussian5x5 &&) = default;
    /** Allow instances of this class to be moved */
    NEGaussian5x5 &operator=(NEGaussian5x5 &&) = default;
    /** Default destructor */
    ~NEGaussian5x5();
    /** Initialise the function's input, output and border mode.
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output                Destination tensor, Data type supported: U8.
     * @param[in]      border_mode           Strategy to use for borders.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

protected:
    MemoryGroup                              _memory_group;   /**< Function memory group */
    std::unique_ptr<NEGaussian5x5HorKernel>  _kernel_hor;     /**< kernel for horizontal pass */
    std::unique_ptr<NEGaussian5x5VertKernel> _kernel_vert;    /**< kernel for vertical pass */
    Tensor                                   _tmp;            /**< temporary buffer for output of horizontal pass */
    std::unique_ptr<NEFillBorderKernel>      _border_handler; /**< kernel to handle tensor borders */
};
}
#endif /*ARM_COMPUTE_NEGAUSSIAN5x5_H */
