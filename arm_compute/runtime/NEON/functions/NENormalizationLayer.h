/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NENORMALIZATIONLAYER_H__
#define __ARM_COMPUTE_NENORMALIZATIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NENormalizationLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEPixelWiseMultiplicationKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class ITensor;

/** Basic function to compute a normalization layer. This function calls the following NEON kernels:
 *
 * -# @ref NEPixelWiseMultiplicationKernel
 * -# @ref NEFillBorderKernel
 * -# @ref NENormalizationLayerKernel
 *
 */
class NENormalizationLayer : public IFunction
{
public:
    /** Default constructor */
    NENormalizationLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                       and an optional 4th dimension for batch of inputs. Data type supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[out] output    Destination with the same dimensions, data type, data layout and number of channels of  @p input
     * @param[in]  norm_info Normalization layer information like the normalization type, normalization size and other parameters.
     */
    void configure(const ITensor *input, ITensor *output, const NormalizationLayerInfo &norm_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NENormalizationLayer
     *
     * @param[in] input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                      and an optional 4th dimension for batch of inputs. Data type supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[in] output    Destination with the same dimensions, data type, data layout and number of channels of  @p input
     * @param[in] norm_info Normalization layer information like the normalization type, normalization size and other parameters.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const NormalizationLayerInfo &norm_info);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                     _memory_group;    /**< Function memory group */
    NENormalizationLayerKernel      _norm_kernel;     /**< Normalization layer kernel */
    NEPixelWiseMultiplicationKernel _multiply_kernel; /**< Pixel multiplication kernel */
    NEFillBorderKernel              _border_handler;  /**< Kernel to handle  borders */
    Tensor                          _input_squared;   /**< The intermediate buffer which stores results of squaring input */
};
}
#endif /* __ARM_COMPUTE_NENORMALIZATIONLAYER_H__ */
