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
#ifndef ARM_COMPUTE_NECANNYEDGE_H
#define ARM_COMPUTE_NECANNYEDGE_H

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
class NEGradientKernel;
class NEFillBorderKernel;
class NEEdgeNonMaxSuppressionKernel;
class NEEdgeTraceKernel;

/** Basic function to execute canny edge on Neon. This function calls the following Neon kernels and functions:
 *
 *  -# @ref NEFillBorderKernel (if border_mode == REPLICATE or border_mode == CONSTANT)
 *  -# @ref NESobel3x3 (if gradient_size == 3) or
 *     @ref NESobel5x5 (if gradient_size == 5) or
 *     @ref NESobel7x7 (if gradient_size == 7)
 *  -# @ref NEGradientKernel
 *  -# @ref NEEdgeNonMaxSuppressionKernel
 *  -# @ref NEEdgeTraceKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 *
 */
class NECannyEdge : public IFunction
{
public:
    /** Constructor
     *
     * Initialize Sobel kernel to nullptr.
     *
     * @param[in] memory_manager (Optional) Memory manager.
     */
    NECannyEdge(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECannyEdge(const NECannyEdge &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECannyEdge &operator=(const NECannyEdge &) = delete;
    /** Default destructor */
    ~NECannyEdge();
    /** Initialise the function's source, destination, thresholds, gradient size, normalization type and border mode.
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output                Destination tensor. Data type supported: U8.
     * @param[in]      upper_thr             Upper threhold used for the hysteresis
     * @param[in]      lower_thr             Lower threshold used for the hysteresis.
     * @param[in]      gradient_size         Gradient size (3, 5 or 7)
     * @param[in]      norm_type             Normalization type. If 1, L1-Norm otherwise L2-Norm
     * @param[in]      border_mode           Border mode to use for the convolution.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, int32_t upper_thr, int32_t lower_thr, int32_t gradient_size, int32_t norm_type, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                                    _memory_group;        /**< Function's memory group */
    std::unique_ptr<IFunction>                     _sobel;               /**< Pointer to Sobel kernel */
    std::unique_ptr<NEGradientKernel>              _gradient;            /**< Gradient kernel */
    std::unique_ptr<NEEdgeNonMaxSuppressionKernel> _non_max_suppr;       /**< Non-Maxima suppression kernel */
    std::unique_ptr<NEEdgeTraceKernel>             _edge_trace;          /**< Edge tracing kernel */
    std::unique_ptr<NEFillBorderKernel>            _border_mag_gradient; /**< Fill border on magnitude tensor kernel */
    std::unique_ptr<NEFillBorderKernel>            _border_edge_trace;   /**< Fill border before edge trace */
    Tensor                                         _gx;                  /**< Source tensor - Gx component */
    Tensor                                         _gy;                  /**< Source tensor - Gy component */
    Tensor                                         _magnitude;           /**< Source tensor - Magnitude */
    Tensor                                         _phase;               /**< Source tensor - Phase */
    Tensor                                         _nonmax;              /**< Source tensor - Non-Maxima suppressed */
    ITensor                                       *_output;              /**< Output tensor provided by the user. */
};
}
#endif /* ARM_COMPUTE_NECANNYEDGE_H */
