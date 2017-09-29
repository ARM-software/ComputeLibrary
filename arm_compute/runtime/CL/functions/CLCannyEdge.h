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
#ifndef __ARM_COMPUTE_CLCANNYEDGE_H__
#define __ARM_COMPUTE_CLCANNYEDGE_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLCannyEdgeKernel.h"
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute canny edge on OpenCL. This function calls the following OpenCL kernels and functions:
 *
 * -# @ref CLFillBorderKernel (if border_mode == REPLICATE or border_mode == CONSTANT)
 * -# @ref CLSobel3x3 (if gradient_size == 3) or @ref CLSobel5x5 (if gradient_size == 5) or @ref CLSobel7x7 (if gradient_size == 7)
 * -# @ref CLGradientKernel
 * -# @ref CLEdgeNonMaxSuppressionKernel
 * -# @ref CLEdgeTraceKernel
 *
 */
class CLCannyEdge : public IFunction
{
public:
    /** Constructor */
    CLCannyEdge(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialise the function's source, destination, thresholds, gradient size, normalization type and border mode.
     *
     * @param[in,out] input                 Source tensor. Data types supported: U8. (Written to only for border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor. Data types supported: U8.
     * @param[in]     upper_thr             Upper threshold used for the hysteresis.
     * @param[in]     lower_thr             Lower threshold used for the hysteresis.
     * @param[in]     gradient_size         Gradient size (3, 5 or 7).
     * @param[in]     norm_type             Normalization type. if 1, L1-Norm otherwise L2-Norm.
     * @param[in]     border_mode           Border mode to use for the convolution.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *input, ICLTensor *output, int32_t upper_thr, int32_t lower_thr, int32_t gradient_size, int32_t norm_type,
                   BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    virtual void run() override;

private:
    CLMemoryGroup                 _memory_group;                                    /**< Function's memory group */
    std::unique_ptr<IFunction>    _sobel;                                           /**< Pointer to Sobel kernel. */
    CLGradientKernel              _gradient;                                        /**< Gradient kernel. */
    CLFillBorderKernel            _border_mag_gradient;                             /**< Fill border on magnitude tensor kernel */
    CLEdgeNonMaxSuppressionKernel _non_max_suppr;                                   /**< Non-Maxima suppression kernel. */
    CLEdgeTraceKernel             _edge_trace;                                      /**< Edge tracing kernel. */
    CLImage                       _gx;                                              /**< Source tensor - Gx component. */
    CLImage                       _gy;                                              /**< Source tensor - Gy component. */
    CLImage                       _mag;                                             /**< Source tensor - Magnitude. */
    CLImage                       _phase;                                           /**< Source tensor - Phase. */
    CLImage                       _nonmax;                                          /**< Source tensor - Non-Maxima suppressed. */
    CLImage                       _visited, _recorded, _l1_list_counter, _l1_stack; /**< Temporary tensors */
};
}

#endif /* __ARM_COMPUTE_CLCANNYEDGE_H__ */
