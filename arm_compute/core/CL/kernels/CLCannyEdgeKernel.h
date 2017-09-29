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
#ifndef __ARM_COMPUTE_CLCANNYEDGEKERNEL_H__
#define __ARM_COMPUTE_CLCANNYEDGEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform Gradient computation.
 */
class CLGradientKernel : public ICLKernel
{
public:
    /** Constructor */
    CLGradientKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLGradientKernel(const CLGradientKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLGradientKernel &operator=(const CLGradientKernel &) = delete;
    /** Initialise the kernel's sources, destinations and border mode.
     *
     * @note gx, gy and mag must all be the same size (either 16 or 32).
     *
     * @param[in]  gx        Source tensor - Gx component. Data types supported: S16/S32.
     * @param[in]  gy        Source tensor - Gy component. Data types supported: Same as gx.
     * @param[out] magnitude Destination tensor - Magnitude. Data types supported: U16/U32. Must match the pixel size of gx, gy.
     * @param[out] phase     Destination tensor - Quantized phase. Data types supported: U8.
     * @param[in]  norm_type Normalization type. if 1, L1-Norm otherwise L2-Norm.
     */
    void configure(const ICLTensor *gx, const ICLTensor *gy, ICLTensor *magnitude, ICLTensor *phase, int32_t norm_type);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_gx;        /**< Source tensor - Gx component */
    const ICLTensor *_gy;        /**< Source tensor - Gy component */
    ICLTensor       *_magnitude; /**< Destination tensor - Magnitude */
    ICLTensor       *_phase;     /**< Destination tensor - Quantized phase */
};

/** OpenCL kernel to perform Non-Maxima suppression for Canny Edge.
 *
 * @note This kernel is meant to be used alongside CannyEdge and performs a non-maxima suppression using magnitude and phase of input
 *       to characterize points as possible edges. The output buffer needs to be cleared before this kernel is executed.
 *
 * @note Hysteresis is computed in @ref CLEdgeTraceKernel
 */
class CLEdgeNonMaxSuppressionKernel : public ICLKernel
{
public:
    /** Constructor */
    CLEdgeNonMaxSuppressionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLEdgeNonMaxSuppressionKernel(const CLEdgeNonMaxSuppressionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLEdgeNonMaxSuppressionKernel &operator=(const CLEdgeNonMaxSuppressionKernel &) = delete;
    /** Initialise the kernel's sources, destination and border mode.
     *
     * @param[in]  magnitude        Source tensor - Magnitude. Data types supported: U16/U32.
     * @param[in]  phase            Source tensor - Quantized phase. Data types supported: U8.
     * @param[out] output           Destination tensor. Data types supported: U16/U32.
     * @param[in]  lower_thr        Lower threshold.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *magnitude, const ICLTensor *phase, ICLTensor *output, int32_t lower_thr, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_magnitude; /**< Source tensor - Magnitude. */
    const ICLTensor *_phase;     /**< Source tensor - Quantized phase. */
    ICLTensor       *_output;    /**< Destination tensor. */
};

/** OpenCL kernel to perform Edge tracing.
 */
class CLEdgeTraceKernel : public ICLKernel
{
public:
    /** Constructor */
    CLEdgeTraceKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLEdgeTraceKernel(const CLEdgeTraceKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLEdgeTraceKernel &operator=(const CLEdgeTraceKernel &) = delete;
    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in]     input            Source tensor. Data types supported: U8.
     * @param[out]    output           Destination tensor. Data types supported: U8.
     * @param[in]     upper_thr        Upper threshold used for the hysteresis
     * @param[in]     lower_thr        Lower threshold used for the hysteresis
     * @param[in,out] visited          Tensor for keeping the visited pixels. Data types supported: U32.
     *                                 Expected to be initialized to 0 before each run.
     * @param[in,out] recorded         Tensor for keeping the recorded pixels. Data types supported: U32
     *                                 Expected to be initialized to 0 before each run.
     * @param[in,out] l1_stack         Tensor with the L1 stack for each pixel. Data types supported: S32.
     *                                 Expected to be initialized to 0 before each run.
     * @param[in,out] l1_stack_counter Tensor for counting the elements in the L1 stack of each pixel. Data types supported: U8.
     *                                              Expected to be initialized to 0 before each run.
     */
    void configure(const ICLTensor *input, ICLTensor *output, int32_t upper_thr, int32_t lower_thr,
                   ICLTensor *visited, ICLTensor *recorded, ICLTensor *l1_stack, ICLTensor *l1_stack_counter);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;            /**< Source tensor. */
    ICLTensor       *_output;           /**< Destination tensor. */
    int32_t          _lower_thr;        /**< Lower threshold used for the hysteresis. */
    int32_t          _upper_thr;        /**< Upper threshold used for the hysteresis. */
    ICLTensor       *_visited;          /**< Marks visited elements */
    ICLTensor       *_recorded;         /**< Marks recorded elements */
    ICLTensor       *_l1_stack;         /**< L1 hysteris stack */
    ICLTensor       *_l1_stack_counter; /**< L1 hysteris stack counter */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLCANNYEDGEKERNEL_H__ */
