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
#ifndef ARM_COMPUTE_NECANNYEDGEKERNEL_H
#define ARM_COMPUTE_NECANNYEDGEKERNEL_H

#include "src/core/NEON/INEKernel.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Computes magnitude and quantised phase from inputs gradients. */
class NEGradientKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEGradientKernel";
    }
    /** Default constructor */
    NEGradientKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGradientKernel(const NEGradientKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGradientKernel &operator=(const NEGradientKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGradientKernel(NEGradientKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGradientKernel &operator=(NEGradientKernel &&) = default;
    /** Default destructor */
    ~NEGradientKernel();

    /** Initialise the kernel's sources, destinations and border mode.
     *
     * @note gx, gy and magnitude must all be the same size (either 16 or 32)
     *
     * @param[in]  gx        Source tensor - Gx component. Data type supported: S16/S32.
     * @param[in]  gy        Source tensor - Gy component. Data type supported: same as @p gx.
     * @param[out] magnitude Destination tensor - Magnitude. Data type supported: U16 (if the data type of @p gx is S16) / U32 (if the data type of @p gx is S32).
     * @param[out] phase     Destination tensor - Quantized phase. Data type supported: U8.
     * @param[in]  norm_type Normalization type. If 1, L1-Norm otherwise L2-Norm
     */
    virtual void configure(const ITensor *gx, const ITensor *gy, ITensor *magnitude, ITensor *phase, int32_t norm_type);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

protected:
    /** Common signature for all the specialised gradient functions
     *
     * @param[in]  gx_ptr        Pointer to the first input tensor.
     * @param[in]  gy_ptr        Pointer to the second input tensor.
     * @param[out] magnitude_ptr Pointer to the first output tensor
     * @param[out] phase_ptr     Pointer to the second output tensor
     */
    using GradientFunction = void(const void *__restrict gx_ptr, const void *__restrict gy_ptr, void *__restrict magnitude_ptr, void *__restrict phase_ptr);

    GradientFunction *_func;      /**< Gradient function to use for the particular tensor types passed to configure() */
    const ITensor    *_gx;        /**< Source tensor - Gx component */
    const ITensor    *_gy;        /**< Source tensor - Gy component */
    ITensor          *_magnitude; /**< Destination tensor - Magnitude */
    ITensor          *_phase;     /**< Destination tensor - Quantized phase */
};

/** Neon kernel to perform Non-Maxima suppression for Canny Edge.
 *
 * @note This kernel is meant to be used alongside CannyEdge and performs a non-maxima suppression using magnitude and phase of input
 *       to characterize points as possible edges. Thus, at the end, each point will be set to EDGE, NO_EDGE or MAYBE.
 *
 * @note Hysteresis is computed in @ref NEEdgeTraceKernel
 */
class NEEdgeNonMaxSuppressionKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEEdgeNonMaxSuppressionKernel";
    }
    /** Default constructor */
    NEEdgeNonMaxSuppressionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEEdgeNonMaxSuppressionKernel(const NEEdgeNonMaxSuppressionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEEdgeNonMaxSuppressionKernel &operator=(const NEEdgeNonMaxSuppressionKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEEdgeNonMaxSuppressionKernel(NEEdgeNonMaxSuppressionKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEEdgeNonMaxSuppressionKernel &operator=(NEEdgeNonMaxSuppressionKernel &&) = default;
    /** Default destructor */
    ~NEEdgeNonMaxSuppressionKernel();

    /** Initialise the kernel's sources, destination and border mode.
     *
     * @param[in]  magnitude        Source tensor - Magnitude. Data type supported: U16/U32.
     * @param[in]  phase            Source tensor - Quantized phase. Data type supported: U8.
     * @param[out] output           Output tensor. Data type supported: U8. It will be filled with 0 for "no edge", 127 for "maybe", 255 for "edge"
     * @param[in]  upper_thr        Upper threshold used for the hysteresis
     * @param[in]  lower_thr        Lower threshold used for the hysteresis
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *magnitude, const ITensor *phase, ITensor *output, int32_t upper_thr, int32_t lower_thr, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Common signature for all the specialised non-maxima suppression functions
     *
     * @param[in]  magnitude_ptr Pointer to the first input tensor.
     * @param[in]  phase_ptr     Pointer to the second input tensor.
     * @param[out] output_ptr    Pointer to the output tensor
     * @param[in]  stride_mag    Stride of the magnitude tensor
     * @param[in]  upper_thr     Upper threshold used for the hysteresis
     * @param[in]  lower_thr     Lower threshold used for the hysteresis
     */
    using EdgeNonMaxSupprFunction = void(const void *__restrict magnitude_ptr, const void *__restrict phase_ptr, void *__restrict output_ptr, const uint32_t stride_mag, const int32_t upper_thr,
                                         const int32_t lower_thr);

    EdgeNonMaxSupprFunction *_func;      /**< Non-Maxima suppression function to use for the particular tensor types passed to configure() */
    const ITensor           *_magnitude; /**< Source tensor - Magnitude */
    const ITensor           *_phase;     /**< Source tensor - Quantized phase */
    ITensor                 *_output;    /**< Destination tensor */
    int32_t                  _lower_thr; /**< Lower threshold used for the hysteresis */
    int32_t                  _upper_thr; /**< Upper threshold used for the hysteresis */
};

/** Neon kernel to perform Edge tracing */
class NEEdgeTraceKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEEdgeTraceKernel";
    }
    /** Default constructor */
    NEEdgeTraceKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEEdgeTraceKernel(const NEEdgeTraceKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEEdgeTraceKernel &operator=(const NEEdgeTraceKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEEdgeTraceKernel(NEEdgeTraceKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEEdgeTraceKernel &operator=(NEEdgeTraceKernel &&) = default;
    /** Default destructor */
    ~NEEdgeTraceKernel();

    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in,out] input  Source tensor. Data type supported: U8. Must contain 0 for "no edge", 127 for "maybe", 255 for "edge"
     * @param[in,out] output Destination tensor. Data type supported: U8. Must be initialized to 0 (No edge).
     */
    void configure(ITensor *input, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;
    bool       is_parallelisable() const override;

private:
    ITensor *_input;  /**< Source tensor */
    ITensor *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NECANNYEDGEKERNEL_H */
