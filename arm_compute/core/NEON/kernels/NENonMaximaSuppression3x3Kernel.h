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
#ifndef __ARM_COMPUTE_NENONMAXIMASUPPRESSION3x3KERNEL_H__
#define __ARM_COMPUTE_NENONMAXIMASUPPRESSION3x3KERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Interface to perform Non-Maxima suppression over a 3x3 window using NEON
 *
 * @note Used by @ref NEFastCorners and @ref NEHarrisCorners
 */
class NENonMaximaSuppression3x3Kernel : public INEKernel
{
public:
    /** Default constructor */
    NENonMaximaSuppression3x3Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NENonMaximaSuppression3x3Kernel(const NENonMaximaSuppression3x3Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NENonMaximaSuppression3x3Kernel &operator=(const NENonMaximaSuppression3x3Kernel &) = delete;
    /** Allow instances of this class to be moved */
    NENonMaximaSuppression3x3Kernel(NENonMaximaSuppression3x3Kernel &&) = default;
    /** Allow instances of this class to be moved */
    NENonMaximaSuppression3x3Kernel &operator=(NENonMaximaSuppression3x3Kernel &&) = default;
    /** Default destructor */
    ~NENonMaximaSuppression3x3Kernel() = default;

    /** Initialise the kernel's sources, destinations and border mode.
     *
     * @param[in]  input            Source tensor. Data types supported: U8/F32
     * @param[out] output           Destination tensor. Data types supported: same as @p input
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

protected:
    /** Common signature for all the specialised non-maxima suppression 3x3 functions
     *
     * @param[in]  input_ptr    Pointer to the input tensor.
     * @param[out] output_ptr   Pointer to the output tensor
     * @param[in]  input_stride Stride of the input tensor
     */
    using NonMaxSuppr3x3Function = void(const void *__restrict input_ptr, void *__restrict output_ptr, const uint32_t input_stride);

    NonMaxSuppr3x3Function *_func;   /**< Non-Maxima suppression function to use for the particular tensor types passed to configure() */
    const ITensor          *_input;  /**< Source tensor */
    ITensor                *_output; /**< Destination tensor */
};

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/** NEON kernel to perform Non-Maxima suppression 3x3 with intermediate results in F16 if the input data type is F32
 */
class NENonMaximaSuppression3x3FP16Kernel : public NENonMaximaSuppression3x3Kernel
{
public:
    /** Initialise the kernel's sources, destinations and border mode.
     *
     * @param[in]  input            Source tensor. Data types supported: U8/F32.
     * @param[out] output           Destination tensor. Data types supported: same as @p input
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output, bool border_undefined);
};
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
using NENonMaximaSuppression3x3FP16Kernel = NENonMaximaSuppression3x3Kernel;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
} // namespace arm_compute
#endif /* _ARM_COMPUTE_NENONMAXIMASUPPRESSION3x3KERNEL_H__ */
