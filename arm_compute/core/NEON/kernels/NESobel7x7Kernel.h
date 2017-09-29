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
#ifndef __ARM_COMPUTE_NESOBEL7x7KERNEL_H__
#define __ARM_COMPUTE_NESOBEL7x7KERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to run the horizontal pass of 7x7 Sobel filter on a tensor.
 *
 */
class NESobel7x7HorKernel : public INEKernel
{
public:
    /** Default constructor */
    NESobel7x7HorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel7x7HorKernel(const NESobel7x7HorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel7x7HorKernel &operator=(const NESobel7x7HorKernel &) = delete;
    /** Allow instances of this class to be moved */
    NESobel7x7HorKernel(NESobel7x7HorKernel &&) = default;
    /** Allow instances of this class to be moved */
    NESobel7x7HorKernel &operator=(NESobel7x7HorKernel &&) = default;
    /** Default destructor */
    ~NESobel7x7HorKernel() = default;

    /** Initialise the kernel's source, destination and border mode.
     *
     * @note At least one of output_x or output_y must be set.
     *
     * @param[in]  input            Source tensor. Data type supported: U8.
     * @param[out] output_x         (Optional) Destination tensor for the X gradient. Data type supported: S32.
     * @param[out] output_y         (Optional) Destination tensor for the Y gradient. Data type supported: S32.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output_x, ITensor *output_y, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    const ITensor *_input;       /**< Input tensor */
    ITensor       *_output_x;    /**< X output of horizontal pass */
    ITensor       *_output_y;    /**< Y output of horizontal pass */
    bool           _run_sobel_x; /**< Do we need to run Sobel X? */
    bool           _run_sobel_y; /**< Do we need to run Sobel Y? */
    BorderSize     _border_size; /**< Border size */
};

/** Interface for the kernel to run the vertical pass of 7x7 Sobel Y filter on a tensor.
 *
*/
class NESobel7x7VertKernel : public INEKernel
{
public:
    /** Default constructor */
    NESobel7x7VertKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel7x7VertKernel(const NESobel7x7VertKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel7x7VertKernel &operator=(const NESobel7x7VertKernel &) = delete;
    /** Allow instances of this class to be moved */
    NESobel7x7VertKernel(NESobel7x7VertKernel &&) = default;
    /** Allow instances of this class to be moved */
    NESobel7x7VertKernel &operator=(NESobel7x7VertKernel &&) = default;
    /** Default destructor */
    ~NESobel7x7VertKernel() = default;

    /** Initialise the kernel's source, destination and border mode.
     *
     * @note At least one of output_x or output_y must be set
     * @note If output_x is set then input_x must be set too
     * @note If output_y is set then input_y must be set too
     *
     * @param[in]  input_x          (Optional) Input for X (X output of hor pass). Data type supported: S32.
     * @param[in]  input_y          (Optional) Input for Y (Y output of hor pass). Data type supported: S32.
     * @param[out] output_x         (Optional) Destination tensor for the X gradient. Data type supported: S32.
     * @param[out] output_y         (Optional) Destination tensor for the Y gradient. Data type supported: S32.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input_x, const ITensor *input_y, ITensor *output_x, ITensor *output_y, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    const ITensor *_input_x;     /**< X input (X output of the hor pass) */
    const ITensor *_input_y;     /**< Y input (Y output of the hor pass) */
    ITensor       *_output_x;    /**< X output of sobel */
    ITensor       *_output_y;    /**< Y output of sobel */
    bool           _run_sobel_x; /**< Do we need to run sobel X? */
    bool           _run_sobel_y; /**< Do we need to run sobel Y? */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NESOBEL7x7KERNEL_H__ */
