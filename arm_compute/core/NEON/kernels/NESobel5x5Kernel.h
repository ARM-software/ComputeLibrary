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
#ifndef __ARM_COMPUTE_NESOBEL5x5KERNEL_H__
#define __ARM_COMPUTE_NESOBEL5x5KERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to run the horizontal pass of 5x5 Sobel filter on a tensor.
 *
 */
class NESobel5x5HorKernel : public INEKernel
{
public:
    /** Default constructor */
    NESobel5x5HorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel5x5HorKernel(const NESobel5x5HorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel5x5HorKernel &operator=(const NESobel5x5HorKernel &) = delete;
    /** Allow instances of this class to be moved */
    NESobel5x5HorKernel(NESobel5x5HorKernel &&) = default;
    /** Allow instances of this class to be moved */
    NESobel5x5HorKernel &operator=(NESobel5x5HorKernel &&) = default;
    /** Default destructor */
    ~NESobel5x5HorKernel() = default;

    /** Initialise the kernel's source, destination and border mode.
     *
     * @note At least one of output_x or output_y must be set
     *
     * @param[in]  input            Source tensor. Data type supported: U8.
     * @param[out] output_x         (Optional) Destination tensor for the X gradient. Data type supported: S16.
     * @param[out] output_y         (Optional) Destination tensor for the Y gradient. Data type supported: S16.
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

/** Interface for the kernel to run the vertical pass of 5x5 Sobel Y filter on a tensor.
 *
*/
class NESobel5x5VertKernel : public INEKernel
{
public:
    /** Default constructor */
    NESobel5x5VertKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel5x5VertKernel(const NESobel5x5VertKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESobel5x5VertKernel &operator=(const NESobel5x5VertKernel &) = delete;
    /** Allow instances of this class to be moved */
    NESobel5x5VertKernel(NESobel5x5VertKernel &&) = default;
    /** Allow instances of this class to be moved */
    NESobel5x5VertKernel &operator=(NESobel5x5VertKernel &&) = default;
    /** Default destructor */
    ~NESobel5x5VertKernel() = default;

    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in]  input_x          Input for X (X output of hor pass). Data type supported: S16.
     * @param[in]  input_y          Input for Y (Y output of hor pass). Data type supported: S16.
     * @param[out] output_x         Destination tensor for the X gradient. Data type supported: S16.
     * @param[out] output_y         Destination tensor for the Y gradient. Data type supported: S16.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(ITensor *input_x, ITensor *input_y, ITensor *output_x, ITensor *output_y, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    ITensor *_input_x;     /**< X input (X output of the hor pass) */
    ITensor *_input_y;     /**< Y input (Y output of the hor pass) */
    ITensor *_output_x;    /**< X output of sobel */
    ITensor *_output_y;    /**< Y output of sobel */
    bool     _run_sobel_x; /**< Do we need to run sobel X? */
    bool     _run_sobel_y; /**< Do we need to run sobel Y? */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NESOBEL5x5KERNEL_H__ */
