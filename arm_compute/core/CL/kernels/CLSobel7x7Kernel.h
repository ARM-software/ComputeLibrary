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
#ifndef __ARM_COMPUTE_CLSOBEL7X7KERNEL_H__
#define __ARM_COMPUTE_CLSOBEL7X7KERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to run the horizontal pass of 7x7 Sobel filter on a tensor. */
class CLSobel7x7HorKernel : public ICLKernel
{
public:
    /** Default constructor: initialize all the pointers to nullptr and parameters to zero. */
    CLSobel7x7HorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLSobel7x7HorKernel(const CLSobel7x7HorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLSobel7x7HorKernel &operator=(const CLSobel7x7HorKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLSobel7x7HorKernel(CLSobel7x7HorKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLSobel7x7HorKernel &operator=(CLSobel7x7HorKernel &&) = default;
    /** Default destructor */
    ~CLSobel7x7HorKernel() = default;

    /** Initialise the kernel's source, destination and border.
     *
     * @note At least one of output_x or output_y must be set.
     *
     * @param[in]  input            Source tensor. Data types supported: U8.
     * @param[out] output_x         (Optional) Destination tensor for the X gradient, Data types supported: S32.
     * @param[out] output_y         (Optional) Destination tensor for the Y gradient, Data types supported: S32.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output_x, ICLTensor *output_y, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_input;       /**< Input tensor */
    ICLTensor       *_output_x;    /**< X output of horizontal pass */
    ICLTensor       *_output_y;    /**< Y output of horizontal pass */
    bool             _run_sobel_x; /**< Do we need to run Sobel X ? */
    bool             _run_sobel_y; /**< Do we need to run Sobel Y ? */
    BorderSize       _border_size; /**< Border size */
};

/** Interface for the kernel to run the vertical pass of 7x7 Sobel filter on a tensor. */
class CLSobel7x7VertKernel : public ICLKernel
{
public:
    /** Default constructor: initialize all the pointers to nullptr and parameters to zero. */
    CLSobel7x7VertKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLSobel7x7VertKernel(const CLSobel7x7VertKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLSobel7x7VertKernel &operator=(const CLSobel7x7VertKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLSobel7x7VertKernel(CLSobel7x7VertKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLSobel7x7VertKernel &operator=(CLSobel7x7VertKernel &&) = default;
    /** Default destructor */
    ~CLSobel7x7VertKernel() = default;

    /** Initialise the kernel's source, destination and border.
     *
     * @note At least one of output_x or output_y must be set and the corresponding input.
     *
     * @param[in]  input_x          (Optional) Input for X (X output of horizontal pass). Data types supported: S32.
     * @param[in]  input_y          (Optional) Input for Y (Y output of horizontal pass). Data types supported: S32.
     * @param[out] output_x         (Optional) Destination tensor for the X gradient, Data types supported: S32.
     * @param[out] output_y         (Optional) Destination tensor for the Y gradient, Data types supported: S32.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input_x, const ICLTensor *input_y, ICLTensor *output_x, ICLTensor *output_y, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_input_x;     /**< X input (X output of the horizontal pass) */
    const ICLTensor *_input_y;     /**< Y input (Y output of the horizontal pass) */
    ICLTensor       *_output_x;    /**< X output of sobel */
    ICLTensor       *_output_y;    /**< Y output of sobel */
    bool             _run_sobel_x; /**< Do we need to run sobel X? */
    bool             _run_sobel_y; /**< Do we need to run sobel Y? */
};
}
#endif /*__ARM_COMPUTE_CLSOBEL7X7KERNEL_H__ */
