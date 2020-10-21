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
#ifndef ARM_COMPUTE_CLINTEGRALIMAGE_H
#define ARM_COMPUTE_CLINTEGRALIMAGE_H

#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class CLIntegralImageHorKernel;
class CLIntegralImageVertKernel;
class ICLTensor;

/** Basic function to execute integral image. This function calls the following OpenCL kernels:
 *
 * -# @ref CLIntegralImageHorKernel
 * -# @ref CLIntegralImageVertKernel
 *
 */
class CLIntegralImage : public IFunction
{
public:
    /** Default Constructor. */
    CLIntegralImage();
    /** Prevent instances of this class from being copied */
    CLIntegralImage(const CLIntegralImage &) = delete;
    /** Prevent instances of this class from being copied */
    CLIntegralImage &operator=(const CLIntegralImage &) = delete;
    /** Default destructor */
    ~CLIntegralImage();
    /** Initialise the function's source, destinations and border mode.
     *
     * @param[in]  input  Source tensor. Data types supported: U8.
     * @param[out] output Destination tensor, Data types supported: U32.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialise the function's source, destinations and border mode.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: U8.
     * @param[out] output          Destination tensor, Data types supported: U32.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);

    // Inherited methods overridden:
    void run() override;

protected:
    std::unique_ptr<CLIntegralImageHorKernel>  _integral_hor;  /**< Integral Image Horizontal kernel */
    std::unique_ptr<CLIntegralImageVertKernel> _integral_vert; /**< Integral Image Vertical kernel */
};
}
#endif /*ARM_COMPUTE_CLINTEGRALIMAGE_H */
