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
#ifndef __ARM_COMPUTE_CLCONVOLUTIONKERNEL_H__
#define __ARM_COMPUTE_CLCONVOLUTIONKERNEL_H__

#include "arm_compute/core/CL/ICLSimple2DKernel.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/****************************************************************************************\
 *                                    Square Convolution                                *
\****************************************************************************************/

/** Interface for the kernel to run an arbitrary size convolution on a tensor. (Currently supports 3x3, 5x5, 7x7 and 9x9).
 * The client can supply a convolution matrix \f$ C_{m,n} \f$.
 * @f{eqnarray}{
 *  k_0 &=& \frac{m}{2}  \\
 *  l_0 &=& \frac{n}{2}  \\
 *  sum &=& \sum_{k=0,l=0}^{k=m-1,l=n-1} input(x+k-k_0, y+l-l_0) C_{k,l}
 *  @f}
 *
 * @note The above equation for this function is similar to the default OpenCV Filter2D function,
 *       which actually computes a correlation and not a convolution.
 *       In case of a real convolution the convolution matrix should be flipped both horizontally and vertically.
 */
template <unsigned int matrix_size>
class CLConvolutionKernel : public ICLSimple2DKernel
{
public:
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input            Source tensor. Data types supported: U8.
     * @param[out] output           Destination tensor, Data types supported: U8, S16.
     * @param[in]  conv             Convolution matrix to apply to the input tensor.
     * @param[in]  scale            Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t scale, bool border_undefined);

    // Inherited methods overridden:
    BorderSize border_size() const override;
};

/** Interface for the kernel which applies a 3x3 convolution to a tensor. */
using CLConvolution3x3Kernel = CLConvolutionKernel<3>;
/** Interface for the kernel which applies a 5x5 convolution to a tensor. */
using CLConvolution5x5Kernel = CLConvolutionKernel<5>;
/** Interface for the kernel which applies a 7x7 convolution to a tensor. */
using CLConvolution7x7Kernel = CLConvolutionKernel<7>;
/** Interface for the kernel which applies a 9x9 convolution to a tensor. */
using CLConvolution9x9Kernel = CLConvolutionKernel<9>;

/****************************************************************************************\
 *                              Separable Square Convolution                            *
\****************************************************************************************/

/** Kernel for the Horizontal pass of a Separable Convolution. Currently support 5x5, 7x7, 9x9 */
template <unsigned int matrix_size>
class CLSeparableConvolutionHorKernel : public ICLSimple2DKernel
{
public:
    /** Default Constructor */
    CLSeparableConvolutionHorKernel();
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input            Source tensor. Data types supported: U8.
     * @param[out] output           Destination tensor, Data types supported: S16.
     * @param[in]  conv             Convolution matrix to apply to the input tensor.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const int16_t *conv, bool border_undefined);

    // Inherited methods overridden:
    BorderSize border_size() const override;

private:
    BorderSize _border_size; /**< Border size */
};

/** Interface for the kernel which applies a horizontal pass of 5x5 convolution to a tensor. */
using CLSeparableConvolution5x5HorKernel = CLSeparableConvolutionHorKernel<5>;
/** Interface for the kernel which applies a horizontal pass of 7x7 convolution to a tensor. */
using CLSeparableConvolution7x7HorKernel = CLSeparableConvolutionHorKernel<7>;
/** Interface for the kernel which applies a horizontal pass of 9x9 convolution to a tensor. */
using CLSeparableConvolution9x9HorKernel = CLSeparableConvolutionHorKernel<9>;

/** Kernel for the Vertical pass of a Separable Convolution. Currently supports 5x5, 7x7, 9x9 */
template <unsigned int matrix_size>
class CLSeparableConvolutionVertKernel : public ICLSimple2DKernel
{
public:
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input            Source tensor. Data types supported: S16.
     * @param[out] output           Destination tensor, Data types supported: U8, S16.
     * @param[in]  conv             Convolution matrix to apply to the input tensor.
     * @param[in]  scale            Scale of the convolution matrix.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     * @param[in]  data_type        Data type to use for intermeidate result. @sa data_type_for_convolution
     */
    void configure(const ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t scale, bool border_undefined, DataType data_type = DataType::S32);

    // Inherited methods overridden:
    BorderSize border_size() const override;
};

/** Interface for the kernel which applies a vertical pass of 5x5 convolution to a tensor. */
using CLSeparableConvolution5x5VertKernel = CLSeparableConvolutionVertKernel<5>;
/** Interface for the kernel which applies a vertical pass of 7x7 convolution to a tensor. */
using CLSeparableConvolution7x7VertKernel = CLSeparableConvolutionVertKernel<7>;
/** Interface for the kernel which applies a vertical pass of 9x9 convolution to a tensor. */
using CLSeparableConvolution9x9VertKernel = CLSeparableConvolutionVertKernel<9>;

/****************************************************************************************\
 *                                 Rectangle Convolution                                *
\****************************************************************************************/

/** Kernel for the running convolution on a rectangle matrix.
 *
 * @note Supports combinations of 3,5,7 and 9.
 */
class CLConvolutionRectangleKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLConvolutionRectangleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConvolutionRectangleKernel(const CLConvolutionRectangleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConvolutionRectangleKernel &operator=(const CLConvolutionRectangleKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLConvolutionRectangleKernel(CLConvolutionRectangleKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLConvolutionRectangleKernel &operator=(CLConvolutionRectangleKernel &&) = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input            Source tensor. Data types supported: U8.
     * @param[out] output           Destination tensor, Data types supported: U8, S16.
     * @param[in]  conv             Convolution matrix to apply to the input tensor.
     * @param[in]  width            Width of convolution matrix (Number of columns)
     * @param[in]  height           Height of convolution matrix (Number of rows)
     * @param[in]  scale            Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t width, uint32_t height, uint32_t scale, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    BorderSize       _border_size;
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLCONVOLUTIONKERNEL_H__ */
