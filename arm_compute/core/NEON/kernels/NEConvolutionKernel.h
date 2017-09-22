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
#ifndef __ARM_COMPUTE_NECONVOLUTIONKERNEL_H__
#define __ARM_COMPUTE_NECONVOLUTIONKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/INESimpleKernel.h"

#include <array>
#include <cstdint>
#include <vector>

namespace arm_compute
{
class ITensor;

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
class NEConvolutionKernel : public INESimpleKernel
{
public:
    /** Default constructor */
    NEConvolutionKernel();
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input            Source tensor. Data type supported: U8.
     * @param[out] output           Destination tensor. Data types supported: U8, S16.
     * @param[in]  conv             Convolution matrix to apply to the input tensor.
     * @param[in]  scale            Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output, const int16_t *conv, uint32_t scale, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    template <typename OutputType>
    void convolution(const Window &win);

protected:
    uint32_t _scale;                                             /**< scale of the convolution */
    std::array<int16_t, matrix_size *matrix_size> _convolution;  /**< convolution matrix */
};

/** Interface for the kernel which applied a 3x3 convolution to a tensor.*/
using NEConvolution3x3Kernel = NEConvolutionKernel<3>;
/** Interface for the kernel which applied a 5x5 convolution to a tensor.*/
using NEConvolution5x5Kernel = NEConvolutionKernel<5>;
/** Interface for the kernel which applied a 7x7 convolution to a tensor.*/
using NEConvolution7x7Kernel = NEConvolutionKernel<7>;
///** Interface for the kernel which applied a 9x9 convolution to a tensor.*/
using NEConvolution9x9Kernel = NEConvolutionKernel<9>;

/****************************************************************************************\
 *                              Separable Square Convolution                            *
\****************************************************************************************/

/** Kernel for the Horizontal pass of a Separable Convolution */
template <unsigned int matrix_size>
class NESeparableConvolutionHorKernel : public INESimpleKernel
{
public:
    /** Default constructor */
    NESeparableConvolutionHorKernel();

    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input            Source tensor. Data type supported: U8.
     * @param[out] output           Destination tensor. Data types supported: U16, S16, S32.
     * @param[in]  conv_row         Convolution matrix to apply to the input tensor.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output, const int16_t *conv_row, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Apply the object's convolution to the given window of the input tensor..
     *
     * @param[in] window Window to apply the convolution on.
     */
    template <typename OutputType>
    void convolve(const Window &window);

    std::array<int16_t, matrix_size> _conv_row; /**< Convolution coefficients */
    BorderSize _border_size;                    /**< Border size */
};

/** Interface for the kernel which applied a 5x1 horizontal convolution to a tensor.*/
using NESeparableConvolution5x5HorKernel = NESeparableConvolutionHorKernel<5>;
/** Interface for the kernel which applied a 7x1 horizontal convolution to a tensor.*/
using NESeparableConvolution7x7HorKernel = NESeparableConvolutionHorKernel<7>;
/** Interface for the kernel which applied a 9x1 horizontal convolution to a tensor.*/
using NESeparableConvolution9x9HorKernel = NESeparableConvolutionHorKernel<9>;

/** Kernel for the Vertical pass of a Separable Convolution */
template <unsigned int matrix_size>
class NESeparableConvolutionVertKernel : public INESimpleKernel
{
public:
    /** Default constructor */
    NESeparableConvolutionVertKernel();

    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input            Source tensor. Data type supported: U16, S16, S32.
     * @param[out] output           Destination tensor, Data types supported: U8, S16.
     * @param[in]  conv_col         Convolution matrix to apply to the input tensor.
     * @param[in]  scale            Scale of the convolution matrix
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output, const int16_t *conv_col, uint32_t scale, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Apply the object's convolution to the given window of the input tensor.
     *  This function is used if the intermediate values have been stored as U16.
     *
     * @param[in] win Window to apply the convolution on.
     */
    template <typename OutputType>
    void convolution_u16(const Window &win);
    /** Apply the object's convolution to the given window of the input tensor.
     *  This function is used if the intermediate values have been stored as S16.
     *
     * @param[in] win Window to apply the convolution on.
     */
    template <typename OutputType>
    void convolution_s16(const Window &win);
    /** Apply the object's convolution to the given window of the input tensor.
     *  This function is used if the intermediate values have been stored as S32.
     *
     * @param[in] win Window to apply the convolution on.
     */
    template <typename OutputType>
    void convolution_s32(const Window &win);

    std::array<int16_t, matrix_size> _conv_col; /**< Convolution coefficients */
    uint32_t _scale;                            /**< Convolution's scale */
};

/** Interface for the kernel which applied a 1x5 vertical convolution to a tensor.*/
using NESeparableConvolution5x5VertKernel = NESeparableConvolutionVertKernel<5>;
/** Interface for the kernel which applied a 1x7 vertical convolution to a tensor.*/
using NESeparableConvolution7x7VertKernel = NESeparableConvolutionVertKernel<7>;
/** Interface for the kernel which applied a 1x9 vertical convolution to a tensor.*/
using NESeparableConvolution9x9VertKernel = NESeparableConvolutionVertKernel<9>;

/****************************************************************************************\
 *                                 Rectangle Convolution                                *
\****************************************************************************************/

/** Kernel for the running convolution on a rectangle matrix.
 *
 * @note Supports combinations of 3,5,7 and 9.
 */
class NEConvolutionRectangleKernel : public INEKernel
{
public:
    /** Default constructor */
    NEConvolutionRectangleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolutionRectangleKernel(NEConvolutionRectangleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolutionRectangleKernel &operator=(NEConvolutionRectangleKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEConvolutionRectangleKernel(NEConvolutionRectangleKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEConvolutionRectangleKernel &operator=(NEConvolutionRectangleKernel &&) = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input            Source tensor. Data type supported: U8.
     * @param[out] output           Destination tensor, Data types supported: U8, S16.
     * @param[in]  conv             Convolution matrix to apply to the input tensor.
     * @param[in]  width            Width of convolution matrix (Number of columns)
     * @param[in]  height           Height of convolution matrix (Number of rows)
     * @param[in]  scale            Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output, const int16_t *conv, uint32_t width, uint32_t height, uint32_t scale, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    unsigned int get_index(uint32_t val);
    /** Apply the object's convolution to the given window of the input tensor.
     *
     * @param[in] win Window to apply the convolution on.
     */
    template <typename OutputType, unsigned int rows, unsigned int cols>
    void convolution(const Window &win);

protected:
    const ITensor            *_input;       /**< Input tensor */
    ITensor                  *_output;      /**< Output tensor */
    uint32_t                  _scale;       /**< Scale of the convolution */
    std::vector<int16_t>      _convolution; /**< Convolution matrix */
    BorderSize                _border_size; /**< Calculated border width */
    uint32_t                  _func_idx;    /**< Index used to specify convolution function to be used */
    const static unsigned int _nr_supported_sizes
    {
        4
    }; /**< Number of supported permutations */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NECONVOLUTIONKERNEL_H__ */
