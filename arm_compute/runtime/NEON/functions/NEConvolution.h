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
#ifndef __ARM_COMPUTE_NECONVOLUTION_H__
#define __ARM_COMPUTE_NECONVOLUTION_H__

#include "arm_compute/core/NEON/kernels/NEConvolutionKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Basic function to execute convolution of size 3x3. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEConvolution3x3Kernel
 *
 */
class NEConvolution3x3 : public INESimpleFunction
{
public:
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8 or S16.
     * @param[in]     conv                  Matrix_size x matrix_size S16 coefficients structured as a row-major 2D array in a linear buffer.
     * @param[in]     scale                 Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value = 0);
};

/** Basic function to execute convolution of size 5x5. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEConvolution5x5Kernel or<br/>
 *    @ref NESeparableConvolution5x5HorKernel and @ref NESeparableConvolution5x5VertKernel (if convolution matrix is separable)
 *
 */
class NEConvolution5x5 : public IFunction
{
public:
    /** Default constructor */
    NEConvolution5x5();
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8 or S16.
     * @param[in]     conv                  matrix_size x matrix_size S16 coefficients structured as a row-major 2D array in a linear buffer.
     * @param[in]     scale                 Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

private:
    Tensor                              _tmp;            /**< temporary buffer for output of horizontal pass */
    bool                                _is_separable;   /**< true if the convolution can be separated */
    NESeparableConvolution5x5HorKernel  _kernel_hor;     /**< kernel for horizontal pass of separated convolution */
    NESeparableConvolution5x5VertKernel _kernel_vert;    /**< kernel for vertical pass of separated convolution */
    NEConvolution5x5Kernel              _kernel;         /**< kernel for non-separated convolution **/
    NEFillBorderKernel                  _border_handler; /**< kernel for border handling */
};

/** Basic function to execute convolution of size 7x7. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEConvolution7x7Kernel or<br/>
 *    @ref NESeparableConvolution7x7HorKernel and @ref NESeparableConvolution7x7VertKernel (if convolution matrix is separable)
 *
 */
class NEConvolution7x7 : public IFunction
{
public:
    /** Default constructor */
    NEConvolution7x7();
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8 or S16.
     * @param[in]     conv                  matrix_size x matrix_size S16 coefficients structured as a row-major 2D array in a linear buffer.
     * @param[in]     scale                 Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

private:
    Tensor                              _tmp;            /**< temporary buffer for output of horizontal pass */
    bool                                _is_separable;   /**< true if the convolution can be separated */
    NESeparableConvolution7x7HorKernel  _kernel_hor;     /**< kernel for horizontal pass of separated convolution */
    NESeparableConvolution7x7VertKernel _kernel_vert;    /**< kernel for vertical pass of separated convolution */
    NEConvolution7x7Kernel              _kernel;         /**< kernel for non-separated convolution **/
    NEFillBorderKernel                  _border_handler; /**< kernel for border handling */
};

/** Basic function to execute convolution of size 9x9. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEConvolution9x9Kernel or<br/>
 *    @ref NESeparableConvolution9x9HorKernel and @ref NESeparableConvolution9x9VertKernel (if convolution matrix is separable)
 *
 */
class NEConvolution9x9 : public IFunction
{
public:
    /** Default constructor */
    NEConvolution9x9();
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8 or S16.
     * @param[in]     conv                  matrix_size x matrix_size S16 coefficients structured as a row-major 2D array in a linear buffer.
     * @param[in]     scale                 Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

private:
    Tensor                              _tmp;            /**< temporary buffer for output of horizontal pass */
    bool                                _is_separable;   /**< true if the convolution can be separated */
    NESeparableConvolution9x9HorKernel  _kernel_hor;     /**< kernel for horizontal pass of separated convolution */
    NESeparableConvolution9x9VertKernel _kernel_vert;    /**< kernel for vertical pass of separated convolution */
    NEConvolution9x9Kernel              _kernel;         /**< kernel for non-separated convolution **/
    NEFillBorderKernel                  _border_handler; /**< kernel for border handling */
};

/** Basic function to execute non-square convolution. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEConvolutionRectangleKernel or<br/>
 *
 * @note Convolution rectangle should have dimensions of 3, 5, 7, 9
 */
class NEConvolutionRectangle : public INESimpleFunction
{
public:
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8 or S16.
     * @param[in]     conv                  Matrix_size x matrix_size S16 coefficients structured as a row-major 2D array in a linear buffer.
     * @param[in]     rows                  Rows of convolution kernel.
     * @param[in]     cols                  Columns of convolution kernel.
     * @param[in]     scale                 Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, const int16_t *conv, uint32_t rows, uint32_t cols, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value = 0);
};
}
#endif /*__ARM_COMPUTE_NECONVOLUTION_H__ */
