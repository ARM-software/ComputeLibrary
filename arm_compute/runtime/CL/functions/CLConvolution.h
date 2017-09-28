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
#ifndef __ARM_COMPUTE_CLCONVOLUTION_H__
#define __ARM_COMPUTE_CLCONVOLUTION_H__

#include "arm_compute/core/CL/kernels/CLConvolutionKernel.h"
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute convolution of size 3x3. This function calls the following OpenCL kernels:
 *
 * -# @ref CLFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref CLConvolution3x3Kernel
 *
 */
class CLConvolution3x3 : public ICLSimpleFunction
{
public:
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8 or S16.
     * @param[in]     conv                  matrix_size x matrix_size S16 coefficients structured as a row-major 2D array in a linear buffer.
     * @param[in]     scale                 Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value = 0);
};

/** Basic function to execute square convolution.Currently it supports 5x5, 7x7, 9x9. This function calls the following OpenCL kernels:
 *
 * -# @ref CLFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref CLConvolutionKernel or<br/>
 *    @ref CLSeparableConvolutionHorKernel and @ref CLSeparableConvolutionVertKernel (if convolution matrix is separable)
 *
 */
template <unsigned int matrix_size>
class CLConvolutionSquare : public IFunction
{
public:
    /** Default constructor */
    CLConvolutionSquare(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8 or S16.
     * @param[in]     conv                  matrix_size x matrix_size S16 coefficients structured as a row-major 2D array in a linear buffer.
     * @param[in]     scale                 Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overriden:
    void run() override;

private:
    CLMemoryGroup                                 _memory_group;   /**< Function's memory group */
    CLTensor                                      _tmp;            /**< temporary buffer for output of horizontal pass */
    bool                                          _is_separable;   /**< true if the convolution can be separated */
    CLSeparableConvolutionHorKernel<matrix_size>  _kernel_hor;     /**< kernel for horizontal pass of separated convolution */
    CLSeparableConvolutionVertKernel<matrix_size> _kernel_vert;    /**< kernel for vertical pass of separated convolution */
    CLConvolutionKernel<matrix_size>              _kernel;         /**< kernel for non-separated convolution **/
    CLFillBorderKernel                            _border_handler; /**< kernel for border handling */
};

/** Basic function to run 5x5 convolution. */
using CLConvolution5x5 = CLConvolutionSquare<5>;
/** Basic function to run 7x7 convolution. */
using CLConvolution7x7 = CLConvolutionSquare<7>;
/** Basic function to run 9x9 convolution. */
using CLConvolution9x9 = CLConvolutionSquare<9>;

/** Basic function to execute non-square convolution. This function calls the following CL kernels:
 *
 * -# @ref CLFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref CLConvolutionRectangleKernel or<br/>
 *
 * @note Convolution rectangle should have dimensions of 3, 5, 7, 9
 */
class CLConvolutionRectangle : public ICLSimpleFunction
{
public:
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8 or S16.
     * @param[in]     conv                  Matrix_size x matrix_size S16 coefficients structured as a row-major 2D array in a linear buffer.
     * @param[in]     rows                  Rows of convolution kernel.
     * @param[in]     cols                  Columns of convolution kernel.
     * @param[in]     scale                 Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t rows, uint32_t cols, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value = 0);
};
}
#endif /*__ARM_COMPUTE_CLCONVOLUTION_H__ */
