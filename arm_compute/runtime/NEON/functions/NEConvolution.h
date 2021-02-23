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
#ifndef ARM_COMPUTE_NECONVOLUTION_H
#define ARM_COMPUTE_NECONVOLUTION_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ITensor;
class NEFillBorderKernel;
template <unsigned int matrix_size>
class NEConvolutionKernel;
template <unsigned int matrix_size>
class NESeparableConvolutionHorKernel;
template <unsigned int matrix_size>
class NESeparableConvolutionVertKernel;

/** Basic function to execute convolution of size 3x3. This function calls the following Neon kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEConvolution3x3Kernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEConvolution3x3 : public INESimpleFunction
{
public:
    /** Constructor */
    NEConvolution3x3() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolution3x3(const NEConvolution3x3 &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolution3x3 &operator=(const NEConvolution3x3 &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConvolution3x3(NEConvolution3x3 &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConvolution3x3 &operator=(NEConvolution3x3 &&) = delete;
    /** Default destructor */
    ~NEConvolution3x3();
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8/S16.
     * @param[in]     conv                  Matrix_size x matrix_size S16 coefficients structured as a row-major 2D array in a linear buffer.
     * @param[in]     scale                 Scale of the convolution matrix. If 0 is passed, it will be set to the sum of the coefficients of the convolution or 1 if they add up to 0.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value = 0);
};

/** Basic function to execute convolution of size 5x5, 7x7, 9x9. This function calls the following Neon kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEConvolutionKernel or<br/>
 *    @ref NESeparableConvolutionHorKernel and @ref NESeparableConvolutionVertKernel (if convolution matrix is separable)
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
template <unsigned int matrix_size>
class NEConvolutionSquare : public IFunction
{
public:
    /** Default constructor */
    NEConvolutionSquare(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolutionSquare(const NEConvolutionSquare &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolutionSquare &operator=(const NEConvolutionSquare &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConvolutionSquare(NEConvolutionSquare &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConvolutionSquare &operator=(NEConvolutionSquare &&) = delete;
    /** Default destructor */
    ~NEConvolutionSquare();
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
    MemoryGroup                                                    _memory_group;   /**< Function memory group */
    Tensor                                                         _tmp;            /**< temporary buffer for output of horizontal pass */
    bool                                                           _is_separable;   /**< true if the convolution can be separated */
    std::unique_ptr<NESeparableConvolutionHorKernel<matrix_size>>  _kernel_hor;     /**< kernel for horizontal pass of separated convolution */
    std::unique_ptr<NESeparableConvolutionVertKernel<matrix_size>> _kernel_vert;    /**< kernel for vertical pass of separated convolution */
    std::unique_ptr<NEConvolutionKernel<matrix_size>>              _kernel;         /**< kernel for non-separated convolution **/
    std::unique_ptr<NEFillBorderKernel>                            _border_handler; /**< kernel for border handling */
};

/** Basic function to run 5x5 convolution. */
using NEConvolution5x5 = NEConvolutionSquare<5>;
/** Basic function to run 7x7 convolution. */
using NEConvolution7x7 = NEConvolutionSquare<7>;
/** Basic function to run 9x9 convolution. */
using NEConvolution9x9 = NEConvolutionSquare<9>;

/** Basic function to execute non-square convolution. This function calls the following Neon kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEConvolutionRectangleKernel or<br/>
 *
 * @note Convolution rectangle should have dimensions of 3, 5, 7, 9
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEConvolutionRectangle : public INESimpleFunction
{
public:
    /** Constructor */
    NEConvolutionRectangle() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolutionRectangle(const NEConvolutionRectangle &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolutionRectangle &operator=(const NEConvolutionRectangle &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConvolutionRectangle(NEConvolutionRectangle &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConvolutionRectangle &operator=(NEConvolutionRectangle &&) = delete;
    /** Default destructor */
    ~NEConvolutionRectangle();
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
#endif /*ARM_COMPUTE_NECONVOLUTION_H */
