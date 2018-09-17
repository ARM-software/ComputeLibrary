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
#ifndef __ARM_COMPUTE_NELAPLACIANRECONSTRUCT_H__
#define __ARM_COMPUTE_NELAPLACIANRECONSTRUCT_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConvert.h"
#include "arm_compute/runtime/NEON/functions/NEScale.h"
#include "arm_compute/runtime/Pyramid.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ITensor;
using IImage = ITensor;

/** Basic function to execute laplacian reconstruction. This function calls the following NEON kernels and functions:
 *
 * -# @ref NEArithmeticAddition
 * -# @ref NEScale
 * -# @ref NEDepthConvert
 *
 * This function reconstructs the original image from a Laplacian Image Pyramid.
 *
 *  The input image is added to the last level of the Laplacian pyramid L(n-2), the resulting image is upsampled to the
 *  resolution of the next pyramid level.
 *
 *  I(n-2) = upsample( input + L(n-1)
 *
 *  For each pyramid level i, except i=0 and i=n-1:
 *  I(i-1) = upsample(I(i) + L(i))
 *
 *  output = I(0) + L(0)
*/
class NELaplacianReconstruct : public IFunction
{
public:
    /** Constructor */
    NELaplacianReconstruct();
    /** Initialise the function's source, destinations and border mode.
     *
     * The Output image must have the same size as the first level of the pyramid.
     * The Input image must have the same size as the last level of the pyramid.
     *
     * The idea is to reconstuct the original hi-res image from a low-res representation of it and the laplacian pyramid.
     *
     * @param[in]  pyramid               Laplacian pyramid tensors, Data type supported at each level: S16.
     * @param[in]  input                 Source tensor. Data type supported: S16.
     * @param[out] output                Output tensor. Data type supported: U8.
     * @param[in]  border_mode           Border mode to use for the convolution.
     * @param[in]  constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    void configure(const IPyramid *pyramid, const ITensor *input, ITensor *output, BorderMode border_mode, uint8_t constant_border_value);

    // Inherited methods overridden:
    void run() override;

private:
    Pyramid                                 _tmp_pyr;
    std::unique_ptr<NEArithmeticAddition[]> _addf;
    std::unique_ptr<NEScale[]>              _scalef;
    NEDepthConvert                          _depthf;
};
}
#endif /*__ARM_COMPUTE_NELAPLACIANRECONSTRUCT_H__ */
