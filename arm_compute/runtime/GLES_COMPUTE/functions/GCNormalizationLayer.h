/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GCNORMALIZATIONLAYER_H__
#define __ARM_COMPUTE_GCNORMALIZATIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/GLES_COMPUTE/kernels/GCFillBorderKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCNormalizationLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCPixelWiseMultiplicationKernel.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to compute a normalization layer. This function calls the following OpenGL ES kernels:
 *
 * -# @ref GCPixelWiseMultiplicationKernel
 * -# @ref GCFillBorderKernel
 * -# @ref GCNormalizationLayerKernel
 *
 */
class GCNormalizationLayer : public IFunction
{
public:
    /** Default constructor */
    GCNormalizationLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                       and an optional 4th dimension for batch of inputs. Data types supported: F32. Number of channels must be 1.
     * @param[out] output    Destination tensor. Dimensions, data type and number of channels must match the input ones.
     * @param[in]  norm_info Normalization layer information like the normalization type, normalization size and other parameters.
     */
    void configure(const IGCTensor *input, IGCTensor *output, const NormalizationLayerInfo &norm_info);

    // Inherited methods overridden:
    void run() override;

private:
    GCTensor                        _squared_input;   /**< The intermediate buffer which stores results of squaring input*/
    GCNormalizationLayerKernel      _norm_kernel;     /**< Normalization layer kernel to run */
    GCPixelWiseMultiplicationKernel _multiply_kernel; /**< Pixel multiplication kernel to run */
    GCFillBorderKernel              _border_handler;  /**< Kernel to handle  borders */
};
}
#endif /* __ARM_COMPUTE_GCNORMALIZATIONLAYER_H__ */
