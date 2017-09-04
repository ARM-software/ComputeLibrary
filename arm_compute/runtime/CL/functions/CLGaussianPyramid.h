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
#ifndef __ARM_COMPUTE_CLGAUSSIANPYRAMID_H__
#define __ARM_COMPUTE_CLGAUSSIANPYRAMID_H__

#include "arm_compute/core/CL/kernels/CLGaussianPyramidKernel.h"

#include "arm_compute/core/CL/kernels/CLScaleKernel.h"
#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLPyramid.h"
#include "arm_compute/runtime/CL/functions/CLGaussian5x5.h"
#include "arm_compute/runtime/IFunction.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Common interface for all Gaussian pyramid functions
 */
class CLGaussianPyramid : public IFunction
{
public:
    /** Constructor */
    CLGaussianPyramid();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGaussianPyramid(const CLGaussianPyramid &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGaussianPyramid &operator=(const CLGaussianPyramid &) = delete;
    /** Allow instances of this class to be moved */
    CLGaussianPyramid(CLGaussianPyramid &&) = default;
    /** Allow instances of this class to be moved */
    CLGaussianPyramid &operator=(CLGaussianPyramid &&) = default;
    /** Default destructor */
    virtual ~CLGaussianPyramid() = default;
    /** Initialise the function's source, destinations and border mode.
     *
     * @param[in, out] input                 Source tensor. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     pyramid               Destination pyramid tensors, Data types supported at each level: U8.
     * @param[in]      border_mode           Border mode to use.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    virtual void configure(ICLTensor *input, CLPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value = 0) = 0;

protected:
    ICLTensor *_input;
    CLPyramid *_pyramid;
    CLPyramid  _tmp;
};

/** Basic function to execute gaussian pyramid with HALF scale factor. This function calls the following OpenCL kernels:
 *
 * -# @ref CLFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref CLGaussianPyramidHorKernel
 * -# @ref CLGaussianPyramidVertKernel
 */
class CLGaussianPyramidHalf : public CLGaussianPyramid
{
public:
    /** Constructor */
    CLGaussianPyramidHalf();

    // Inherited methods overridden:
    void configure(ICLTensor *input, CLPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value) override;
    void run() override;

private:
    std::unique_ptr<CLFillBorderKernel[]>          _border_handler;
    std::unique_ptr<CLGaussianPyramidHorKernel[]>  _horizontal_reduction;
    std::unique_ptr<CLGaussianPyramidVertKernel[]> _vertical_reduction;
};

/** Basic function to execute gaussian pyramid with ORB scale factor. This function calls the following OpenCL kernels and functions:
 *
 * -# @ref CLFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref CLGaussian5x5
 * -# @ref CLScaleKernel
 */
class CLGaussianPyramidOrb : public CLGaussianPyramid
{
public:
    /** Constructor */
    CLGaussianPyramidOrb();

    // Inherited methods overridden:
    void configure(ICLTensor *input, CLPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value) override;
    void run() override;

private:
    std::unique_ptr<CLGaussian5x5[]> _gauss5x5;
    std::unique_ptr<CLScaleKernel[]> _scale_nearest;
};
}
#endif /*__ARM_COMPUTE_CLGAUSSIANPYRAMID_H__ */
