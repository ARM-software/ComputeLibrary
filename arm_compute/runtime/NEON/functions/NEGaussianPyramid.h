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
#ifndef ARM_COMPUTE_NEGAUSSIANPYRAMID_H
#define ARM_COMPUTE_NEGAUSSIANPYRAMID_H

#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/functions/NEGaussian5x5.h"
#include "arm_compute/runtime/NEON/functions/NEScale.h"
#include "arm_compute/runtime/Pyramid.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ITensor;
class NEGaussianPyramidHorKernel;
class NEGaussianPyramidVertKernel;
class NEFillBorderKernel;

/** Common interface for all Gaussian pyramid functions
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEGaussianPyramid : public IFunction
{
public:
    /** Default constructor */
    NEGaussianPyramid();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramid(const NEGaussianPyramid &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramid &operator=(const NEGaussianPyramid &) = delete;
    /** Allow instances of this class to be moved */
    NEGaussianPyramid(NEGaussianPyramid &&) = default;
    /** Allow instances of this class to be moved */
    NEGaussianPyramid &operator=(NEGaussianPyramid &&) = default;
    /** Default destructor */
    virtual ~NEGaussianPyramid() = default;

    /** Initialise the function's source, destinations and border mode.
     *
     * @param[in]  input                 Source tensor. Data type supported: U8.
     * @param[out] pyramid               Destination pyramid tensors, Data type supported at each level: U8.
     * @param[in]  border_mode           Border mode to use.
     * @param[in]  constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    virtual void configure(const ITensor *input, IPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value) = 0;

protected:
    const ITensor *_input;
    IPyramid      *_pyramid;
    Pyramid        _tmp;
};

/** Basic function to execute gaussian pyramid with HALF scale factor. This function calls the following Neon kernels:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEGaussianPyramidHorKernel
 * -# @ref NEGaussianPyramidVertKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 *
 */
class NEGaussianPyramidHalf : public NEGaussianPyramid
{
public:
    /** Constructor */
    NEGaussianPyramidHalf();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramidHalf(const NEGaussianPyramidHalf &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramidHalf &operator=(const NEGaussianPyramidHalf &) = delete;
    /** Allow instances of this class to be moved */
    NEGaussianPyramidHalf(NEGaussianPyramidHalf &&) = default;
    /** Allow instances of this class to be moved */
    NEGaussianPyramidHalf &operator=(NEGaussianPyramidHalf &&) = default;
    /** Default destructor */
    ~NEGaussianPyramidHalf();

    // Inherited methods overridden:
    void configure(const ITensor *input, IPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value) override;
    void run() override;

private:
    std::vector<std::unique_ptr<NEFillBorderKernel>>          _horizontal_border_handler;
    std::vector<std::unique_ptr<NEFillBorderKernel>>          _vertical_border_handler;
    std::vector<std::unique_ptr<NEGaussianPyramidHorKernel>>  _horizontal_reduction;
    std::vector<std::unique_ptr<NEGaussianPyramidVertKernel>> _vertical_reduction;
};

/** Basic function to execute gaussian pyramid with ORB scale factor. This function calls the following Neon kernels and functions:
 *
 * -# @ref NEFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref NEGaussian5x5
 * -# @ref NEScaleKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 *
 */
class NEGaussianPyramidOrb : public NEGaussianPyramid
{
public:
    /** Constructor */
    NEGaussianPyramidOrb();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramidOrb(const NEGaussianPyramidOrb &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramidOrb &operator=(const NEGaussianPyramidOrb &) = delete;
    /** Allow instances of this class to be moved */
    NEGaussianPyramidOrb(NEGaussianPyramidOrb &&) = default;
    /** Allow instances of this class to be moved */
    NEGaussianPyramidOrb &operator=(NEGaussianPyramidOrb &&) = default;
    /** Default destructor */
    ~NEGaussianPyramidOrb();

    // Inherited methods overridden:
    void configure(const ITensor *input, IPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value) override;
    void run() override;

private:
    std::vector<NEGaussian5x5> _gaus5x5;
    std::vector<NEScale>       _scale_nearest;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGAUSSIANPYRAMID_H */
