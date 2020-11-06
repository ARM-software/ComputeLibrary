/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLHOGDESCRIPTOR_H
#define ARM_COMPUTE_CLHOGDESCRIPTOR_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLHOGGradient.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
class IHOG;
class CLHOGOrientationBinningKernel;
class CLHOGBlockNormalizationKernel;
/** Basic function to calculate HOG descriptor. This function calls the following OpenCL kernels:
 *
 * -# @ref CLHOGGradient
 * -# @ref CLHOGOrientationBinningKernel
 * -# @ref CLHOGBlockNormalizationKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class CLHOGDescriptor : public IFunction
{
public:
    /** Default constructor */
    CLHOGDescriptor(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied */
    CLHOGDescriptor(const CLHOGDescriptor &) = delete;
    /** Prevent instances of this class from being copied */
    CLHOGDescriptor &operator=(const CLHOGDescriptor &) = delete;
    /** Default destructor */
    ~CLHOGDescriptor();
    /** Initialise the function's source, destination, HOG data-object and border mode
     *
     * @param[in, out] input                 Input tensor. Data type supported: U8
     *                                       (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output                Output tensor which stores the HOG descriptor. DataType supported: F32. The number of channels is equal to the number of histogram bins per block
     * @param[in]      hog                   HOG data object which describes the HOG descriptor
     * @param[in]      border_mode           Border mode to use.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *input, ICLTensor *output, const IHOG *hog, BorderMode border_mode, uint8_t constant_border_value = 0);
    /** Initialise the function's source, destination, HOG data-object and border mode
     *
     * @param[in]      compile_context       The compile context to be used.
     * @param[in, out] input                 Input tensor. Data type supported: U8
     *                                       (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output                Output tensor which stores the HOG descriptor. DataType supported: F32. The number of channels is equal to the number of histogram bins per block
     * @param[in]      hog                   HOG data object which describes the HOG descriptor
     * @param[in]      border_mode           Border mode to use.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const IHOG *hog, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited method overridden:
    void run() override;

private:
    MemoryGroup                                    _memory_group;
    CLHOGGradient                                  _gradient;
    std::unique_ptr<CLHOGOrientationBinningKernel> _orient_bin;
    std::unique_ptr<CLHOGBlockNormalizationKernel> _block_norm;
    CLTensor                                       _mag;
    CLTensor                                       _phase;
    CLTensor                                       _hog_space;
};
}

#endif /* ARM_COMPUTE_CLHOGDESCRIPTOR_H */
