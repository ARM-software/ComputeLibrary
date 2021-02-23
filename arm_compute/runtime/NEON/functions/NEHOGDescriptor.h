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
#ifndef ARM_COMPUTE_NEHOGDESCRIPTOR_H
#define ARM_COMPUTE_NEHOGDESCRIPTOR_H

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEHOGGradient.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class IHOG;
class NEHOGOrientationBinningKernel;
class NEHOGBlockNormalizationKernel;

/** Basic function to calculate HOG descriptor. This function calls the following Neon kernels:
 *
 * -# @ref NEHOGGradient
 * -# @ref NEHOGOrientationBinningKernel
 * -# @ref NEHOGBlockNormalizationKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEHOGDescriptor : public IFunction
{
public:
    /** Default constructor */
    NEHOGDescriptor(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGDescriptor(const NEHOGDescriptor &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGDescriptor &operator=(const NEHOGDescriptor &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEHOGDescriptor(NEHOGDescriptor &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEHOGDescriptor &operator=(NEHOGDescriptor &&) = delete;
    /** Default destructor */
    ~NEHOGDescriptor();
    /** Initialise the function's source, destination, HOG data-object and border mode
     *
     * @param[in, out] input                 Input tensor. Data type supported: U8
     *                                       (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output                Output tensor which stores the HOG descriptor. DataType supported: F32. The number of channels is equal to the number of histogram bins per block
     * @param[in]      hog                   HOG data object which describes the HOG descriptor
     * @param[in]      border_mode           Border mode to use.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ITensor *input, ITensor *output, const IHOG *hog, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited method overridden:
    void run() override;

private:
    MemoryGroup                                    _memory_group;
    NEHOGGradient                                  _gradient;
    std::unique_ptr<NEHOGOrientationBinningKernel> _orient_bin;
    std::unique_ptr<NEHOGBlockNormalizationKernel> _block_norm;
    Tensor                                         _mag;
    Tensor                                         _phase;
    Tensor                                         _hog_space;
};
}

#endif /* ARM_COMPUTE_NEHOGDESCRIPTOR_H */
