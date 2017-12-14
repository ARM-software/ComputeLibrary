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
#ifndef __ARM_COMPUTE_NEDECONVOLUTIONUPSAMPLE_H__
#define __ARM_COMPUTE_NEDECONVOLUTIONUPSAMPLE_H__

#include "arm_compute/core/NEON/kernels/NEDeconvolutionLayerUpsampleKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEDeconvolutionLayerUpsampleKernel */
class NEDeconvolutionLayerUpsample : public IFunction
{
public:
    /** Constructor
     *
     * Initialize NEDeconvolutionLayerUpsample
     */
    NEDeconvolutionLayerUpsample(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in, out] input  Source tensor. Data type supported: F32.
     * @param[out]     output Destination tensor. Data type supported: F32.
     * @param[in]      a      Top and right inner border sizes. These rows and columns will be filled with zero.
     * @param[in]      iz     The number of zeros to be inserted between each input sample
     * @param[in]      info   Contains padding and policies to be used in the deconvolution, this is decribed in @ref PadStrideInfo.
     */
    void configure(ITensor *input, ITensor *output, const std::pair<unsigned int, unsigned int> &a,
                   const std::pair<unsigned int, unsigned int> &iz, const PadStrideInfo &info);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                        _memory_group;
    Tensor                             _offsets;
    NEFillBorderKernel                 _border_handler;
    NEDeconvolutionLayerUpsampleKernel _upsample;
};
} // arm_compute
#endif /*__ARM_COMPUTE_NEDECONVOLUTIONUPSAMPLE_H__ */
