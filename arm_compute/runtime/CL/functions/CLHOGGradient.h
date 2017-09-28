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
#ifndef __ARM_COMPUTE_CLHOGGRADIENT_H__
#define __ARM_COMPUTE_CLHOGGRADIENT_H__

#include "arm_compute/core/CL/ICLKernel.h"

#include "arm_compute/core/CL/kernels/CLMagnitudePhaseKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLDerivative.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
/** Basic function to calculate the gradient for HOG. This function calls the following OpenCL kernels:
 *
 * -# @ref CLDerivative
 * -# @ref CLMagnitudePhaseKernel
 *
 */
class CLHOGGradient : public IFunction
{
public:
    /** Default constructor */
    CLHOGGradient(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialise the function's source, destinations, phase type and border mode
     *
     * @param[in, out] input                 Input tensor. Data type supported: U8.
     *                                       (Written to only for @p border_mode != UNDEFINED)
     * @param[out]     output_magnitude      Output tensor (magnitude). Data type supported: U16.
     * @param[out]     output_phase          Output tensor.(phase). Format supported: U8
     * @param[in]      phase_type            Type of @ref PhaseType
     * @param[in]      border_mode           Border mode to use
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *input, ICLTensor *output_magnitude, ICLTensor *output_phase, PhaseType phase_type, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited method overridden:
    void run() override;

private:
    CLMemoryGroup          _memory_group;
    CLDerivative           _derivative;
    CLMagnitudePhaseKernel _mag_phase;
    CLTensor               _gx;
    CLTensor               _gy;
};
}
#endif /*__ARM_COMPUTE_CLHOGGRADIENT_H__ */
