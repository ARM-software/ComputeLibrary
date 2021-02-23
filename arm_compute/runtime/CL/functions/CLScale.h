/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLSCALE_H
#define ARM_COMPUTE_CLSCALE_H

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLRuntimeContext.h"
#include "arm_compute/runtime/IFunction.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLScaleKernel.h"

#include <cstdint>

namespace arm_compute
{
// Forward declarations
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref CLScaleKernel */
class CLScale : public IFunction
{
public:
    /** Default Constructor */
    CLScale();
    /** Default Destructor */
    ~CLScale() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScale(const CLScale &) = delete;
    /** Default move constructor */
    CLScale(CLScale &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScale &operator=(const CLScale &) = delete;
    /** Default move assignment operator */
    CLScale &operator=(CLScale &&) = default;

    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in,out] input  Source tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output Destination tensor. Data types supported: Same as @p input
     *                       All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]     info   @ref ScaleKernelInfo descriptor to be used to configure
     */
    void configure(ICLTensor *input, ICLTensor *output, const ScaleKernelInfo &info);
    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] input           Source tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output          Destination tensor. Data types supported: Same as @p input
     *                                All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]     info            @ref ScaleKernelInfo descriptor to be used to configure
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const ScaleKernelInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref CLScale
     *
     * @param[in] input  Source tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/F32.
     * @param[in] output Output tensor info. Data type supported: Same as @p input
     *                   All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in] info   @ref ScaleKernelInfo descriptor to be used to validate
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ScaleKernelInfo &info);

    // Inherited methods overridden:
    void run() override;

protected:
    std::unique_ptr<CLFillBorderKernel> _border_handler;
    std::unique_ptr<CLScaleKernel>      _kernel;
};
}
#endif /*ARM_COMPUTE_CLSCALE_H */
