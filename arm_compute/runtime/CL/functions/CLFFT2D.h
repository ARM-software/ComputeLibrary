/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLFFT2D_H
#define ARM_COMPUTE_CLFFT2D_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLFFT1D.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
// Forward declaration
class ICLTensor;

/** Basic function to execute two dimensional FFT. This function calls the following OpenCL kernels:
 *
 * -# @ref CLFFT1D 1D FFT is performed on the first given axis
 * -# @ref CLFFT1D 1D FFT is performed on the second given axis
 */
class CLFFT2D : public IFunction
{
public:
    /** Default Constructor */
    CLFFT2D(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied */
    CLFFT2D(const CLFFT2D &) = delete;
    /** Prevent instances of this class from being copied */
    CLFFT2D &operator=(const CLFFT2D &) = delete;
    /** Default move constructor */
    CLFFT2D(CLFFT2D &&) = default;
    /** Default move assignment operator */
    CLFFT2D &operator=(CLFFT2D &&) = default;
    /** Default destructor */
    ~CLFFT2D();
    /** Initialise the function's source, destinations and border mode.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src    |dst    |
     * |:------|:------|
     * |F32    |F32    |
     * |F16    |F16    |
     *
     * @param[in]  input  Source tensor. Data types supported: F16/F32.
     * @param[out] output Destination tensor. Data types and data layouts supported: Same as @p input.
     * @param[in]  config FFT related configuration
     */
    void configure(const ICLTensor *input, ICLTensor *output, const FFT2DInfo &config);
    /** Initialise the function's source, destinations and border mode.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: F16/F32.
     * @param[out] output          Destination tensor. Data types and data layouts supported: Same as @p input.
     * @param[in]  config          FFT related configuration
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const FFT2DInfo &config);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFFT2D.
     *
     * @param[in] input  Source tensor info. Data types supported: F16/F32.
     * @param[in] output Destination tensor info. Data types and data layouts supported: Same as @p input.
     * @param[in] config FFT related configuration
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const FFT2DInfo &config);

    // Inherited methods overridden:
    void run() override;

protected:
    MemoryGroup _memory_group;
    CLFFT1D     _first_pass_func;
    CLFFT1D     _second_pass_func;
    CLTensor    _first_pass_tensor;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLFFT2D_H */
