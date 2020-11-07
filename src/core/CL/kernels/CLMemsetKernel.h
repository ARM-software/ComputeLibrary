/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLMEMSETKERNEL_H
#define ARM_COMPUTE_CLMEMSETKERNEL_H

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for filling the planes of a tensor */
class CLMemsetKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLMemsetKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMemsetKernel(const CLMemsetKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMemsetKernel &operator=(const CLMemsetKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLMemsetKernel(CLMemsetKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLMemsetKernel &operator=(CLMemsetKernel &&) = default;
    /** Default destructor */
    ~CLMemsetKernel() = default;

    /** Initialise the kernel's tensor and filling value
     *
     * @param[in,out] tensor         Input tensor to fill. Supported data types: All.
     * @param[in]     constant_value The value used to fill the planes of the tensor
     * @param[in]     window         Window to be used in case setting only part of a tensor. Default is nullptr.
     */
    void configure(ICLTensor *tensor, const PixelValue &constant_value, Window *window = nullptr);
    /** Initialise the kernel's tensor and filling value
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] tensor          Input tensor to fill. Supported data types: All.
     * @param[in]     constant_value  The value used to fill the planes of the tensor
     * @param[in]     window          Window to be used in case setting only part of a tensor. Default is nullptr.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *tensor, const PixelValue &constant_value, Window *window = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLMemsetKernel
     *
     * @param[in] tensor         Source tensor info. Data types supported: All.
     * @param[in] constant_value The value used to fill the planes of the tensor
     * @param[in] window         Window to be used in case setting only part of a tensor. Default is nullptr.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *tensor, const PixelValue &constant_value, Window *window = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    ICLTensor *_tensor;
    Window     _full_window;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLMEMSETRKERNEL_H */
