/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#ifndef __ARM_COMPUTE_NESTACKLAYERKERNEL_H__
#define __ARM_COMPUTE_NESTACKLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to stacks a rank-R tensor into one with rank-(R+1) along the axis dimension.*/
class NEStackLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEStackLayerKernel";
    }
    /** Default constructor */
    NEStackLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEStackLayerKernel(const NEStackLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEStackLayerKernel &operator=(const NEStackLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEStackLayerKernel(NEStackLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEStackLayerKernel &operator=(NEStackLayerKernel &&) = default;
    /** Default destructor */
    ~NEStackLayerKernel() = default;
    /** Initialise the kernel's inputs and output
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in]  input       Input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[in]  axis        The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     * @param[in]  idx_input   Index of the input tensor in the list of tensors to stack.
     *                         All tensors in the list must have the same shape
     * @param[in]  num_tensors Number of tensors to stack
     * @param[out] output      Output tensor. Data types supported: Same as @p input.
     *
     */
    void configure(const ITensor *input, unsigned int axis, unsigned int idx_input, unsigned int num_tensors, ITensor *output);
    /**  Static function to check if given info will lead to a valid configuration of @ref NEStackLayerKernel
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in] input       Input tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[in] axis        The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     * @param[in] idx_input   Index of the input tensor in the list of tensors to stack
     *                        All tensors in the list must have the same shape
     * @param[in] num_tensors Number of tensors to stack
     * @param[in] output      Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, unsigned int axis, unsigned int idx_input, unsigned int num_tensors, const ITensorInfo *output);

    // Inherited methods overridden
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    ITensor       *_output;
    unsigned int   _axis;
    unsigned int   _idx_input;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NESTACKLAYERKERNEL_H__ */
