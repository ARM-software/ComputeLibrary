/*
 * Copyright (c) 2019 ARM Limited.
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

#ifndef __ARM_COMPUTE_NEBATCHCONCATENATEKERNEL_H__
#define __ARM_COMPUTE_NEBATCHCONCATENATEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the batch concatenate kernel.
 *  The input tensor will be concatenated into the output tensor.
 */
class NEBatchConcatenateLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEBatchConcatenateLayerKernel";
    }
    /** Default constructor */
    NEBatchConcatenateLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchConcatenateLayerKernel(const NEBatchConcatenateLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchConcatenateLayerKernel &operator=(const NEBatchConcatenateLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEBatchConcatenateLayerKernel(NEBatchConcatenateLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEBatchConcatenateLayerKernel &operator=(NEBatchConcatenateLayerKernel &&) = default;
    /** Default destructor */
    ~NEBatchConcatenateLayerKernel() = default;
    /** Initialise the kernel's inputs and output
     *
     * @param[in]     input        Input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]     batch_offset The offset on axis # 3.
     * @param[in,out] output       Output tensor. Data types supported: Same as @p input.
     *
     * @note: The output tensor's low two dimensions can't be smaller than the input one's.
     * @note: The gaps between the two lowest dimensions of input and output need to be divisible by 2.
     *
     */
    void configure(const ITensor *input, unsigned int batch_offset, ITensor *output);
    /**  Static function to check if given info will lead to a valid configuration of @ref NEBatchConcatenateLayerKernel
     *
     * @param[in] input        Input tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] batch_offset The offset on axis # 3.
     * @param[in] output       Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, unsigned int batch_offset, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    using BatchConcatFunction = void(const ITensor *in, ITensor *out, int batch_offset, const Window &window);

private:
    BatchConcatFunction *_func;
    const ITensor       *_input;
    ITensor             *_output;
    unsigned int         _batch_offset;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEBATCHCONCATENATEKERNEL_H__ */
