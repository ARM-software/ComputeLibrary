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

#ifndef ARM_COMPUTE_NEWIDTHCONCATENATELAYERKERNEL_H
#define ARM_COMPUTE_NEWIDTHCONCATENATELAYERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for the width concatenate kernel.
 *  The input tensor will be concatenated into the output tensor.
 */
class NEWidthConcatenateLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEWidthConcatenateLayerKernel";
    }
    /** Default constructor */
    NEWidthConcatenateLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWidthConcatenateLayerKernel(const NEWidthConcatenateLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWidthConcatenateLayerKernel &operator=(const NEWidthConcatenateLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEWidthConcatenateLayerKernel(NEWidthConcatenateLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEWidthConcatenateLayerKernel &operator=(NEWidthConcatenateLayerKernel &&) = default;
    /** Default destructor */
    ~NEWidthConcatenateLayerKernel() = default;
    /** Initialise the kernel's inputs and output
     *
     * @param[in]     input        Input tensor info. Data types supported: All
     * @param[in]     width_offset The offset on the X axis.
     * @param[in,out] output       Output tensor info. Data types supported: Same as @p input.
     */
    void configure(const ITensorInfo *input, unsigned int width_offset, ITensorInfo *output);
    /**  Static function to check if given info will lead to a valid configuration of @ref NEWidthConcatenateLayerKernel
     *
     * @param[in] input        Input tensor info. Data types supported: All
     * @param[in] width_offset The offset on the X axis.
     * @param[in] output       Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, unsigned int width_offset, const ITensorInfo *output);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

private:
    unsigned int _width_offset;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEWIDTHCONCATENATELAYERKERNEL_H */
