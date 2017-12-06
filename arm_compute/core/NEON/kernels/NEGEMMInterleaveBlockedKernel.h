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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVEBLOCKEDKERNEL_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVEBLOCKEDKERNEL_H__

#include "arm_compute/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to interleave the elements of a matrix
 *
 * Interleave_Blocked copies a block of values at a time instead of just one.  The main use of this is the gemmlowp with the "dot product"
 * instruction, where each operation consumes 4 values, so we need to copy blocks of 4 values.
 *
 */
class NEGEMMInterleaveBlockedKernel : public INESimpleKernel
{
public:
    /* Constructor */
    NEGEMMInterleaveBlockedKernel();
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input        Input tensor. Data types supported: U8
     * @param[out] output       Output tensor which stores the interleaved matrix. Data type supported: same as @p input.
     * @param[in]  block_height The height of the blocks to be interleaved.
     * @param[in]  block_width  The width of the blocks to be interleaved.
     * @param[in]  transpose    True if transpose operation must be performed, false otherwise.
     */
    void configure(const ITensor *input, ITensor *output, unsigned int block_height, unsigned int block_width, bool transpose);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMInterleaveBlockedKernel
     *
     * @param[in] input        Input tensor. Data types supported: U8
     * @param[in] output       Output tensor which stores the interleaved matrix. Data type supported: same as @p input.
     * @param[in] block_height The height of the blocks to be interleaved.
     * @param[in] block_width  The width of the blocks to be interleaved.
     * @param[in] transpose    True if transpose operation must be performed, false otherwise.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int block_height, unsigned int block_width, bool transpose);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    unsigned int _block_height;
    unsigned int _block_width;
    bool         _transpose;
};

} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMINTERLEAVEBLOCKEDKERNEL_H__*/
