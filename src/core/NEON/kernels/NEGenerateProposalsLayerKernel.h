/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGENERATEPROPOSALSLAYERKERNEL_H
#define ARM_COMPUTE_NEGENERATEPROPOSALSLAYERKERNEL_H

#include "src/core/NEON/INEKernel.h"
namespace arm_compute
{
class ITensor;

/** Interface for Compute All Anchors kernel */
class NEComputeAllAnchorsKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEComputeAllAnchorsKernel";
    }

    /** Default constructor */
    NEComputeAllAnchorsKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEComputeAllAnchorsKernel(const NEComputeAllAnchorsKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEComputeAllAnchorsKernel &operator=(const NEComputeAllAnchorsKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEComputeAllAnchorsKernel(NEComputeAllAnchorsKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEComputeAllAnchorsKernel &operator=(NEComputeAllAnchorsKernel &&) = default;
    /** Default destructor */
    ~NEComputeAllAnchorsKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  anchors     Source tensor. Original set of anchors of size (4, A), where A is the number of anchors. Data types supported: QSYMM16/F16/F32
     * @param[out] all_anchors Destination tensor. Destination anchors of size (4, H*W*A) where H and W are the height and width of the feature map and A is the number of anchors. Data types supported: Same as @p input
     * @param[in]  info        Contains Compute Anchors operation information described in @ref ComputeAnchorsInfo
     *
     */
    void configure(const ITensor *anchors, ITensor *all_anchors, const ComputeAnchorsInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref NEComputeAllAnchorsKernel
     *
     * @param[in] anchors     Source tensor info. Original set of anchors of size (4, A), where A is the number of anchors. Data types supported: QSYMM16/F16/F32
     * @param[in] all_anchors Destination tensor info. Destination anchors of size (4, H*W*A) where H and W are the height and width of the feature map and A is the number of anchors. Data types supported: Same as @p input
     * @param[in] info        Contains Compute Anchors operation information described in @ref ComputeAnchorsInfo
     *
     * @return a Status
     */
    static Status validate(const ITensorInfo *anchors, const ITensorInfo *all_anchors, const ComputeAnchorsInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor     *_anchors;
    ITensor           *_all_anchors;
    ComputeAnchorsInfo _anchors_info;
};
} // arm_compute
#endif // ARM_COMPUTE_NEGENERATEPROPOSALSLAYERKERNEL_H
