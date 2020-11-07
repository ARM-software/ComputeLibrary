/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLGENERATEPROPOSALSLAYERKERNEL_H
#define ARM_COMPUTE_CLGENERATEPROPOSALSLAYERKERNEL_H

#include "src/core/CL/ICLKernel.h"
namespace arm_compute
{
class ICLTensor;

/** Interface for Compute All Anchors kernel */
class CLComputeAllAnchorsKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLComputeAllAnchorsKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLComputeAllAnchorsKernel(const CLComputeAllAnchorsKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLComputeAllAnchorsKernel &operator=(const CLComputeAllAnchorsKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLComputeAllAnchorsKernel(CLComputeAllAnchorsKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLComputeAllAnchorsKernel &operator=(CLComputeAllAnchorsKernel &&) = default;
    /** Default destructor */
    ~CLComputeAllAnchorsKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  anchors     Source tensor. Original set of anchors of size (4, A), where A is the number of anchors. Data types supported: QSYMM16/F16/F32
     * @param[out] all_anchors Destination tensor. Destination anchors of size (4, H*W*A) where H and W are the height and width of the feature map and A is the number of anchors. Data types supported: Same as @p input
     * @param[in]  info        Contains Compute Anchors operation information described in @ref ComputeAnchorsInfo
     *
     */
    void configure(const ICLTensor *anchors, ICLTensor *all_anchors, const ComputeAnchorsInfo &info);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  anchors         Source tensor. Original set of anchors of size (4, A), where A is the number of anchors. Data types supported: QSYMM16/F16/F32
     * @param[out] all_anchors     Destination tensor. Destination anchors of size (4, H*W*A) where H and W are the height and width of the feature map and A is the number of anchors. Data types supported: Same as @p input
     * @param[in]  info            Contains Compute Anchors operation information described in @ref ComputeAnchorsInfo
     *
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *anchors, ICLTensor *all_anchors, const ComputeAnchorsInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref CLComputeAllAnchorsKernel
     *
     * @param[in] anchors     Source tensor info. Original set of anchors of size (4, A), where A is the number of anchors. Data types supported: QSYMM16/F16/F32
     * @param[in] all_anchors Destination tensor info. Destination anchors of size (4, H*W*A) where H and W are the height and width of the feature map and A is the number of anchors. Data types supported: Same as @p input
     * @param[in] info        Contains Compute Anchors operation information described in @ref ComputeAnchorsInfo
     *
     * @return a Status
     */
    static Status validate(const ITensorInfo *anchors, const ITensorInfo *all_anchors, const ComputeAnchorsInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_anchors;
    ICLTensor       *_all_anchors;
};
} // arm_compute
#endif // ARM_COMPUTE_CLGENERATEPROSPOSALSLAYERKERNEL_H
