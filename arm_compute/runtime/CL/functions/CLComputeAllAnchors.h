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
#ifndef __ARM_COMPUTE_CLCOMPUTEALLANCHORS_H__
#define __ARM_COMPUTE_CLCOMPUTEALLANCHORS_H__

#include "arm_compute/core/CL/kernels/CLGenerateProposalsLayerKernel.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLComputeAllAnchorsKernel.
 *
 * This function calls the following OpenCL kernels:
 * -# @ref CLComputeAllAnchorsKernel
 */
class CLComputeAllAnchors : public ICLSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * @param[in]  anchors     Source tensor. Original set of anchors of size (4, A) where A is the number of anchors. Data types supported: F16/F32
     * @param[out] all_anchors Destination tensor. Destination anchors of size (4, H*W*A) where H and W are the height and width of the feature map and A is the number of anchors. Data types supported: Same as @p input
     * @param[in]  info        Contains Compute Anchors operation information described in @ref ComputeAnchorsInfo
     *
     */
    void configure(const ICLTensor *anchors, ICLTensor *all_anchors, const ComputeAnchorsInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref CLComputeAllAnchorsKernel
     *
     * @param[in] anchors     Source tensor info. Original set of anchors of size (4, A) where A is the number of anchors. Data types supported: F16/F32
     * @param[in] all_anchors Destination tensor info. Destination anchors of size (4, H*W*A) where H and W are the height and width of the feature map and A is the number of anchors. Data types supported: Same as @p input
     * @param[in] info        Contains Compute Anchors operation information described in @ref ComputeAnchorsInfo
     *
     * @return a Status
     */
    static Status validate(const ITensorInfo *anchors, const ITensorInfo *all_anchors, const ComputeAnchorsInfo &info);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLCOMPUTEALLANCOHORS_H__ */
