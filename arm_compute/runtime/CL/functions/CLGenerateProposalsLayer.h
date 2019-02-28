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
#ifndef __ARM_COMPUTE_CLGENERATEPROPOSALSLAYER_H__
#define __ARM_COMPUTE_CLGENERATEPROPOSALSLAYER_H__
#include "arm_compute/core/CL/kernels/CLBoundingBoxTransformKernel.h"
#include "arm_compute/core/CL/kernels/CLCopyKernel.h"
#include "arm_compute/core/CL/kernels/CLGenerateProposalsLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLMemsetKernel.h"
#include "arm_compute/core/CL/kernels/CLPermuteKernel.h"
#include "arm_compute/core/CL/kernels/CLReshapeLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLStridedSliceKernel.h"
#include "arm_compute/core/CPP/kernels/CPPBoxWithNonMaximaSuppressionLimitKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to generate proposals for a RPN (Region Proposal Network)
 *
 * This function calls the following OpenCL kernels:
 * -# @ref CLComputeAllAnchors
 * -# @ref CLPermute x 2
 * -# @ref CLReshapeLayer x 2
 * -# @ref CLStridedSlice x 3
 * -# @ref CLBoundingBoxTransform
 * -# @ref CLCopyKernel
 * -# @ref CLMemsetKernel
 * And the following CPP kernels:
 * -# @ref CPPBoxWithNonMaximaSuppressionLimit
 */
class CLGenerateProposalsLayer : public IFunction
{
public:
    /** Default constructor
     *
     * @param[in] memory_manager (Optional) Memory manager.
     */
    CLGenerateProposalsLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGenerateProposalsLayer(const CLGenerateProposalsLayer &) = delete;
    /** Default move constructor */
    CLGenerateProposalsLayer(CLGenerateProposalsLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGenerateProposalsLayer &operator=(const CLGenerateProposalsLayer &) = delete;
    /** Default move assignment operator */
    CLGenerateProposalsLayer &operator=(CLGenerateProposalsLayer &&) = default;

    /** Set the input and output tensors.
     *
     * @param[in]  scores              Scores from convolution layer of size (W, H, A), where H and W are the height and width of the feature map, and A is the number of anchors. Data types supported: F16/F32
     * @param[in]  deltas              Bounding box deltas from convolution layer of size (W, H, 4*A). Data types supported: Same as @p scores
     * @param[in]  anchors             Anchors tensor of size (4, A). Data types supported: Same as @p input
     * @param[out] proposals           Box proposals output tensor of size (5, W*H*A). Data types supported: Same as @p input
     * @param[out] scores_out          Box scores output tensor of size (W*H*A). Data types supported: Same as @p input
     * @param[out] num_valid_proposals Scalar output tensor which says which of the first proposals are valid. Data types supported: U32
     * @param[in]  info                Contains GenerateProposals operation information described in @ref GenerateProposalsInfo
     *
     * @note Only single image prediction is supported. Height and Width (and scale) of the image will be contained in the @ref GenerateProposalsInfo struct.
     * @note Proposals contains all the proposals. Of those, only the first num_valid_proposals are valid.
     */
    void configure(const ICLTensor *scores, const ICLTensor *deltas, const ICLTensor *anchors, ICLTensor *proposals, ICLTensor *scores_out, ICLTensor *num_valid_proposals,
                   const GenerateProposalsInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref CLGenerateProposalsLayer
     *
     * @param[in] scores              Scores info from convolution layer of size (W, H, A), where H and W are the height and width of the feature map, and A is the number of anchors. Data types supported: F16/F32
     * @param[in] deltas              Bounding box deltas info from convolution layer of size (W, H, 4*A). Data types supported: Same as @p scores
     * @param[in] anchors             Anchors tensor info of size (4, A). Data types supported: Same as @p input
     * @param[in] proposals           Box proposals info  output tensor of size (5, W*H*A). Data types supported: Data types supported: U32
     * @param[in] scores_out          Box scores output tensor info of size (W*H*A). Data types supported: Same as @p input
     * @param[in] num_valid_proposals Scalar output tensor info which says which of the first proposals are valid. Data types supported: Same as @p input
     * @param[in] info                Contains GenerateProposals operation information described in @ref GenerateProposalsInfo
     *
     * @return a Status
     */
    static Status validate(const ITensorInfo *scores, const ITensorInfo *deltas, const ITensorInfo *anchors, const ITensorInfo *proposals, const ITensorInfo *scores_out,
                           const ITensorInfo           *num_valid_proposals,
                           const GenerateProposalsInfo &info);

    // Inherited methods overridden:
    void run() override;

private:
    // Memory group manager
    CLMemoryGroup _memory_group;

    // OpenCL kernels
    CLPermuteKernel              _permute_deltas_kernel;
    CLReshapeLayerKernel         _flatten_deltas_kernel;
    CLPermuteKernel              _permute_scores_kernel;
    CLReshapeLayerKernel         _flatten_scores_kernel;
    CLComputeAllAnchorsKernel    _compute_anchors_kernel;
    CLBoundingBoxTransformKernel _bounding_box_kernel;
    CLMemsetKernel               _memset_kernel;
    CLCopyKernel                 _padded_copy_kernel;

    // CPP kernels
    CPPBoxWithNonMaximaSuppressionLimitKernel _cpp_nms_kernel;

    bool _is_nhwc;

    // Temporary tensors
    CLTensor _deltas_permuted;
    CLTensor _deltas_flattened;
    CLTensor _scores_permuted;
    CLTensor _scores_flattened;
    CLTensor _all_anchors;
    CLTensor _all_proposals;
    CLTensor _keeps_nms_unused;
    CLTensor _classes_nms_unused;
    CLTensor _proposals_4_roi_values;

    // Output tensor pointers
    ICLTensor *_num_valid_proposals;
    ICLTensor *_scores_out;

    /** Internal function to run the CPP BoxWithNMS kernel */
    void run_cpp_nms_kernel();
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLGENERATEPROPOSALSLAYER_H__ */
