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
#ifndef ARM_COMPUTE_NEGENERATEPROPOSALSLAYER_H
#define ARM_COMPUTE_NEGENERATEPROPOSALSLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#include "arm_compute/runtime/CPP/functions/CPPBoxWithNonMaximaSuppressionLimit.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEBoundingBoxTransform.h"
#include "arm_compute/runtime/NEON/functions/NEDequantizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPadLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/NEON/functions/NEQuantizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;
class NEComputeAllAnchorsKernel;

/** Basic function to generate proposals for a RPN (Region Proposal Network)
 *
 * This function calls the following Arm(R) Neon(TM) layers/kernels:
 * -# @ref NEComputeAllAnchorsKernel
 * -# @ref NEPermute x 2
 * -# @ref NEReshapeLayer x 2
 * -# @ref NEBoundingBoxTransform
 * -# @ref NEPadLayerKernel
 * -# @ref NEDequantizationLayer x 2
 * -# @ref NEQuantizationLayer
 * And the following CPP kernels:
 * -# @ref CPPBoxWithNonMaximaSuppressionLimit
 */
class NEGenerateProposalsLayer : public IFunction
{
public:
    /** Default constructor
     *
     * @param[in] memory_manager (Optional) Memory manager.
     */
    NEGenerateProposalsLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGenerateProposalsLayer(const NEGenerateProposalsLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGenerateProposalsLayer &operator=(const NEGenerateProposalsLayer &) = delete;
    /** Default destructor */
    ~NEGenerateProposalsLayer();

    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1               |src2     |dst            |
     * |:--------------|:------------------|:--------|:--------------|
     * |F16            |F16                |F16      |F16            |
     * |F32            |F32                |F32      |F32            |
     * |QASYMM8        |QSYMM8             |QSYMM16  |QASYMM8        |
     *
     * @param[in]  scores              Scores from convolution layer of size (W, H, A), where H and W are the height and width of the feature map, and A is the number of anchors.
     *                                 Data types supported: QASYMM8/F16/F32
     * @param[in]  deltas              Bounding box deltas from convolution layer of size (W, H, 4*A). Data types supported: Same as @p scores
     * @param[in]  anchors             Anchors tensor of size (4, A). Data types supported: QSYMM16 with scale of 0.125 if @p scores is QASYMM8, otherwise same as @p scores
     * @param[out] proposals           Box proposals output tensor of size (5, W*H*A).
     *                                 Data types supported: QASYMM16 with scale of 0.125 and 0 offset if @p scores is QASYMM8, otherwise same as @p scores
     * @param[out] scores_out          Box scores output tensor of size (W*H*A). Data types supported: Same as @p input
     * @param[out] num_valid_proposals Scalar output tensor which says which of the first proposals are valid. Data types supported: U32
     * @param[in]  info                Contains GenerateProposals operation information described in @ref GenerateProposalsInfo
     *
     * @note Only single image prediction is supported. Height and Width (and scale) of the image will be contained in the @ref GenerateProposalsInfo struct.
     * @note Proposals contains all the proposals. Of those, only the first num_valid_proposals are valid.
     */
    void configure(const ITensor *scores, const ITensor *deltas, const ITensor *anchors, ITensor *proposals, ITensor *scores_out, ITensor *num_valid_proposals,
                   const GenerateProposalsInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref NEGenerateProposalsLayer
     *
     * @param[in] scores              Scores info from convolution layer of size (W, H, A), where H and W are the height and width of the feature map, and A is the number of anchors.
     *                                Data types supported: QASYMM8/F16/F32
     * @param[in] deltas              Bounding box deltas info from convolution layer of size (W, H, 4*A). Data types supported: Same as @p scores
     * @param[in] anchors             Anchors tensor info of size (4, A). Data types supported: QSYMM16 with scale of 0.125 if @p scores is QASYMM8, otherwise same as @p scores
     * @param[in] proposals           Box proposals info  output tensor of size (5, W*H*A).
     *                                Data types supported: QASYMM16 with scale of 0.125 and 0 offset if @p scores is QASYMM8, otherwise same as @p scores
     * @param[in] scores_out          Box scores output tensor info of size (W*H*A). Data types supported: Same as @p input
     * @param[in] num_valid_proposals Scalar output tensor info which says which of the first proposals are valid. Data types supported: U32
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
    MemoryGroup _memory_group;

    // kernels/layers
    NEPermute                                  _permute_deltas;
    NEReshapeLayer                             _flatten_deltas;
    NEPermute                                  _permute_scores;
    NEReshapeLayer                             _flatten_scores;
    std::unique_ptr<NEComputeAllAnchorsKernel> _compute_anchors;
    NEBoundingBoxTransform                     _bounding_box;
    NEPadLayer                                 _pad;
    NEDequantizationLayer                      _dequantize_anchors;
    NEDequantizationLayer                      _dequantize_deltas;
    NEQuantizationLayer                        _quantize_all_proposals;

    // CPP functions
    CPPBoxWithNonMaximaSuppressionLimit _cpp_nms;

    bool _is_nhwc;
    bool _is_qasymm8;

    // Temporary tensors
    Tensor _deltas_permuted;
    Tensor _deltas_flattened;
    Tensor _deltas_flattened_f32;
    Tensor _scores_permuted;
    Tensor _scores_flattened;
    Tensor _all_anchors;
    Tensor _all_anchors_f32;
    Tensor _all_proposals;
    Tensor _all_proposals_quantized;
    Tensor _keeps_nms_unused;
    Tensor _classes_nms_unused;
    Tensor _proposals_4_roi_values;

    // Temporary tensor pointers
    Tensor *_all_proposals_to_use;

    // Output tensor pointers
    ITensor *_num_valid_proposals;
    ITensor *_scores_out;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEGENERATEPROPOSALSLAYER_H */
