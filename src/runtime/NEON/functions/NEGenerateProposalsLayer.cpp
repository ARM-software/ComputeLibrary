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
#include "arm_compute/runtime/NEON/functions/NEGenerateProposalsLayer.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
NEGenerateProposalsLayer::NEGenerateProposalsLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _permute_deltas_kernel(),
      _flatten_deltas_kernel(),
      _permute_scores_kernel(),
      _flatten_scores_kernel(),
      _compute_anchors_kernel(),
      _bounding_box_kernel(),
      _memset_kernel(),
      _padded_copy_kernel(),
      _cpp_nms_kernel(),
      _is_nhwc(false),
      _deltas_permuted(),
      _deltas_flattened(),
      _scores_permuted(),
      _scores_flattened(),
      _all_anchors(),
      _all_proposals(),
      _keeps_nms_unused(),
      _classes_nms_unused(),
      _proposals_4_roi_values(),
      _num_valid_proposals(nullptr),
      _scores_out(nullptr)
{
}

void NEGenerateProposalsLayer::configure(const ITensor *scores, const ITensor *deltas, const ITensor *anchors, ITensor *proposals, ITensor *scores_out, ITensor *num_valid_proposals,
                                         const GenerateProposalsInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(scores, deltas, anchors, proposals, scores_out, num_valid_proposals);
    ARM_COMPUTE_ERROR_THROW_ON(NEGenerateProposalsLayer::validate(scores->info(), deltas->info(), anchors->info(), proposals->info(), scores_out->info(), num_valid_proposals->info(), info));

    _is_nhwc                         = scores->info()->data_layout() == DataLayout::NHWC;
    const DataType data_type         = deltas->info()->data_type();
    const int      num_anchors       = scores->info()->dimension(get_data_layout_dimension_index(scores->info()->data_layout(), DataLayoutDimension::CHANNEL));
    const int      feat_width        = scores->info()->dimension(get_data_layout_dimension_index(scores->info()->data_layout(), DataLayoutDimension::WIDTH));
    const int      feat_height       = scores->info()->dimension(get_data_layout_dimension_index(scores->info()->data_layout(), DataLayoutDimension::HEIGHT));
    const int      total_num_anchors = num_anchors * feat_width * feat_height;
    const int      pre_nms_topN      = info.pre_nms_topN();
    const int      post_nms_topN     = info.post_nms_topN();
    const size_t   values_per_roi    = info.values_per_roi();

    // Compute all the anchors
    _memory_group.manage(&_all_anchors);
    _compute_anchors_kernel.configure(anchors, &_all_anchors, ComputeAnchorsInfo(feat_width, feat_height, info.spatial_scale()));

    const TensorShape flatten_shape_deltas(values_per_roi, total_num_anchors);
    _deltas_flattened.allocator()->init(TensorInfo(flatten_shape_deltas, 1, data_type));
    _memory_group.manage(&_deltas_flattened);

    // Permute and reshape deltas
    if(!_is_nhwc)
    {
        _memory_group.manage(&_deltas_permuted);
        _permute_deltas_kernel.configure(deltas, &_deltas_permuted, PermutationVector{ 2, 0, 1 });
        _flatten_deltas_kernel.configure(&_deltas_permuted, &_deltas_flattened);
        _deltas_permuted.allocator()->allocate();
    }
    else
    {
        _flatten_deltas_kernel.configure(deltas, &_deltas_flattened);
    }

    const TensorShape flatten_shape_scores(1, total_num_anchors);
    _scores_flattened.allocator()->init(TensorInfo(flatten_shape_scores, 1, data_type));
    _memory_group.manage(&_scores_flattened);
    // Permute and reshape scores
    if(!_is_nhwc)
    {
        _memory_group.manage(&_scores_permuted);
        _permute_scores_kernel.configure(scores, &_scores_permuted, PermutationVector{ 2, 0, 1 });
        _flatten_scores_kernel.configure(&_scores_permuted, &_scores_flattened);
        _scores_permuted.allocator()->allocate();
    }
    else
    {
        _flatten_scores_kernel.configure(scores, &_scores_flattened);
    }

    // Bounding box transform
    _memory_group.manage(&_all_proposals);
    BoundingBoxTransformInfo bbox_info(info.im_width(), info.im_height(), 1.f);
    _bounding_box_kernel.configure(&_all_anchors, &_all_proposals, &_deltas_flattened, bbox_info);
    _deltas_flattened.allocator()->allocate();
    _all_anchors.allocator()->allocate();

    // The original layer implementation first selects the best pre_nms_topN anchors (thus having a lightweight sort)
    // that are then transformed by bbox_transform. The boxes generated are then fed into a non-sorting NMS operation.
    // Since we are reusing the NMS layer and we don't implement any CL/sort, we let NMS do the sorting (of all the input)
    // and the filtering
    const int   scores_nms_size = std::min<int>(std::min<int>(post_nms_topN, pre_nms_topN), total_num_anchors);
    const float min_size_scaled = info.min_size() * info.im_scale();
    _memory_group.manage(&_classes_nms_unused);
    _memory_group.manage(&_keeps_nms_unused);

    // Note that NMS needs outputs preinitialized.
    auto_init_if_empty(*scores_out->info(), TensorShape(scores_nms_size), 1, data_type);
    auto_init_if_empty(*_proposals_4_roi_values.info(), TensorShape(values_per_roi, scores_nms_size), 1, data_type);
    auto_init_if_empty(*num_valid_proposals->info(), TensorShape(scores_nms_size), 1, DataType::U32);

    // Initialize temporaries (unused) outputs
    _classes_nms_unused.allocator()->init(TensorInfo(TensorShape(8, 1), 1, data_type));
    _keeps_nms_unused.allocator()->init(*scores_out->info());

    // Save the output (to map and unmap them at run)
    _scores_out          = scores_out;
    _num_valid_proposals = num_valid_proposals;

    _memory_group.manage(&_proposals_4_roi_values);

    const BoxNMSLimitInfo box_nms_info(0.0f, info.nms_thres(), scores_nms_size, false, NMSType::LINEAR, 0.5f, 0.001f, true, min_size_scaled, info.im_width(), info.im_height());
    _cpp_nms_kernel.configure(&_scores_flattened /*scores_in*/,
                              &_all_proposals /*boxes_in,*/,
                              nullptr /* batch_splits_in*/,
                              scores_out /* scores_out*/,
                              &_proposals_4_roi_values /*boxes_out*/,
                              &_classes_nms_unused /*classes*/,
                              nullptr /*batch_splits_out*/,
                              &_keeps_nms_unused /*keeps*/,
                              num_valid_proposals /* keeps_size*/,
                              box_nms_info);

    _keeps_nms_unused.allocator()->allocate();
    _classes_nms_unused.allocator()->allocate();
    _all_proposals.allocator()->allocate();
    _scores_flattened.allocator()->allocate();

    // Add the first column that represents the batch id. This will be all zeros, as we don't support multiple images
    _padded_copy_kernel.configure(&_proposals_4_roi_values, proposals, PaddingList{ { 1, 0 } });
    _proposals_4_roi_values.allocator()->allocate();

    _memset_kernel.configure(proposals, PixelValue());
}

Status NEGenerateProposalsLayer::validate(const ITensorInfo *scores, const ITensorInfo *deltas, const ITensorInfo *anchors, const ITensorInfo *proposals, const ITensorInfo *scores_out,
                                          const ITensorInfo *num_valid_proposals, const GenerateProposalsInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(scores, deltas, anchors, proposals, scores_out, num_valid_proposals);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(scores, DataLayout::NCHW, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(scores, deltas);

    const int num_anchors       = scores->dimension(get_data_layout_dimension_index(scores->data_layout(), DataLayoutDimension::CHANNEL));
    const int feat_width        = scores->dimension(get_data_layout_dimension_index(scores->data_layout(), DataLayoutDimension::WIDTH));
    const int feat_height       = scores->dimension(get_data_layout_dimension_index(scores->data_layout(), DataLayoutDimension::HEIGHT));
    const int num_images        = scores->dimension(3);
    const int total_num_anchors = num_anchors * feat_width * feat_height;
    const int values_per_roi    = info.values_per_roi();

    ARM_COMPUTE_RETURN_ERROR_ON(num_images > 1);

    TensorInfo all_anchors_info(anchors->clone()->set_tensor_shape(TensorShape(values_per_roi, total_num_anchors)).set_is_resizable(true));
    ARM_COMPUTE_RETURN_ON_ERROR(NEComputeAllAnchorsKernel::validate(anchors, &all_anchors_info, ComputeAnchorsInfo(feat_width, feat_height, info.spatial_scale())));

    TensorInfo deltas_permuted_info = deltas->clone()->set_tensor_shape(TensorShape(values_per_roi * num_anchors, feat_width, feat_height)).set_is_resizable(true);
    TensorInfo scores_permuted_info = scores->clone()->set_tensor_shape(TensorShape(num_anchors, feat_width, feat_height)).set_is_resizable(true);
    if(scores->data_layout() == DataLayout::NHWC)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(deltas, &deltas_permuted_info);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(scores, &scores_permuted_info);
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEPermuteKernel::validate(deltas, &deltas_permuted_info, PermutationVector{ 2, 0, 1 }));
        ARM_COMPUTE_RETURN_ON_ERROR(NEPermuteKernel::validate(scores, &scores_permuted_info, PermutationVector{ 2, 0, 1 }));
    }

    TensorInfo deltas_flattened_info(deltas->clone()->set_tensor_shape(TensorShape(values_per_roi, total_num_anchors)).set_is_resizable(true));
    ARM_COMPUTE_RETURN_ON_ERROR(NEReshapeLayerKernel::validate(&deltas_permuted_info, &deltas_flattened_info));

    TensorInfo scores_flattened_info(scores->clone()->set_tensor_shape(TensorShape(1, total_num_anchors)).set_is_resizable(true));
    TensorInfo proposals_4_roi_values(deltas->clone()->set_tensor_shape(TensorShape(values_per_roi, total_num_anchors)).set_is_resizable(true));

    ARM_COMPUTE_RETURN_ON_ERROR(NEReshapeLayerKernel::validate(&scores_permuted_info, &scores_flattened_info));
    ARM_COMPUTE_RETURN_ON_ERROR(NEBoundingBoxTransformKernel::validate(&all_anchors_info, &proposals_4_roi_values, &deltas_flattened_info, BoundingBoxTransformInfo(info.im_width(), info.im_height(),
                                                                       1.f)));

    ARM_COMPUTE_RETURN_ON_ERROR(NECopyKernel::validate(&proposals_4_roi_values, proposals, PaddingList{ { 0, 1 } }));

    if(num_valid_proposals->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(num_valid_proposals->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(num_valid_proposals->dimension(0) > 1);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(num_valid_proposals, 1, DataType::U32);
    }

    if(proposals->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(proposals->num_dimensions() > 2);
        ARM_COMPUTE_RETURN_ERROR_ON(proposals->dimension(0) != size_t(values_per_roi) + 1);
        ARM_COMPUTE_RETURN_ERROR_ON(proposals->dimension(1) != size_t(total_num_anchors));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(proposals, deltas);
    }

    if(scores_out->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(scores_out->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(scores_out->dimension(0) != size_t(total_num_anchors));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(scores_out, scores);
    }

    return Status{};
}

void NEGenerateProposalsLayer::run()
{
    // Acquire all the temporaries
    MemoryGroupResourceScope scope_mg(_memory_group);

    // Compute all the anchors
    NEScheduler::get().schedule(&_compute_anchors_kernel, Window::DimY);

    // Transpose and reshape the inputs
    if(!_is_nhwc)
    {
        NEScheduler::get().schedule(&_permute_deltas_kernel, Window::DimY);
        NEScheduler::get().schedule(&_permute_scores_kernel, Window::DimY);
    }

    NEScheduler::get().schedule(&_flatten_deltas_kernel, Window::DimY);
    NEScheduler::get().schedule(&_flatten_scores_kernel, Window::DimY);

    // Build the boxes
    NEScheduler::get().schedule(&_bounding_box_kernel, Window::DimY);

    // Non maxima suppression
    CPPScheduler::get().schedule(&_cpp_nms_kernel, Window::DimX);

    // Add dummy batch indexes

    NEScheduler::get().schedule(&_memset_kernel, Window::DimY);
    NEScheduler::get().schedule(&_padded_copy_kernel, Window::DimY);
}
} // namespace arm_compute
