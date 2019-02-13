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
#include "helpers.h"

/** Generate all the region of interests based on the image size and the anchors passed in. For each element (x,y) of the
 * grid, it will generate NUM_ANCHORS rois, given by shifting the grid position to match the anchor.
 *
 * @attention The following variables must be passed at compile time:
 * -# -DDATA_TYPE= Tensor data type. Supported data types: F16/F32
 * -# -DHEIGHT= Height of the feature map on which this kernel is applied
 * -# -DWIDTH= Width of the feature map on which this kernel is applied
 * -# -DNUM_ANCHORS= Number of anchors to be used to generate the rois per each pixel
 * -# -DSTRIDE= Stride to be applied at each different pixel position (i.e., x_range = (1:WIDTH)*STRIDE and y_range = (1:HEIGHT)*STRIDE
 * -# -DNUM_ROI_FIELDS= Number of fields used to represent a roi
 *
 * @param[in]  anchors_ptr                           Pointer to the anchors tensor. Supported data types: F16/F32
 * @param[in]  anchors_stride_x                      Stride of the anchors tensor in X dimension (in bytes)
 * @param[in]  anchors_step_x                        anchors_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  anchors_stride_y                      Stride of the anchors tensor in Y dimension (in bytes)
 * @param[in]  anchors_step_y                        anchors_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  anchors_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  anchors_step_z                        anchors_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  anchors_offset_first_element_in_bytes The offset of the first element in the boxes tensor
 * @param[out] rois_ptr                              Pointer to the rois. Supported data types: same as @p in_ptr
 * @param[out] rois_stride_x                         Stride of the rois in X dimension (in bytes)
 * @param[out] rois_step_x                           pred_boxes_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[out] rois_stride_y                         Stride of the rois in Y dimension (in bytes)
 * @param[out] rois_step_y                           pred_boxes_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[out] rois_stride_z                         Stride of the rois in Z dimension (in bytes)
 * @param[out] rois_step_z                           pred_boxes_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[out] rois_offset_first_element_in_bytes    The offset of the first element in the rois
 */
#if defined(DATA_TYPE) && defined(WIDTH) && defined(HEIGHT) && defined(NUM_ANCHORS) && defined(STRIDE) && defined(NUM_ROI_FIELDS)
__kernel void generate_proposals_compute_all_anchors(
    VECTOR_DECLARATION(anchors),
    VECTOR_DECLARATION(rois))
{
    Vector anchors = CONVERT_TO_VECTOR_STRUCT_NO_STEP(anchors);
    Vector rois    = CONVERT_TO_VECTOR_STRUCT(rois);

    const size_t idx = get_global_id(0);
    // Find the index of the anchor
    const size_t anchor_idx = idx % NUM_ANCHORS;

    // Find which shift is this thread using
    const size_t shift_idx = idx / NUM_ANCHORS;

    // Compute the shift on the X and Y direction (the shift depends exclusively by the index thread id)
    const DATA_TYPE
    shift_x = (DATA_TYPE)(shift_idx % WIDTH) * STRIDE;
    const DATA_TYPE
    shift_y = (DATA_TYPE)(shift_idx / WIDTH) * STRIDE;

    const VEC_DATA_TYPE(DATA_TYPE, NUM_ROI_FIELDS)
    shift = (VEC_DATA_TYPE(DATA_TYPE, NUM_ROI_FIELDS))(shift_x, shift_y, shift_x, shift_y);

    // Read the given anchor
    const VEC_DATA_TYPE(DATA_TYPE, NUM_ROI_FIELDS)
    anchor = vload4(0, (__global DATA_TYPE *)vector_offset(&anchors, anchor_idx * NUM_ROI_FIELDS));

    // Apply the shift to the anchor
    const VEC_DATA_TYPE(DATA_TYPE, NUM_ROI_FIELDS)
    shifted_anchor = anchor + shift;

    vstore4(shifted_anchor, 0, (__global DATA_TYPE *)rois.ptr);
}
#endif //defined(DATA_TYPE) && defined(WIDTH) && defined(HEIGHT) && defined(NUM_ANCHORS) && defined(STRIDE) && defined(NUM_ROI_FIELDS)
