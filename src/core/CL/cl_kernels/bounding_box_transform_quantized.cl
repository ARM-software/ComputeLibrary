/*
 * Copyright (c) 2019 Arm Limited.
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
#include "helpers_asymm.h"

#if defined(DATA_TYPE) && defined(DATA_TYPE_DELTAS) && defined(WEIGHT_X) && defined(WEIGHT_Y) && defined(WEIGHT_W) && defined(WEIGHT_H) && defined(IMG_WIDTH) && defined(IMG_HEIGHT) && defined(BOX_FIELDS) && defined(SCALE_BEFORE) && defined(OFFSET_BOXES) && defined(SCALE_BOXES) && defined(OFFSET_DELTAS) && defined(SCALE_DELTAS) && defined(OFFSET_PRED_BOXES) && defined(SCALE_PRED_BOXES) // Check for compile time constants

/** Perform a padded copy of input tensor to the output tensor for quantized data types. Padding values are defined at compile time
 *
 * @attention The following variables must be passed at compile time:
 * -# -DDATA_TYPE= Tensor data type. Supported data types: QASYMM16 for boxes and pred_boxes, QASYMM8 for for deltas
 * -# -DWEIGHT{X,Y,W,H}= Weights [wx, wy, ww, wh] for the deltas
 * -# -DIMG_WIDTH= Original image width
 * -# -DIMG_HEIGHT= Original image height
 * -# -DBOX_FIELDS= Number of fields that are used to represent a box in boxes
 *
 * @param[in]  boxes_ptr                                Pointer to the boxes tensor. Supported data types: QASYMM16
 * @param[in]  boxes_stride_x                           Stride of the boxes tensor in X dimension (in bytes)
 * @param[in]  boxes_step_x                             boxes_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  boxes_stride_y                           Stride of the boxes tensor in Y dimension (in bytes)
 * @param[in]  boxes_step_y                             boxes_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  boxes_stride_z                           Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  boxes_step_z                             boxes_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  boxes_offset_first_element_in_bytes      The offset of the first element in the boxes tensor
 * @param[out] pred_boxes_ptr                           Pointer to the predicted boxes. Supported data types: same as @p in_ptr
 * @param[in]  pred_boxes_stride_x                      Stride of the predicted boxes in X dimension (in bytes)
 * @param[in]  pred_boxes_step_x                        pred_boxes_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  pred_boxes_stride_y                      Stride of the predicted boxes in Y dimension (in bytes)
 * @param[in]  pred_boxes_step_y                        pred_boxes_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  pred_boxes_stride_z                      Stride of the predicted boxes in Z dimension (in bytes)
 * @param[in]  pred_boxes_step_z                        pred_boxes_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  pred_boxes_offset_first_element_in_bytes The offset of the first element in the predicted boxes
 * @param[in]  deltas_ptr                               Pointer to the deltas tensor. Supported data types: QASYMM8
 * @param[in]  deltas_stride_x                          Stride of the deltas tensor in X dimension (in bytes)
 * @param[in]  deltas_step_x                            deltas_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  deltas_stride_y                          Stride of the deltas tensor in Y dimension (in bytes)
 * @param[in]  deltas_step_y                            deltas_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  deltas_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  deltas_step_z                            deltas_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  deltas_offset_first_element_in_bytes     The offset of the first element in the deltas tensor
 */
__kernel void bounding_box_transform_quantized(
    VECTOR_DECLARATION(boxes),
    IMAGE_DECLARATION(pred_boxes),
    IMAGE_DECLARATION(deltas))
{
    // Get pixels pointer
    Vector boxes      = CONVERT_TO_VECTOR_STRUCT_NO_STEP(boxes);
    Image  pred_boxes = CONVERT_TO_IMAGE_STRUCT(pred_boxes);
    Image  deltas     = CONVERT_TO_IMAGE_STRUCT(deltas);

    // Load delta and box values into registers
    const float one     = 1.f;
    const float halfone = 0.5f;

    const int py           = get_global_id(1); // box
    float4    scale_before = (float4)SCALE_BEFORE;
    float4 delta           = DEQUANTIZE(vload4(0, (__global DATA_TYPE_DELTAS *)deltas.ptr), OFFSET_DELTAS, SCALE_DELTAS, DATA_TYPE_DELTAS, 4);
    float4 box             = DEQUANTIZE(vload4(0, (__global DATA_TYPE *)vector_offset(&boxes, BOX_FIELDS * py)), OFFSET_BOXES, SCALE_BOXES, DATA_TYPE, 4) / scale_before;

    // Calculate width and centers of the old boxes
    float2 dims    = box.s23 - box.s01 + one;
    float2 ctr     = box.s01 + halfone * dims;
    float4 weights = (float4)(WEIGHT_X, WEIGHT_Y, WEIGHT_W, WEIGHT_H);
    delta /= weights;
    delta.s23 = min(delta.s23, (float)BBOX_XFORM_CLIP);

    // Calculate widths and centers of the new boxes (translation + aspect ratio transformation)
    float2 pred_ctr  = delta.s01 * dims + ctr;
    float2 pred_dims = exp(delta.s23) * dims;

    // Useful vector constant definitions
    float4 max_values = (float4)(IMG_WIDTH - 1, IMG_HEIGHT - 1, IMG_WIDTH - 1, IMG_HEIGHT - 1);
    float4 sign       = (float4)(-1, -1, 1, 1);
    float4 min_values = 0;

    // Calculate the coordinates of the new boxes
    float4 pred_box = pred_ctr.s0101 + sign * halfone * pred_dims.s0101;
#ifdef OFFSET // Possibly adjust the predicted boxes
    pred_box.s23 -= one;
#endif // Possibly adjust the predicted boxes
    pred_box = CLAMP(pred_box, min_values, max_values);
#ifdef SCALE_AFTER // Possibly scale the predicted boxes
    pred_box *= (float4)SCALE_AFTER;
#endif // Possibly scale the predicted boxes

    // Store them into the output
    vstore4(QUANTIZE(pred_box, OFFSET_PRED_BOXES, SCALE_PRED_BOXES, DATA_TYPE, 4), 0, (__global DATA_TYPE *)pred_boxes.ptr);
}
#endif // Check for compile time constants
