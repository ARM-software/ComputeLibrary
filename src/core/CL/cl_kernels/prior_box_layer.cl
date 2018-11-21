/*
 * Copyright (c) 2018 ARM Limited.
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

#if defined(DATA_TYPE) && defined(WIDTH) && defined(HEIGHT) && defined(LAYER_WIDTH) && defined(LAYER_HEIGHT) && defined(OFFSET) && defined(STEP_X) && defined(STEP_Y) && defined(NUM_PRIORS) && defined(VARIANCE_0) && defined(VARIANCE_1) && defined(VARIANCE_2) && defined(VARIANCE_3)

/**  Compute prior boxes and clip (NCHW)
 *
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: F32
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  idx                                  Index to write to
 * @param[in]  center_x                             Center value of the x axis
 * @param[in]  center_y                             Center value of the y axis
 * @param[in]  box_width                            Prior box width
 * @param[in]  box_height                           Prior box height
 *
 */
inline void calculate_xy_min_max_nchw(Image *out, int idx, float center_x, float center_y, float box_width, float box_height)
{
    float xmin = (center_x - box_width / 2.f) / WIDTH;
    float ymin = (center_y - box_height / 2.f) / HEIGHT;
    float xmax = (center_x + box_width / 2.f) / WIDTH;
    float ymax = (center_y + box_height / 2.f) / HEIGHT;

#if defined(CLIP)
    xmin = clamp(xmin, 0.f, 1.f);
    ymin = clamp(ymin, 0.f, 1.f);
    xmax = clamp(xmax, 0.f, 1.f);
    ymax = clamp(ymax, 0.f, 1.f);
#endif // defined(CLIP)

    // Store result
    vstore4((VEC_DATA_TYPE(DATA_TYPE, 4))(xmin, ymin, xmax, ymax), 0, ((__global DATA_TYPE *)offset(out, idx + 0, 0)));
}

/** Compute prior boxes (NCHW)
 *
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: F32
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  min_size                             Prior box min size
 * @param[in]  min_idx                              Index of the min vector
 * @param[in]  idx                                  Index to write to
 *
 * @return The updated index
 */
inline int calculate_min_nchw(Image *out, __global float *max, __global float *aspect_ratios, int max_size, int aspect_ratios_size, float min_size, int min_idx, int idx)
{
    const float center_x = ((float)(get_global_id(0) % LAYER_WIDTH) + OFFSET) * STEP_X;
    const float center_y = ((float)(get_global_id(0) / LAYER_WIDTH) + OFFSET) * STEP_Y;

    float box_width  = min_size;
    float box_height = min_size;
    calculate_xy_min_max_nchw(out, idx, center_x, center_y, box_width, box_height);
    idx += 4;

    if(max_size > 0)
    {
        box_width  = sqrt(min_size * max[min_idx]);
        box_height = box_width;
        calculate_xy_min_max_nchw(out, idx, center_x, center_y, box_width, box_height);
        idx += 4;
    }
    for(unsigned int i = 0; i < aspect_ratios_size; ++i)
    {
        if(fabs(aspect_ratios[i] - 1.f) < 1e-6f)
        {
            continue;
        }
        box_width  = min_size * sqrt(aspect_ratios[i]);
        box_height = min_size * rsqrt(aspect_ratios[i]);

        calculate_xy_min_max_nchw(out, idx, center_x, center_y, box_width, box_height);
        idx += 4;
    }

    return idx;
}
/** Calculate prior boxes with NCHW format.
 *
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: F32
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  min                                  The minimum values
 * @param[in]  max                                  The maximum_values
 * @param[in]  aspect_ratios                        The aspect ratio values
 * @param[in]  min_size                             The minimum values size
 * @param[in]  max_size                             The maximum_values values size
 * @param[in]  aspect_ratios_size                   The aspect ratio values size
 */
__kernel void prior_box_layer_nchw(IMAGE_DECLARATION(output), __global float *min, __global float *max, __global float *aspect_ratios, unsigned int min_size, unsigned int max_size,
                                   unsigned int aspect_ratios_size)
{
    Image out = CONVERT_TO_IMAGE_STRUCT(output);

    int idx = 0;
    for(unsigned int i = 0; i < min_size; ++i)
    {
        idx = calculate_min_nchw(&out, max, aspect_ratios, max_size, aspect_ratios_size, min[i], i, idx);
    }

    // Store variances
    for(int i = 0; i < (NUM_PRIORS * 4); i += 4)
    {
        vstore4((VEC_DATA_TYPE(DATA_TYPE, 4))(VARIANCE_0, VARIANCE_1, VARIANCE_2, VARIANCE_3), 0, ((__global DATA_TYPE *)offset(&out, i, 1)));
    }
}
#endif /* defined(DATA_TYPE) && defined(WIDTH) && defined(HEIGHT) && defined(LAYER_WIDTH) && defined(LAYER_HEIGHT) && defined(OFFSET) && defined(STEP_X) && defined(STEP_Y) && defined(NUM_PRIORS) && defined(VARIANCE_0) && defined(VARIANCE_1) && defined(VARIANCE_2) && defined(VARIANCE_3) */
