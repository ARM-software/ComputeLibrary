/*
 * Copyright (c) 2018-2019 ARM Limited.
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

// This specifies the value to shift the result of roi_dims / pooled_dims before ceiling.
// It is close to the epsilon machine (for a floating point system, x and x+EPS are the same number).
#define EPS_GRID 0.00001f

#if defined(DATA_TYPE) && defined(POOLED_DIM_X) && defined(POOLED_DIM_Y) && defined(MAX_DIM_X) && defined(MAX_DIM_Y) && defined(MAX_DIM_Z) && defined(SPATIAL_SCALE) // Check for compile time constants

/** Performs a roi align on a single output pixel.
 *
 * @param[in] input          Pointer to input Tensor3D struct.
 * @param[in] region_start_x Start x index projected onto the input tensor.
 * @param[in] region_end_x   End x index projected onto the input tensor.
 * @param[in] region_start_y Start y index projected onto the input tensor.
 * @param[in] region_end_y   End y index projected onto the input tensor.
 * @param[in] pz             z index of the input tensor.
 *
 * @return An average pooled value from the region specified in the input tensor.
 */
inline DATA_TYPE roi_align_1x1(const Tensor3D *input, float region_start_x,
                               float bin_size_x,
                               float grid_size_x,
                               float region_end_x,
                               float region_start_y,
                               float bin_size_y,
                               float grid_size_y,
                               float region_end_y,
                               int   pz)
{
    // Iterate through the pooling region
    float sum = 0;
    for(int iy = 0; iy < grid_size_y; ++iy)
    {
        for(int ix = 0; ix < grid_size_x; ++ix)
        {
            // Align the window in the middle of every bin
            const float y = region_start_y + (iy + 0.5f) * bin_size_y / (float)grid_size_y;
            const float x = region_start_x + (ix + 0.5f) * bin_size_x / (float)grid_size_x;

            // Interpolation in the unit square
            const int y_low  = (int)y;
            const int x_low  = (int)x;
            const int y_high = y_low + 1;
            const int x_high = x_low + 1;

            const float ly = y - y_low;
            const float lx = x - x_low;
            const float hy = 1.f - ly;
            const float hx = 1.f - lx;

            const float w1 = hy * hx;
            const float w2 = hy * lx;
            const float w3 = ly * hx;
            const float w4 = ly * lx;
#if defined(NHWC)
            const DATA_TYPE data1 = *(__global DATA_TYPE *)tensor3D_offset(input, pz, x_low, y_low);
            const DATA_TYPE data2 = *(__global DATA_TYPE *)tensor3D_offset(input, pz, x_high, y_low);
            const DATA_TYPE data3 = *(__global DATA_TYPE *)tensor3D_offset(input, pz, x_low, y_high);
            const DATA_TYPE data4 = *(__global DATA_TYPE *)tensor3D_offset(input, pz, x_high, y_high);
#else  // !defined(NHWC)
            const DATA_TYPE data1                 = *(__global DATA_TYPE *)tensor3D_offset(input, x_low, y_low, pz);
            const DATA_TYPE data2                 = *(__global DATA_TYPE *)tensor3D_offset(input, x_high, y_low, pz);
            const DATA_TYPE data3                 = *(__global DATA_TYPE *)tensor3D_offset(input, x_low, y_high, pz);
            const DATA_TYPE data4                 = *(__global DATA_TYPE *)tensor3D_offset(input, x_high, y_high, pz);
#endif // defined(NHWC)
            sum += w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
        }
    }

    return (DATA_TYPE)(sum / (grid_size_x * grid_size_y));
}

/** Performs a roi align function.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16, F32;
 * @note Datasize must be passed using -DDATA_SIZE e.g. -DDATA_SIZE=32;
 * @note Input dimensions must be passed using -DMAX_DIM_X, -DMAX_DIM_Y and -DMAX_DIM_Z;
 * @note Pooled region dimensions must be passed using -DPOOLED_DIM_X and -DPOOLED_DIM_Y;
 * @note Spatial scale must be passed using -DSPATIAL_SCALE;
 * @note Sampling ratio (i.e., the number of samples in each bin) may be passed using -DSAMPLING_RATIO. If not defined each roi
 *       will have a default sampling ratio of roi_dims/pooling_dims
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16, F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the pooled region of the source tensor as specifed by ROI
 * @param[in]  rois_ptr                             Pointer to the ROIs tensor. Layout: { batch_index, x1, y1, x2, y2 }. Supported data types: same as @p input_ptr
 * @param[in]  rois_stride_x                        Stride of the ROIs tensor in X dimension (in bytes)
 * @param[in]  rois_step_x                          Step of the ROIs tensor in X dimension (in bytes)
 * @param[in]  rois_stride_y                        Stride of the ROIs tensor in Y dimension (in bytes)
 * @param[in]  rois_step_y                          Step of the ROIs tensor in Y dimension (in bytes)
 * @param[in]  rois_offset_first_element_in_bytes   The offset of the first element in the ROIs tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  input_stride_w                       Stride of the source tensor in W dimension (in bytes)
 * @param[in]  output_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void roi_align_layer(
    TENSOR3D_DECLARATION(input),
    IMAGE_DECLARATION(rois),
    TENSOR3D_DECLARATION(output),
    unsigned int input_stride_w, unsigned int output_stride_w)
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    Image    rois   = CONVERT_TO_IMAGE_STRUCT_NO_STEP(rois);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);

#if defined(NHWC)
    const int px = get_global_id(1);
    const int py = get_global_id(2);
    const int pw = get_global_id(0);
#else  // !defined(NHWC)
    const int                                  px = get_global_id(0);
    const int                                  py = get_global_id(1);
    const int                                  pw = get_global_id(2);
#endif // defined(NHWC)

    // Load roi parameters
    // roi is laid out as follows { batch_index, x1, y1, x2, y2 }
    const ushort roi_batch = (ushort) * ((__global DATA_TYPE *)offset(&rois, 0, pw));
    const VEC_DATA_TYPE(DATA_TYPE, 4)
    roi                 = vload4(0, (__global DATA_TYPE *)offset(&rois, 1, pw));
    const float2 roi_anchor = convert_float2(roi.s01) * convert_float(SPATIAL_SCALE);
    const float2 roi_dims   = fmax(convert_float2(roi.s23 - roi.s01) * convert_float(SPATIAL_SCALE), 1.f);

    // Calculate pooled region start and end
    const float2 spatial_indx     = (float2)(px, py);
    const float2 pooled_dims      = (float2)(POOLED_DIM_X, POOLED_DIM_Y);
    const float2 max_spatial_dims = (float2)(MAX_DIM_X, MAX_DIM_Y);

    const float2 bin_size     = (float2)((roi_dims.s0 / (float)POOLED_DIM_X), (roi_dims.s1 / (float)POOLED_DIM_Y));
    float2       region_start = spatial_indx * bin_size + roi_anchor;
    float2       region_end   = (spatial_indx + 1) * bin_size + roi_anchor;

    region_start = clamp(region_start, 0, max_spatial_dims);
    region_end   = clamp(region_end, 0, max_spatial_dims);

#if defined(SAMPLING_RATIO)
    const float2 roi_bin_grid = SAMPLING_RATIO;
#else  // !defined(SAMPLING_RATIO)
    // Note that we subtract EPS_GRID before ceiling. This is to avoid situations where 1.000001 gets ceiled to 2.
    const float2   roi_bin_grid = ceil(bin_size - EPS_GRID);
#endif // defined(SAMPLING_RATIO)

    // Move input and output pointer across the fourth dimension
    input.ptr += roi_batch * input_stride_w;
    output.ptr += pw * output_stride_w;
    for(int pz = 0; pz < MAX_DIM_Z; ++pz)
    {
#if defined(NHWC)
        __global DATA_TYPE *_output_ptr = (__global DATA_TYPE *)tensor3D_offset(&output, pz, px, py);
#else  // !defined(NHWC)
        __global DATA_TYPE *_output_ptr  = (__global DATA_TYPE *)tensor3D_offset(&output, px, py, pz);
#endif // defined(NHWC)
        *_output_ptr = (__global DATA_TYPE)roi_align_1x1(&input,
                                                         region_start.x,
                                                         bin_size.x,
                                                         roi_bin_grid.x,
                                                         region_end.x,
                                                         region_start.y,
                                                         bin_size.y,
                                                         roi_bin_grid.y,
                                                         region_end.y, pz);
    }
}
#endif // Check for compile time constants
