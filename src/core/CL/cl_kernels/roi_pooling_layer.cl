/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#if DATA_SIZE == 32
#define VEC_SIZE 4
#define VEC_MAX vec4_max
#elif DATA_SIZE == 16
#define VEC_SIZE 8
#define VEC_MAX vec8_max
#else /* DATA_SIZE not equals 32 or 16 */
#error "Unsupported data size"
#endif /* DATA_SIZE == 32 */

inline DATA_TYPE vec4_max(VEC_DATA_TYPE(DATA_TYPE, 4) vec)
{
    VEC_DATA_TYPE(DATA_TYPE, 2)
    temp = fmax(vec.lo, vec.hi);
    return fmax(temp.x, temp.y);
}

inline DATA_TYPE vec8_max(VEC_DATA_TYPE(DATA_TYPE, 8) vec)
{
    VEC_DATA_TYPE(DATA_TYPE, 4)
    temp = fmax(vec.lo, vec.hi);
    return vec4_max(temp);
}

/** Performs a roi pooling on a single output pixel.
 *
 * @param[in] input          Pointer to input Tensor3D struct.
 * @param[in] region_start_x Start x index projected onto the input tensor.
 * @param[in] region_end_x   End x index projected onto the input tensor.
 * @param[in] region_start_y Start y index projected onto the input tensor.
 * @param[in] region_end_y   End y index projected onto the input tensor.
 * @param[in] pz             z index of the input tensor.
 *
 * @return A max pooled value from the region specified in the input tensor.
 */
inline DATA_TYPE roi_pool_1x1(const Tensor3D *input, int region_start_x, int region_end_x, int region_start_y, int region_end_y, int pz)
{
    // Iterate through the pooling region
    if((region_end_x <= region_start_x) || (region_end_y <= region_start_y))
    {
        return (DATA_TYPE)0;
    }
    else
    {
        int num_iter = (int)((region_end_x - region_start_x) / VEC_SIZE);
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        curr_max = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))(-FLT_MAX);
        for(int j = region_start_y; j < region_end_y; ++j)
        {
            int i = region_start_x;
            for(; i < region_start_x + num_iter * VEC_SIZE; i += VEC_SIZE)
            {
                VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
                val      = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(input, i, j, pz));
                curr_max = fmax(val, curr_max);
            }
            for(; i < region_end_x; ++i)
            {
                DATA_TYPE val = *(__global DATA_TYPE *)tensor3D_offset(input, i, j, pz);
                curr_max      = fmax(curr_max, val);
            }
        }
        return (DATA_TYPE)VEC_MAX(curr_max);
    }
}

/** Performs a roi pooling function.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16, F32;
 * @note Datasize must be passed using -DDATA_SIZE e.g. -DDATA_SIZE=32;
 * @note Input dimensions must be passed using -DMAX_DIM_X, -DMAX_DIM_Y and -DMAX_DIM_Z;
 * @note Pooled region dimensions must be passed using -DPOOLED_DIM_X and -DPOOLED_DIM_Y;
 * @note Spatial scale must be passed using -DSPATIAL_SCALE;
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: F16, F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the pooled region of the source image as specifed by ROI
 * @param[in]  rois_ptr                             Pointer to the ROIs tensor. Layout: { batch_index, x1, y1, x2, y2 }. Supported data types: same as @p input_ptr
 * @param[in]  rois_stride_x                        Stride of the ROIs tensor in X dimension (in bytes)
 * @param[in]  rois_step_x                          Step of the ROIs tensor in X dimension (in bytes)
 * @param[in]  rois_stride_y                        Stride of the ROIs tensor in Y dimension (in bytes)
 * @param[in]  rois_step_y                          Step of the ROIs tensor in Y dimension (in bytes)
 * @param[in]  rois_offset_first_element_in_bytes   The offset of the first element in the ROIs tensor
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: F16, F32
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  input_stride_w                       Stride of the source image in W dimension (in bytes)
 * @param[in]  output_stride_w                      Stride of the destination image in W dimension (in bytes)
 */
__kernel void roi_pooling_layer(
    TENSOR3D_DECLARATION(input),
    IMAGE_DECLARATION(rois),
    TENSOR3D_DECLARATION(output),
    unsigned int input_stride_w, unsigned int output_stride_w)
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(input);
    Image    rois   = CONVERT_TO_IMAGE_STRUCT_NO_STEP(rois);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(output);

    const int px = get_global_id(0);
    const int py = get_global_id(1);
    const int pw = get_global_id(2);

    // Load roi parameters
    // roi is laid out as follows { batch_index, x1, y1, x2, y2 }
    const ushort roi_batch = (ushort) * ((__global DATA_TYPE *)offset(&rois, 0, pw));
    const VEC_DATA_TYPE(DATA_TYPE, 4)
    roi               = vload4(0, (__global DATA_TYPE *)offset(&rois, 1, pw));
    const int2 roi_anchor = convert_int2_sat(round(convert_float2(roi.s01) * (float)SPATIAL_SCALE));
    const int2 roi_dims   = convert_int2_sat(fmax(round(convert_float2(roi.s23 - roi.s01) * (float)SPATIAL_SCALE), 1.f));

    // Calculate pooled region start and end
    const float2 spatial_indx     = (float2)(px, py);
    const float2 pooled_dims      = (float2)(POOLED_DIM_X, POOLED_DIM_Y);
    const int2   max_spatial_dims = (int2)(MAX_DIM_X, MAX_DIM_Y);
    int2         region_start     = convert_int2_sat(floor(spatial_indx / pooled_dims * convert_float2(roi_dims))) + roi_anchor;
    int2         region_end       = convert_int2_sat(floor((spatial_indx + 1) / pooled_dims * convert_float2(roi_dims))) + roi_anchor;

    region_start = clamp(region_start, 0, max_spatial_dims);
    region_end   = clamp(region_end, 0, max_spatial_dims);

    // Move input and output pointer across the fourth dimension
    input.ptr += roi_batch * input_stride_w;
    output.ptr += pw * output_stride_w;

    for(int pz = 0; pz < MAX_DIM_Z; ++pz)
    {
        *(__global DATA_TYPE *)tensor3D_offset(&output, px, py, pz) = (__global DATA_TYPE)roi_pool_1x1(&input,
                                                                                                       region_start.x,
                                                                                                       region_end.x,
                                                                                                       region_start.y,
                                                                                                       region_end.y, pz);
    }
}
