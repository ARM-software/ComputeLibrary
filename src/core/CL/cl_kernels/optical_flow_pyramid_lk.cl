/*
 * Copyright (c) 2017 ARM Limited.
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
#include "types.h"

/*
 *The criteria for lost tracking is that the spatial gradient matrix has:
 * - Determinant less than DETERMINANT_THR
 * - or minimum eigenvalue is smaller then EIGENVALUE_THR
 *
 * The thresholds for the determinant and the minimum eigenvalue is
 * defined by the OpenVX spec
 *
 * Note: Also lost tracking happens when the point tracked coordinate is outside
 * the image coordinates
 *
 * https://www.khronos.org/registry/vx/specs/1.0/html/d0/d0c/group__group__vision__function__opticalflowpyrlk.html
 */

/* Internal Lucas-Kanade Keypoint struct */
typedef struct InternalKeypoint
{
    float x;               /**< The x coordinate. */
    float y;               /**< The y coordinate. */
    float tracking_status; /**< A zero indicates a lost point. Initialized to 1 by corner detectors. */
    float dummy;
} InternalKeypoint;

/** Threshold for the determinant. Used for lost tracking criteria */
#define DETERMINANT_THR 1.0e-07f

/** Thresholds for minimum eigenvalue. Used for lost tracking criteria */
#define EIGENVALUE_THR 1.0e-04f

/** Constants used for Lucas-Kanade Algorithm */
#define W_BITS (14)
#define FLT_SCALE (1.0f / (float)(1 << 20))
#define D0 ((float)(1 << W_BITS))
#define D1 (1.0f / (float)(1 << (W_BITS - 5)))

/** Initializes the internal new points array when the level of pyramid is NOT equal to max.
 *
 * @param[in,out] old_points_internal An array of internal key points that are defined at the old_images high resolution pyramid.
 * @param[in,out] new_points_internal An array of internal key points that are defined at the new_images high resolution pyramid.
 * @param[in]     scale               Scale factor to apply for the new_point coordinates.
 */
__kernel void init_level(
    __global float4 *old_points_internal,
    __global float4 *new_points_internal,
    const float      scale)
{
    int idx = get_global_id(0);

    // Get old and new keypoints
    float4 old_point = old_points_internal[idx];
    float4 new_point = new_points_internal[idx];

    // Scale accordingly with the pyramid_scale
    old_point.xy *= (float2)(2.0f);
    new_point.xy *= (float2)(2.0f);

    old_points_internal[idx] = old_point;
    new_points_internal[idx] = new_point;
}

/** Initializes the internal new points array when the level of pyramid is equal to max.
 *
 * @param[in]     old_points          An array of key points that are defined at the old_images high resolution pyramid.
 * @param[in,out] old_points_internal An array of internal key points that are defined at the old_images high resolution pyramid.
 * @param[out]    new_points_internal An array of internal key points that are defined at the new_images high resolution pyramid.
 * @param[in]     scale               Scale factor to apply for the new_point coordinates.
 */
__kernel void init_level_max(
    __global Keypoint *old_points,
    __global InternalKeypoint *old_points_internal,
    __global InternalKeypoint *new_points_internal,
    const float                scale)
{
    int idx = get_global_id(0);

    Keypoint old_point = old_points[idx];

    // Get old keypoint to track
    InternalKeypoint old_point_internal;
    old_point_internal.x               = old_point.x * scale;
    old_point_internal.y               = old_point.y * scale;
    old_point_internal.tracking_status = 1.f;

    // Store internal keypoints
    old_points_internal[idx] = old_point_internal;
    new_points_internal[idx] = old_point_internal;
}

/** Initializes the new_points array when the level of pyramid is equal to max and if use_initial_estimate = 1.
 *
 * @param[in]     old_points           An array of key points that are defined at the old_images high resolution pyramid.
 * @param[in]     new_points_estimates An array of estimate key points that are defined at the old_images high resolution pyramid.
 * @param[in,out] old_points_internal  An array of internal key points that are defined at the old_images high resolution pyramid.
 * @param[out]    new_points_internal  An array of internal key points that are defined at the new_images high resolution pyramid.
 * @param[in]     scale                Scale factor to apply for the new_point coordinates.
 */
__kernel void init_level_max_initial_estimate(
    __global Keypoint *old_points,
    __global Keypoint *new_points_estimates,
    __global InternalKeypoint *old_points_internal,
    __global InternalKeypoint *new_points_internal,
    const float                scale)
{
    int idx = get_global_id(0);

    Keypoint         old_point          = old_points[idx];
    Keypoint         new_point_estimate = new_points_estimates[idx];
    InternalKeypoint old_point_internal;
    InternalKeypoint new_point_internal;

    // Get old keypoint to track
    old_point_internal.x               = old_point.x * scale;
    old_point_internal.y               = old_point.y * scale;
    old_point_internal.tracking_status = 1.f;

    // Get new keypoint to track
    new_point_internal.x               = new_point_estimate.x * scale;
    new_point_internal.y               = new_point_estimate.y * scale;
    new_point_internal.tracking_status = new_point_estimate.tracking_status;

    // Store internal keypoints
    old_points_internal[idx] = old_point_internal;
    new_points_internal[idx] = new_point_internal;
}

/** Truncates the coordinates stored in new_points array
 *
 * @param[in]  new_points_internal An array of estimate key points that are defined at the new_images high resolution pyramid.
 * @param[out] new_points          An array of internal key points that are defined at the new_images high resolution pyramid.
 */
__kernel void finalize(
    __global InternalKeypoint *new_points_internal,
    __global Keypoint *new_points)
{
    int idx = get_global_id(0);

    // Load internal keypoint
    InternalKeypoint new_point_internal = new_points_internal[idx];

    // Calculate output point
    Keypoint new_point;
    new_point.x               = round(new_point_internal.x);
    new_point.y               = round(new_point_internal.y);
    new_point.tracking_status = new_point_internal.tracking_status;

    // Store new point
    new_points[idx] = new_point;
}

/** Computes A11, A12, A22, min_eig, ival, ixval and iyval at level 0th of the pyramid. These values will be used in step 1.
 *
 * @param[in]      old_image_ptr                               Pointer to the input old image. Supported data types: U8
 * @param[in]      old_image_stride_x                          Stride of the input old image in X dimension (in bytes)
 * @param[in]      old_image_step_x                            old_image_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]      old_image_stride_y                          Stride of the input old image in Y dimension (in bytes)
 * @param[in]      old_image_step_y                            old_image_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]      old_image_offset_first_element_in_bytes     The offset of the first element in the input old image
 * @param[in]      old_scharr_gx_ptr                           Pointer to the input scharr x image. Supported data types: S16
 * @param[in]      old_scharr_gx_stride_x                      Stride of the input scharr x image in X dimension (in bytes)
 * @param[in]      old_scharr_gx_step_x                        old_scharr_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]      old_scharr_gx_stride_y                      Stride of the input scharr x image in Y dimension (in bytes)
 * @param[in]      old_scharr_gx_step_y                        old_scharr_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]      old_scharr_gx_offset_first_element_in_bytes The offset of the first element in the input scharr x image
 * @param[in]      old_scharr_gy_ptr                           Pointer to the input scharr y image. Supported data types: S16
 * @param[in]      old_scharr_gy_stride_x                      Stride of the input scharr y image in X dimension (in bytes)
 * @param[in]      old_scharr_gy_step_x                        old_scharr_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]      old_scharr_gy_stride_y                      Stride of the input scharr y image in Y dimension (in bytes)
 * @param[in]      old_scharr_gy_step_y                        old_scharr_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]      old_scharr_gy_offset_first_element_in_bytes The offset of the first element in the input scharr y image
 * @param[in]      old_points                                  An array of key points. Those key points are defined at the old_images high resolution pyramid
 * @param[in, out] new_points                                  An output array of key points. Those key points are defined at the new_images high resolution pyramid
 * @param[out]     coeff                                       It stores | A11 | A12 | A22 | min_eig | for each keypoint
 * @param[out]     iold_val                                    It stores | ival | ixval | iyval | dummy | for each point in the window centered on old_keypoint
 * @param[in]      window_dimension                            The size of the window on which to perform the algorithm
 * @param[in]      window_dimension_pow2                       The squared size of the window on which to perform the algorithm
 * @param[in]      half_window                                 The half size of the window on which to perform the algorithm
 * @param[in]      border_limits                               It stores the right border limit (width - window_dimension - 1, height - window_dimension - 1,)
 * @param[in]      eig_const                                   1.0f / (float)(2.0f * window_dimension * window_dimension)
 * @param[in]      level0                                      It is set to 1 if level 0 of the pyramid
 */
void __kernel lktracker_stage0(
    IMAGE_DECLARATION(old_image),
    IMAGE_DECLARATION(old_scharr_gx),
    IMAGE_DECLARATION(old_scharr_gy),
    __global float4 *old_points,
    __global float4 *new_points,
    __global float4 *coeff,
    __global short4 *iold_val,
    const int        window_dimension,
    const int        window_dimension_pow2,
    const int        half_window,
    const float3     border_limits,
    const float      eig_const,
    const int        level0)
{
    int idx = get_global_id(0);

    Image old_image     = CONVERT_TO_IMAGE_STRUCT_NO_STEP(old_image);
    Image old_scharr_gx = CONVERT_TO_IMAGE_STRUCT_NO_STEP(old_scharr_gx);
    Image old_scharr_gy = CONVERT_TO_IMAGE_STRUCT_NO_STEP(old_scharr_gy);

    // Get old keypoint
    float2 old_keypoint = old_points[idx].xy - (float2)half_window;

    // Get the floor value
    float2 iold_keypoint = floor(old_keypoint);

    // Check if using the window dimension we can go out of boundary in the following for loops. If so, invalidate the tracked point
    if(any(iold_keypoint < border_limits.zz) || any(iold_keypoint >= border_limits.xy))
    {
        if(level0 == 1)
        {
            // Invalidate tracked point as we are at level 0
            new_points[idx].s2 = 0.0f;
        }

        // Not valid coordinate. It sets min_eig to 0.0f
        coeff[idx].s3 = 0.0f;

        return;
    }

    // Compute weight for the bilinear interpolation
    float2 ab = old_keypoint - iold_keypoint;

    // Weight used for Bilinear-Interpolation on Scharr images
    // w_scharr.s0 = (1.0f - ab.x) * (1.0f - ab.y)
    // w_scharr.s1 = ab.x * (1.0f - ab.y)
    // w_scharr.s2 = (1.0f - ab.x) * ab.y
    // w_scharr.s3 = ab.x * ab.y

    float4 w_scharr;
    w_scharr.s3  = ab.x * ab.y;
    w_scharr.s0  = w_scharr.s3 + 1.0f - ab.x - ab.y;
    w_scharr.s12 = ab - (float2)w_scharr.s3;

    // Weight used for Bilinear-Interpolation on Old and New images
    // w.s0 = round(w_scharr.s0 * D0)
    // w.s1 = round(w_scharr.s1 * D0)
    // w.s2 = round(w_scharr.s2 * D0)
    // w.s3 = w.s3 = D0 - w.s0 - w.s1 - w.s2

    float4 w;
    w    = round(w_scharr * (float4)D0);
    w.s3 = D0 - w.s0 - w.s1 - w.s2; // Added for matching VX implementation

    // G.s0 = A11, G.s1 = A12, G.s2 = A22, G.s3 = min_eig
    int4 iG = (int4)0;

    // Window offset
    int window_offset = idx * window_dimension_pow2;

    // Compute Spatial Gradient Matrix G
    for(ushort ky = 0; ky < window_dimension; ++ky)
    {
        int offset_y = iold_keypoint.y + ky;
        for(ushort kx = 0; kx < window_dimension; ++kx)
        {
            int    offset_x = iold_keypoint.x + kx;
            float4 px;

            // Load values from old_image for computing the bilinear interpolation
            px = convert_float4((uchar4)(vload2(0, offset(&old_image, offset_x, offset_y)),
                                         vload2(0, offset(&old_image, offset_x, offset_y + 1))));

            // old_i.s0 = ival, old_i.s1 = ixval, old_i.s2 = iyval, old_i.s3 = dummy
            float4 old_i;

            // Compute bilinear interpolation (with D1 scale factor) for ival
            old_i.s0 = dot(px, w) * D1;

            // Load values from old_scharr_gx for computing the bilinear interpolation
            px = convert_float4((short4)(vload2(0, (__global short *)offset(&old_scharr_gx, offset_x, offset_y)),
                                         vload2(0, (__global short *)offset(&old_scharr_gx, offset_x, offset_y + 1))));

            // Compute bilinear interpolation for ixval
            old_i.s1 = dot(px, w_scharr);

            // Load values from old_scharr_gy for computing the bilinear interpolation
            px = convert_float4((short4)(vload2(0, (__global short *)offset(&old_scharr_gy, offset_x, offset_y)),
                                         vload2(0, (__global short *)offset(&old_scharr_gy, offset_x, offset_y + 1))));

            // Compute bilinear interpolation for iyval
            old_i.s2 = dot(px, w_scharr);

            // Rounding (it could be omitted. Used just for matching the VX implementation)
            int4 iold = convert_int4(round(old_i));

            // Accumulate values in the Spatial Gradient Matrix
            iG.s0 += (int)(iold.s1 * iold.s1);
            iG.s1 += (int)(iold.s1 * iold.s2);
            iG.s2 += (int)(iold.s2 * iold.s2);

            // Store ival, ixval and iyval
            iold_val[window_offset + kx] = convert_short4(iold);
        }
        window_offset += window_dimension;
    }

    // Scale iA11, iA12 and iA22
    float4 G = convert_float4(iG) * (float4)FLT_SCALE;

    // Compute minimum eigen value
    G.s3 = (float)(G.s2 + G.s0 - sqrt(pown(G.s0 - G.s2, 2) + 4.0f * G.s1 * G.s1)) * eig_const;

    // Store A11. A11, A22 and min_eig
    coeff[idx] = G;
}

/** Computes the motion vector for a given keypoint
 *
 * @param[in]      new_image_ptr                           Pointer to the input new image. Supported data types: U8
 * @param[in]      new_image_stride_x                      Stride of the input new image in X dimension (in bytes)
 * @param[in]      new_image_step_x                        new_image_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]      new_image_stride_y                      Stride of the input new image in Y dimension (in bytes)
 * @param[in]      new_image_step_y                        new_image_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]      new_image_offset_first_element_in_bytes The offset of the first element in the input new image
 * @param[in, out] new_points                              An output array of key points. Those key points are defined at the new_images high resolution pyramid
 * @param[in]      coeff                                   The | A11 | A12 | A22 | min_eig | for each keypoint
 * @param[in]      iold_val                                The | ival | ixval | iyval | dummy | for each point in the window centered on old_keypoint
 * @param[in]      window_dimension                        The size of the window on which to perform the algorithm
 * @param[in]      window_dimension_pow2                   The squared size of the window on which to perform the algorithm
 * @param[in]      half_window                             The half size of the window on which to perform the algorithm
 * @param[in]      num_iterations                          The maximum number of iterations
 * @param[in]      epsilon                                 The value for terminating the algorithm.
 * @param[in]      border_limits                           It stores the right border limit (width - window_dimension - 1, height - window_dimension - 1,)
 * @param[in]      eig_const                               1.0f / (float)(2.0f * window_dimension * window_dimension)
 * @param[in]      level0                                  It is set to 1 if level of pyramid = 0
 * @param[in]      term_iteration                          It is set to 1 if termination = VX_TERM_CRITERIA_ITERATIONS
 * @param[in]      term_epsilon                            It is set to 1 if termination = VX_TERM_CRITERIA_EPSILON
 */
void __kernel lktracker_stage1(
    IMAGE_DECLARATION(new_image),
    __global float4 *new_points,
    __global float4 *coeff,
    __global short4 *iold_val,
    const int        window_dimension,
    const int        window_dimension_pow2,
    const int        half_window,
    const int        num_iterations,
    const float      epsilon,
    const float3     border_limits,
    const float      eig_const,
    const int        level0,
    const int        term_iteration,
    const int        term_epsilon)
{
    int   idx       = get_global_id(0);
    Image new_image = CONVERT_TO_IMAGE_STRUCT_NO_STEP(new_image);

    // G.s0 = A11, G.s1 = A12, G.s2 = A22, G.s3 = min_eig
    float4 G = coeff[idx];

    // Determinant
    float D = G.s0 * G.s2 - G.s1 * G.s1;

    // Check if it is a good point to track
    if(G.s3 < EIGENVALUE_THR || D < DETERMINANT_THR)
    {
        if(level0 == 1)
        {
            // Invalidate tracked point as we are at level 0
            new_points[idx].s2 = 0;
        }

        return;
    }

    // Compute inverse
    //D = native_recip(D);
    D = 1.0 / D;

    // Get new keypoint
    float2 new_keypoint = new_points[idx].xy - (float)half_window;

    // Get new point
    float2 out_new_point = new_points[idx].xy;

    // Keep delta obtained in the previous iteration
    float2 prev_delta = (float2)0.0f;

    int j = 0;
    while(j < num_iterations)
    {
        // Get the floor value
        float2 inew_keypoint = floor(new_keypoint);

        // Check if using the window dimension we can go out of boundary in the following for loops. If so, invalidate the tracked point
        if(any(inew_keypoint < border_limits.zz) || any(inew_keypoint >= border_limits.xy))
        {
            if(level0 == 1)
            {
                // Invalidate tracked point as we are at level 0
                new_points[idx].s2 = 0.0f;
            }
            else
            {
                new_points[idx].xy = out_new_point;
            }

            return;
        }

        // Compute weight for the bilinear interpolation
        float2 ab = new_keypoint - inew_keypoint;

        // Weight used for Bilinear-Interpolation on Old and New images
        // w.s0 = round((1.0f - ab.x) * (1.0f - ab.y) * D0)
        // w.s1 = round(ab.x * (1.0f - ab.y) * D0)
        // w.s2 = round((1.0f - ab.x) * ab.y * D0)
        // w.s3 = D0 - w.s0 - w.s1 - w.s2

        float4 w;
        w.s3  = ab.x * ab.y;
        w.s0  = w.s3 + 1.0f - ab.x - ab.y;
        w.s12 = ab - (float2)w.s3;
        w     = round(w * (float4)D0);
        w.s3  = D0 - w.s0 - w.s1 - w.s2;

        // Mismatch vector
        int2 ib = 0;

        // Old val offset
        int old_val_offset = idx * window_dimension_pow2;

        for(int ky = 0; ky < window_dimension; ++ky)
        {
            for(int kx = 0; kx < window_dimension; ++kx)
            {
                // ival, ixval and iyval have been computed in the previous stage
                int4 old_ival = convert_int4(iold_val[old_val_offset]);

                // Load values from old_image for computing the bilinear interpolation
                float4 px = convert_float4((uchar4)(vload2(0, offset(&new_image, inew_keypoint.x + kx, inew_keypoint.y + ky)),
                                                    vload2(0, offset(&new_image, inew_keypoint.x + kx, inew_keypoint.y + ky + 1))));

                // Compute bilinear interpolation on new image
                int jval = (int)round(dot(px, w) * D1);

                // Compute luminance difference
                int diff = (int)(jval - old_ival.s0);

                // Accumulate values in mismatch vector
                ib += (diff * old_ival.s12);

                // Update old val offset
                old_val_offset++;
            }
        }

        float2 b = convert_float2(ib) * (float2)FLT_SCALE;

        // Optical Flow
        float2 delta;

        delta.x = (float)((G.s1 * b.y - G.s2 * b.x) * D);
        delta.y = (float)((G.s1 * b.x - G.s0 * b.y) * D);

        // Update new point coordinate
        new_keypoint += delta;

        out_new_point = new_keypoint + (float2)half_window;

        if(term_epsilon == 1)
        {
            float mag2 = dot(delta, delta);

            if(mag2 <= epsilon)
            {
                new_points[idx].xy = out_new_point;

                return;
            }
        }

        // Check convergence analyzing the previous delta
        if(j > 0 && all(fabs(delta + prev_delta) < (float2)0.01f))
        {
            out_new_point -= delta * (float2)0.5f;

            new_points[idx].xy = out_new_point;

            return;
        }

        // Update previous delta
        prev_delta = delta;

        if(term_iteration == 1)
        {
            j++;
        }
    }

    new_points[idx].xy = out_new_point;
}
