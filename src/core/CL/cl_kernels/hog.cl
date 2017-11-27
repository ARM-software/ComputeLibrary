/*
 * Copyright (c) 2017, 2018 ARM Limited.
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

#if defined(CELL_WIDTH) && defined(CELL_HEIGHT) && defined(NUM_BINS) && defined(PHASE_SCALE)

/** This OpenCL kernel computes the HOG orientation binning
 *
 * @attention The following variables must be passed at compile time:
 *
 * -# -DCELL_WIDTH = Width of the cell
 * -# -DCELL_HEIGHT = height of the cell
 * -# -DNUM_BINS = Number of bins for each cell
 * -# -DPHASE_SCALE = Scale factor used to evaluate the index of the local HOG
 *
 * @note Each work-item computes a single cell
 *
 * @param[in]  mag_ptr                             Pointer to the source image which stores the magnitude of the gradient for each pixel. Supported data types: S16
 * @param[in]  mag_stride_x                        Stride of the magnitude image in X dimension (in bytes)
 * @param[in]  mag_step_x                          mag_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mag_stride_y                        Stride of the magnitude image in Y dimension (in bytes)
 * @param[in]  mag_step_y                          mag_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  mag_offset_first_element_in_bytes   The offset of the first element in the magnitude image
 * @param[in]  phase_ptr                           Pointer to the source image which stores the phase of the gradient for each pixel. Supported data types: U8
 * @param[in]  phase_stride_x                      Stride of the phase image in X dimension (in bytes)
 * @param[in]  phase_step_x                        phase_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  phase_stride_y                      Stride of the the phase image in Y dimension (in bytes)
 * @param[in]  phase_step_y                        phase_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  phase_offset_first_element_in_bytes The offset of the first element in the the phase image
 * @param[out] dst_ptr                             Pointer to the destination image which stores the local HOG for each cell Supported data types: F32. Number of channels supported: equal to the number of histogram bins per cell
 * @param[in]  dst_stride_x                        Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                          dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                        Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                          dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes   The offset of the first element in the destination image
 */
__kernel void hog_orientation_binning(IMAGE_DECLARATION(mag),
                                      IMAGE_DECLARATION(phase),
                                      IMAGE_DECLARATION(dst))
{
    float bins[NUM_BINS] = { 0 };

    // Compute address for the magnitude and phase images
    Image mag   = CONVERT_TO_IMAGE_STRUCT(mag);
    Image phase = CONVERT_TO_IMAGE_STRUCT(phase);

    __global uchar *mag_row_ptr   = mag.ptr;
    __global uchar *phase_row_ptr = phase.ptr;

    for(int yc = 0; yc < CELL_HEIGHT; ++yc)
    {
        int xc = 0;
        for(; xc <= (CELL_WIDTH - 4); xc += 4)
        {
            // Load magnitude and phase values
            const float4 mag_f32   = convert_float4(vload4(0, (__global short *)mag_row_ptr + xc));
            float4       phase_f32 = convert_float4(vload4(0, phase_row_ptr + xc));

            // Scale phase: phase * scale + 0.5f
            phase_f32 = (float4)0.5f + phase_f32 * (float4)PHASE_SCALE;

            // Compute histogram index.
            int4 hidx_s32 = convert_int4(phase_f32);

            // Compute magnitude weights (w0 and w1)
            const float4 hidx_f32 = convert_float4(hidx_s32);

            // w1 = phase_f32 - hidx_s32
            const float4 w1_f32 = phase_f32 - hidx_f32;

            // w0 = 1.0 - w1
            const float4 w0_f32 = (float4)1.0f - w1_f32;

            // Calculate the weights for splitting vote
            const float4 mag_w0_f32 = mag_f32 * w0_f32;
            const float4 mag_w1_f32 = mag_f32 * w1_f32;

            // Weighted vote between 2 bins

            // Check if the histogram index is equal to NUM_BINS. If so, replace the index with 0
            hidx_s32 = select(hidx_s32, (int4)0, hidx_s32 == (int4)(NUM_BINS));

            // Bin 0
            bins[hidx_s32.s0] += mag_w0_f32.s0;
            bins[hidx_s32.s1] += mag_w0_f32.s1;
            bins[hidx_s32.s2] += mag_w0_f32.s2;
            bins[hidx_s32.s3] += mag_w0_f32.s3;

            hidx_s32 += (int4)1;

            // Check if the histogram index is equal to NUM_BINS. If so, replace the index with 0
            hidx_s32 = select(hidx_s32, (int4)0, hidx_s32 == (int4)(NUM_BINS));

            // Bin1
            bins[hidx_s32.s0] += mag_w1_f32.s0;
            bins[hidx_s32.s1] += mag_w1_f32.s1;
            bins[hidx_s32.s2] += mag_w1_f32.s2;
            bins[hidx_s32.s3] += mag_w1_f32.s3;
        }

        // Left over computation
        for(; xc < CELL_WIDTH; xc++)
        {
            const float mag_value   = *((__global short *)mag_row_ptr + xc);
            const float phase_value = *(phase_row_ptr + xc) * (float)PHASE_SCALE + 0.5f;
            const float w1          = phase_value - floor(phase_value);

            // The quantised phase is the histogram index [0, NUM_BINS - 1]
            // Check limit of histogram index. If hidx == NUM_BINS, hidx = 0
            const uint hidx = (uint)(phase_value) % NUM_BINS;

            // Weighted vote between 2 bins
            bins[hidx] += mag_value * (1.0f - w1);
            bins[(hidx + 1) % NUM_BINS] += mag_value * w1;
        }

        // Point to the next row of magnitude and phase images
        mag_row_ptr += mag_stride_y;
        phase_row_ptr += phase_stride_y;
    }

    // Compute address for the destination image
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Store the local HOG in the global memory
    int xc = 0;
    for(; xc <= (NUM_BINS - 4); xc += 4)
    {
        float4 values = vload4(0, bins + xc);

        vstore4(values, 0, ((__global float *)dst.ptr) + xc);
    }

    // Left over stores
    for(; xc < NUM_BINS; ++xc)
    {
        ((__global float *)dst.ptr)[xc] = bins[xc];
    }
}
#endif /* CELL_WIDTH and CELL_HEIGHT and NUM_BINS and PHASE_SCALE */

#if defined(NUM_CELLS_PER_BLOCK_HEIGHT) && defined(NUM_BINS_PER_BLOCK_X) && defined(NUM_BINS_PER_BLOCK) && defined(HOG_NORM_TYPE) && defined(L2_HYST_THRESHOLD)

#ifndef L2_NORM
#error The value of enum class HOGNormType::L2_NORM has not be passed to the OpenCL kernel
#endif /* not L2_NORM */

#ifndef L2HYS_NORM
#error The value of enum class HOGNormType::L2HYS_NORM has not be passed to the OpenCL kernel
#endif /* not L2HYS_NORM */

#ifndef L1_NORM
#error The value of enum class HOGNormType::L1_NORM has not be passed to the OpenCL kernel
#endif /* not L1_NORM */

/** This OpenCL kernel computes the HOG block normalization
 *
 * @attention The following variables must be passed at compile time:
 *
 * -# -DNUM_CELLS_PER_BLOCK_HEIGHT = Number of cells for each block
 * -# -DNUM_BINS_PER_BLOCK_X = Number of bins for each block along the X direction
 * -# -DNUM_BINS_PER_BLOCK = Number of bins for each block
 * -# -DHOG_NORM_TYPE = Normalization type
 * -# -DL2_HYST_THRESHOLD = Threshold used for L2HYS_NORM normalization method
 * -# -DL2_NORM = Value of the enum class HOGNormType::L2_NORM
 * -# -DL2HYS_NORM = Value of the enum class HOGNormType::L2HYS_NORM
 * -# -DL1_NORM = Value of the enum class HOGNormType::L1_NORM
 *
 * @note Each work-item computes a single block
 *
 * @param[in]  src_ptr                           Pointer to the source image which stores the local HOG. Supported data types: F32. Number of channels supported: equal to the number of histogram bins per cell
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image which stores the normlized HOG Supported data types: F32. Number of channels supported: equal to the number of histogram bins per block
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void hog_block_normalization(IMAGE_DECLARATION(src),
                                      IMAGE_DECLARATION(dst))
{
    float  sum     = 0.0f;
    float4 sum_f32 = (float4)(0.0f);

    // Compute address for the source and destination tensor
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    for(size_t yc = 0; yc < NUM_CELLS_PER_BLOCK_HEIGHT; ++yc)
    {
        const __global float *hist_ptr = (__global float *)(src.ptr + yc * src_stride_y);

        int xc = 0;
        for(; xc <= (NUM_BINS_PER_BLOCK_X - 16); xc += 16)
        {
            const float4 val0 = vload4(0, hist_ptr + xc + 0);
            const float4 val1 = vload4(0, hist_ptr + xc + 4);
            const float4 val2 = vload4(0, hist_ptr + xc + 8);
            const float4 val3 = vload4(0, hist_ptr + xc + 12);

#if(HOG_NORM_TYPE == L2_NORM) || (HOG_NORM_TYPE == L2HYS_NORM)
            // Compute val^2 for L2_NORM or L2HYS_NORM
            sum_f32 += val0 * val0;
            sum_f32 += val1 * val1;
            sum_f32 += val2 * val2;
            sum_f32 += val3 * val3;
#else  /* (HOG_NORM_TYPE == L2_NORM) || (HOG_NORM_TYPE == L2HYS_NORM) */
            // Compute |val| for L1_NORM
            sum_f32 += fabs(val0);
            sum_f32 += fabs(val1);
            sum_f32 += fabs(val2);
            sum_f32 += fabs(val3);
#endif /* (HOG_NORM_TYPE == L2_NORM) || (HOG_NORM_TYPE == L2HYS_NORM) */

            // Store linearly the input values un-normalized in the output image. These values will be reused for the normalization.
            // This approach will help us to be cache friendly in the next for loop where the normalization will be done because all the values
            // will be accessed consecutively
            vstore4(val0, 0, ((__global float *)dst.ptr) + xc + 0 + yc * NUM_BINS_PER_BLOCK_X);
            vstore4(val1, 0, ((__global float *)dst.ptr) + xc + 4 + yc * NUM_BINS_PER_BLOCK_X);
            vstore4(val2, 0, ((__global float *)dst.ptr) + xc + 8 + yc * NUM_BINS_PER_BLOCK_X);
            vstore4(val3, 0, ((__global float *)dst.ptr) + xc + 12 + yc * NUM_BINS_PER_BLOCK_X);
        }

        // Compute left over
        for(; xc < NUM_BINS_PER_BLOCK_X; ++xc)
        {
            const float val = hist_ptr[xc];

#if(HOG_NORM_TYPE == L2_NORM) || (HOG_NORM_TYPE == L2HYS_NORM)
            sum += val * val;
#else  /* (HOG_NORM_TYPE == L2_NORM) || (HOG_NORM_TYPE == L2HYS_NORM) */
            sum += fabs(val);
#endif /* (HOG_NORM_TYPE == L2_NORM) || (HOG_NORM_TYPE == L2HYS_NORM) */

            ((__global float *)dst.ptr)[xc + 0 + yc * NUM_BINS_PER_BLOCK_X] = val;
        }
    }

    sum += dot(sum_f32, (float4)1.0f);

    float scale = 1.0f / (sqrt(sum) + NUM_BINS_PER_BLOCK * 0.1f);

#if(HOG_NORM_TYPE == L2HYS_NORM)
    // Reset sum
    sum_f32 = (float4)0.0f;
    sum     = 0.0f;

    int k = 0;
    for(; k <= NUM_BINS_PER_BLOCK - 16; k += 16)
    {
        float4 val0 = vload4(0, ((__global float *)dst.ptr) + k + 0);
        float4 val1 = vload4(0, ((__global float *)dst.ptr) + k + 4);
        float4 val2 = vload4(0, ((__global float *)dst.ptr) + k + 8);
        float4 val3 = vload4(0, ((__global float *)dst.ptr) + k + 12);

        // Scale val
        val0 = val0 * (float4)scale;
        val1 = val1 * (float4)scale;
        val2 = val2 * (float4)scale;
        val3 = val3 * (float4)scale;

        // Clip val if over _threshold_l2hys
        val0 = fmin(val0, (float4)L2_HYST_THRESHOLD);
        val1 = fmin(val1, (float4)L2_HYST_THRESHOLD);
        val2 = fmin(val2, (float4)L2_HYST_THRESHOLD);
        val3 = fmin(val3, (float4)L2_HYST_THRESHOLD);

        // Compute val^2
        sum_f32 += val0 * val0;
        sum_f32 += val1 * val1;
        sum_f32 += val2 * val2;
        sum_f32 += val3 * val3;

        vstore4(val0, 0, ((__global float *)dst.ptr) + k + 0);
        vstore4(val1, 0, ((__global float *)dst.ptr) + k + 4);
        vstore4(val2, 0, ((__global float *)dst.ptr) + k + 8);
        vstore4(val3, 0, ((__global float *)dst.ptr) + k + 12);
    }

    // Compute left over
    for(; k < NUM_BINS_PER_BLOCK; ++k)
    {
        float val = ((__global float *)dst.ptr)[k] * scale;

        // Clip scaled input_value if over L2_HYST_THRESHOLD
        val = fmin(val, (float)L2_HYST_THRESHOLD);

        sum += val * val;

        ((__global float *)dst.ptr)[k] = val;
    }

    sum += dot(sum_f32, (float4)1.0f);

    // We use the same constants of OpenCV
    scale = 1.0f / (sqrt(sum) + 1e-3f);

#endif /* (HOG_NORM_TYPE == L2HYS_NORM) */

    int i = 0;
    for(; i <= (NUM_BINS_PER_BLOCK - 16); i += 16)
    {
        float4 val0 = vload4(0, ((__global float *)dst.ptr) + i + 0);
        float4 val1 = vload4(0, ((__global float *)dst.ptr) + i + 4);
        float4 val2 = vload4(0, ((__global float *)dst.ptr) + i + 8);
        float4 val3 = vload4(0, ((__global float *)dst.ptr) + i + 12);

        // Multiply val by the normalization scale factor
        val0 = val0 * (float4)scale;
        val1 = val1 * (float4)scale;
        val2 = val2 * (float4)scale;
        val3 = val3 * (float4)scale;

        vstore4(val0, 0, ((__global float *)dst.ptr) + i + 0);
        vstore4(val1, 0, ((__global float *)dst.ptr) + i + 4);
        vstore4(val2, 0, ((__global float *)dst.ptr) + i + 8);
        vstore4(val3, 0, ((__global float *)dst.ptr) + i + 12);
    }

    for(; i < NUM_BINS_PER_BLOCK; ++i)
    {
        ((__global float *)dst.ptr)[i] *= scale;
    }
}
#endif /* NUM_CELLS_PER_BLOCK_HEIGHT and NUM_BINS_PER_BLOCK_X and NUM_BINS_PER_BLOCK and HOG_NORM_TYPE and L2_HYST_THRESHOLD */

#if defined(NUM_BLOCKS_PER_DESCRIPTOR_Y) && defined(NUM_BINS_PER_DESCRIPTOR_X) && defined(THRESHOLD) && defined(MAX_NUM_DETECTION_WINDOWS) && defined(IDX_CLASS) && defined(BLOCK_STRIDE_WIDTH) && defined(BLOCK_STRIDE_HEIGHT) && defined(DETECTION_WINDOW_WIDTH) && defined(DETECTION_WINDOW_HEIGHT)

/** This OpenCL kernel computes the HOG detector using linear SVM
 *
 * @attention The following variables must be passed at compile time:
 *
 * -# -DNUM_BLOCKS_PER_DESCRIPTOR_Y = Number of blocks per descriptor along the Y direction
 * -# -DNUM_BINS_PER_DESCRIPTOR_X = Number of bins per descriptor along the X direction
 * -# -DTHRESHOLD = Threshold for the distance between features and SVM classifying plane
 * -# -DMAX_NUM_DETECTION_WINDOWS = Maximum number of possible detection windows. It is equal to the size of the DetectioWindow array
 * -# -DIDX_CLASS = Index of the class to detect
 * -# -DBLOCK_STRIDE_WIDTH = Block stride for the X direction
 * -# -DBLOCK_STRIDE_HEIGHT = Block stride for the Y direction
 * -# -DDETECTION_WINDOW_WIDTH = Width of the detection window
 * -# -DDETECTION_WINDOW_HEIGHT = Height of the detection window
 *
 * @note Each work-item computes a single detection window
 *
 * @param[in]  src_ptr                           Pointer to the source image which stores the local HOG. Supported data types: F32. Number of channels supported: equal to the number of histogram bins per cell
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  hog_descriptor                    Pointer to HOG descriptor. Supported data types: F32
 * @param[out] dst                               Pointer to DetectionWindow array
 * @param[out] num_detection_windows             Number of objects detected
 */
__kernel void hog_detector(IMAGE_DECLARATION(src),
                           __global float *hog_descriptor,
                           __global DetectionWindow *dst,
                           __global uint *num_detection_windows)
{
    // Check if the DetectionWindow array is full
    if(*num_detection_windows >= MAX_NUM_DETECTION_WINDOWS)
    {
        return;
    }

    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    const int src_step_y_f32 = src_stride_y / sizeof(float);

    // Init score_f32 with 0
    float4 score_f32 = (float4)0.0f;

    // Init score with 0
    float score = 0.0f;

    __global float *src_row_ptr = (__global float *)src.ptr;

    // Compute Linear SVM
    for(int yb = 0; yb < NUM_BLOCKS_PER_DESCRIPTOR_Y; ++yb, src_row_ptr += src_step_y_f32)
    {
        int xb = 0;

        const int offset_y = yb * NUM_BINS_PER_DESCRIPTOR_X;

        for(; xb < (int)NUM_BINS_PER_DESCRIPTOR_X - 8; xb += 8)
        {
            // Load descriptor values
            float4 a0_f32 = vload4(0, src_row_ptr + xb + 0);
            float4 a1_f32 = vload4(0, src_row_ptr + xb + 4);

            float4 b0_f32 = vload4(0, hog_descriptor + xb + 0 + offset_y);
            float4 b1_f32 = vload4(0, hog_descriptor + xb + 4 + offset_y);

            // Multiply accumulate
            score_f32 += a0_f32 * b0_f32;
            score_f32 += a1_f32 * b1_f32;
        }

        for(; xb < NUM_BINS_PER_DESCRIPTOR_X; ++xb)
        {
            const float a = src_row_ptr[xb];
            const float b = hog_descriptor[xb + offset_y];

            score += a * b;
        }
    }

    score += dot(score_f32, (float4)1.0f);

    // Add the bias. The bias is located at the position (descriptor_size() - 1)
    // (descriptor_size - 1) = NUM_BINS_PER_DESCRIPTOR_X * NUM_BLOCKS_PER_DESCRIPTOR_Y
    score += hog_descriptor[NUM_BINS_PER_DESCRIPTOR_X * NUM_BLOCKS_PER_DESCRIPTOR_Y];

    if(score > (float)THRESHOLD)
    {
        int id = atomic_inc(num_detection_windows);
        if(id < MAX_NUM_DETECTION_WINDOWS)
        {
            dst[id].x         = get_global_id(0) * BLOCK_STRIDE_WIDTH;
            dst[id].y         = get_global_id(1) * BLOCK_STRIDE_HEIGHT;
            dst[id].width     = DETECTION_WINDOW_WIDTH;
            dst[id].height    = DETECTION_WINDOW_HEIGHT;
            dst[id].idx_class = IDX_CLASS;
            dst[id].score     = score;
        }
    }
}
#endif /* NUM_BLOCKS_PER_DESCRIPTOR_Y && NUM_BINS_PER_DESCRIPTOR_X && THRESHOLD && MAX_NUM_DETECTION_WINDOWS && IDX_CLASS &&
        * BLOCK_STRIDE_WIDTH && BLOCK_STRIDE_HEIGHT && DETECTION_WINDOW_WIDTH && DETECTION_WINDOW_HEIGHT */
