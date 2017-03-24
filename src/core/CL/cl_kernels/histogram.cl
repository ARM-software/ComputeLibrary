/*
 * Copyright (c) 2016, 2017 ARM Limited.
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

#define VATOMIC_INC16(histogram, win_pos)   \
    {                                       \
        atomic_inc(histogram + win_pos.s0); \
        atomic_inc(histogram + win_pos.s1); \
        atomic_inc(histogram + win_pos.s2); \
        atomic_inc(histogram + win_pos.s3); \
        atomic_inc(histogram + win_pos.s4); \
        atomic_inc(histogram + win_pos.s5); \
        atomic_inc(histogram + win_pos.s6); \
        atomic_inc(histogram + win_pos.s7); \
        atomic_inc(histogram + win_pos.s8); \
        atomic_inc(histogram + win_pos.s9); \
        atomic_inc(histogram + win_pos.sa); \
        atomic_inc(histogram + win_pos.sb); \
        atomic_inc(histogram + win_pos.sc); \
        atomic_inc(histogram + win_pos.sd); \
        atomic_inc(histogram + win_pos.se); \
        atomic_inc(histogram + win_pos.sf); \
    }

/** Calculate the histogram of an 8 bit grayscale image.
 *
 * Each thread will process 16 pixels and use one local atomic operation per pixel.
 * When all work items in a work group are done the resulting local histograms are
 * added to the global histogram using global atomics.
 *
 * @note The input image is represented as a two-dimensional array of type uchar.
 * The output is represented as a one-dimensional uint array of length of num_bins
 *
 * @param[in]  input_ptr                           Pointer to the first source image. Supported data types: U8
 * @param[in]  input_stride_x                      Stride of the first source image in X dimension (in bytes)
 * @param[in]  input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                      Stride of the first source image in Y dimension (in bytes)
 * @param[in]  input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes The offset of the first element in the first source image
 * @param[in]  histogram_local                     The local buffer to hold histogram result in per workgroup. Supported data types: U32
 * @param[out] histogram                           The output buffer to hold histogram final result. Supported data types: U32
 * @param[out] num_bins                            The number of bins
 * @param[out] offset                              The start of values to use (inclusive)
 * @param[out] range                               The range of a bin
 * @param[out] offrange                            The maximum value (exclusive)
 */
__kernel void hist_local_kernel(IMAGE_DECLARATION(input),
                                __local uint *histogram_local,
                                __global uint *restrict histogram,
                                uint                    num_bins,
                                uint                    offset,
                                uint                    range,
                                uint                    offrange)
{
    Image input_buffer = CONVERT_TO_IMAGE_STRUCT(input);
    uint  local_id_x   = get_local_id(0);

    uint local_x_size = get_local_size(0);

    if(num_bins > local_x_size)
    {
        for(int i = local_id_x; i < num_bins; i += local_x_size)
        {
            histogram_local[i] = 0;
        }
    }
    else
    {
        if(local_id_x <= num_bins)
        {
            histogram_local[local_id_x] = 0;
        }
    }

    uint16 vals = convert_uint16(vload16(0, input_buffer.ptr));

    uint16 win_pos = select(num_bins, ((vals - offset) * num_bins) / range, (vals >= offset && vals < offrange));

    barrier(CLK_LOCAL_MEM_FENCE);
    VATOMIC_INC16(histogram_local, win_pos);
    barrier(CLK_LOCAL_MEM_FENCE);

    if(num_bins > local_x_size)
    {
        for(int i = local_id_x; i < num_bins; i += local_x_size)
        {
            atomic_add(histogram + i, histogram_local[i]);
        }
    }
    else
    {
        if(local_id_x <= num_bins)
        {
            atomic_add(histogram + local_id_x, histogram_local[local_id_x]);
        }
    }
}

/** Calculate the histogram of an 8 bit grayscale image's border.
 *
 * Each thread will process one pixel using global atomic.
 * When all work items in a work group are done the resulting local histograms are
 * added to the global histogram using global atomics.
 *
 * @note The input image is represented as a two-dimensional array of type uchar.
 * The output is represented as a one-dimensional uint array of length of num_bins
 *
 * @param[in]  input_ptr                           Pointer to the first source image. Supported data types: U8
 * @param[in]  input_stride_x                      Stride of the first source image in X dimension (in bytes)
 * @param[in]  input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                      Stride of the first source image in Y dimension (in bytes)
 * @param[in]  input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes The offset of the first element in the first source image
 * @param[out] histogram                           The output buffer to hold histogram final result. Supported data types: U32
 * @param[out] num_bins                            The number of bins
 * @param[out] offset                              The start of values to use (inclusive)
 * @param[out] range                               The range of a bin
 * @param[out] offrange                            The maximum value (exclusive)
 */
__kernel void hist_border_kernel(IMAGE_DECLARATION(input),
                                 __global uint *restrict histogram,
                                 uint                    num_bins,
                                 uint                    offset,
                                 uint                    range,
                                 uint                    offrange)
{
    Image input_buffer = CONVERT_TO_IMAGE_STRUCT(input);

    uint val = (uint)(*input_buffer.ptr);

    uint win_pos = (val >= offset) ? (((val - offset) * num_bins) / range) : 0;

    if(val >= offset && (val < offrange))
    {
        atomic_inc(histogram + win_pos);
    }
}

/** Calculate the histogram of an 8 bit grayscale image with bin size of 256 and window size of 1.
 *
 * Each thread will process 16 pixels and use one local atomic operation per pixel.
 * When all work items in a work group are done the resulting local histograms are
 * added to the global histogram using global atomics.
 *
 * @note The input image is represented as a two-dimensional array of type uchar.
 * The output is represented as a one-dimensional uint array of 256 elements
 *
 * @param[in]  input_ptr                           Pointer to the first source image. Supported data types: U8
 * @param[in]  input_stride_x                      Stride of the first source image in X dimension (in bytes)
 * @param[in]  input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                      Stride of the first source image in Y dimension (in bytes)
 * @param[in]  input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes The offset of the first element in the first source image
 * @param[in]  histogram_local                     The local buffer to hold histogram result in per workgroup. Supported data types: U32
 * @param[out] histogram                           The output buffer to hold histogram final result. Supported data types: U32
 */
__kernel void hist_local_kernel_fixed(IMAGE_DECLARATION(input),
                                      __local uint *histogram_local,
                                      __global uint *restrict histogram)
{
    Image input_buffer = CONVERT_TO_IMAGE_STRUCT(input);

    uint local_index  = get_local_id(0);
    uint local_x_size = get_local_size(0);

    for(int i = local_index; i < 256; i += local_x_size)
    {
        histogram_local[i] = 0;
    }

    uint16 vals = convert_uint16(vload16(0, input_buffer.ptr));

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(histogram_local + vals.s0);
    atomic_inc(histogram_local + vals.s1);
    atomic_inc(histogram_local + vals.s2);
    atomic_inc(histogram_local + vals.s3);
    atomic_inc(histogram_local + vals.s4);
    atomic_inc(histogram_local + vals.s5);
    atomic_inc(histogram_local + vals.s6);
    atomic_inc(histogram_local + vals.s7);
    atomic_inc(histogram_local + vals.s8);
    atomic_inc(histogram_local + vals.s9);
    atomic_inc(histogram_local + vals.sa);
    atomic_inc(histogram_local + vals.sb);
    atomic_inc(histogram_local + vals.sc);
    atomic_inc(histogram_local + vals.sd);
    atomic_inc(histogram_local + vals.se);
    atomic_inc(histogram_local + vals.sf);

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = local_index; i < 256; i += local_x_size)
    {
        atomic_add(histogram + i, histogram_local[i]);
    }
}

/** Calculate the histogram of an 8 bit grayscale image with bin size as 256 and window size as 1.
 *
 * Each thread will process one pixel using global atomic.
 * When all work items in a work group are done the resulting local histograms are
 * added to the global histogram using global atomics.
 *
 * @note The input image is represented as a two-dimensional array of type uchar.
 * The output is represented as a one-dimensional uint array of 256
 *
 * @param[in]  input_ptr                           Pointer to the first source image. Supported data types: U8
 * @param[in]  input_stride_x                      Stride of the first source image in X dimension (in bytes)
 * @param[in]  input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                      Stride of the first source image in Y dimension (in bytes)
 * @param[in]  input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes The offset of the first element in the first source image
 * @param[out] histogram                           The output buffer to hold histogram final result. Supported data types: U32
 */
__kernel void hist_border_kernel_fixed(IMAGE_DECLARATION(input),
                                       __global uint *restrict histogram)
{
    Image input_buffer = CONVERT_TO_IMAGE_STRUCT(input);
    atomic_inc(histogram + *input_buffer.ptr);
}
