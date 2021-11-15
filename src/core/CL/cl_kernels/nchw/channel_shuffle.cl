/*
* Copyright (c) 2018-2021 Arm Limited.
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
#include "tile_helpers.h"

#if defined(DATA_TYPE) && defined(VEC_SIZE) && defined(NUM_GROUPS) && defined(K) && defined(SRC_DIM_Z)

// Check valid VEC_SIZES
#if VEC_SIZE != 1 && VEC_SIZE != 2 && VEC_SIZE != 3 && VEC_SIZE != 4 && VEC_SIZE != 8 && VEC_SIZE != 16
#error "Only vector sizes 1, 2, 3, 4, 8 and 16 are supported"
#endif // VEC_SIZE != 1 && VEC_SIZE != 2 && VEC_SIZE != 3 && VEC_SIZE != 4 && VEC_SIZE != 8 && VEC_SIZE != 16

#define DIV_MOD_UINT(x, y, div_res, mod_res)                \
    ({                                                      \
        div_res = (uint)((x)/(y)); \
        uint r  = div_res * (y);                            \
        mod_res = (x)-r;                                    \
    })

/** Performs channel shuffle when the data layout is NCHW. See https://arxiv.org/pdf/1707.01083.pdf for details.
 *
 * @note The vector size must be given as a preprocessor argument using -DVEC_SIZE=num. e.g. -DVEC_SIZE=4
 * @note The depth of the tensor must be given as a preprocessor argument using -DSRC_DIM_Z=num. e.g. -DSRC_DIM_Z=64
 * @note The number of groups must be given as a preprocessor argument using -DNUM_GROUPS=num_groups. e.g. -DNUM_GROUPS=2
 * @note The number of channels in each group must be given as a preprocessor argument using -DK=num. e.g. -DK=1
 *       K is equal to num_channels / num_groups.
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: All
 * @param[in]  src_stride_x                      Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the first source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_w                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void channel_shuffle_nchw(TENSOR4D_DECLARATION(src),
                                   TENSOR4D_DECLARATION(dst))
{
    uint curr_channel = 0; // channel id of input
    uint batch_id     = 0; // batch id
    uint group_id     = 0; // group id
    uint channel_id   = 0; // channel id within the group

    // Compute curr_channel and batch_id
    DIV_MOD_UINT(get_global_id(2), SRC_DIM_Z, batch_id, curr_channel);

    // Compute group_id and channel_id
    DIV_MOD_UINT(curr_channel, K, group_id, channel_id);

    const uint x = get_global_id(0) * VEC_SIZE;
    const uint y = get_global_id(1) * 2;
    const uint z = channel_id * NUM_GROUPS + group_id;

    // Load the Nx2 block
    const __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * src_stride_y + curr_channel * src_stride_z + batch_id * src_stride_w;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    u0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_ptr + 0 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    u1 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_ptr + 1 * src_stride_y));

    // Store blocks
    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z + batch_id * dst_stride_w;
    VSTORE(VEC_SIZE)
    (u0, 0, (__global DATA_TYPE *)(output_ptr + 0 * dst_stride_y));
    VSTORE(VEC_SIZE)
    (u1, 0, (__global DATA_TYPE *)(output_ptr + 1 * dst_stride_y));
}

#endif // defined(DATA_TYPE) && defined(VEC_SIZE) && defined(NUM_GROUPS) && defined(K) && defined(SRC_DIM_Z)
