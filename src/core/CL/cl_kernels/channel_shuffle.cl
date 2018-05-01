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

#if defined(DATA_TYPE) && defined(BLOCK_SIZE) && defined(NUM_GROUPS) && defined(K)

// Check valid BLOCK_SIZES
#if BLOCK_SIZE != 4 && BLOCK_SIZE != 8 && BLOCK_SIZE != 16
#error "Only block sizes 4, 8 and 16 are supported"
#endif /* BLOCK_SIZE != 4 && BLOCK_SIZE != 8 && BLOCK_SIZE != 16 */

#define TYPE VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)

/** Perfoms channel shuffle see https://arxiv.org/pdf/1707.01083.pdf for details.
 *
 * @note The number of groups should be given as a preprocessor argument using -DNUM_GROUPS=num_groups. e.g. -DNUM_GROUPS=2
 * @note The number of channels in each group should be given as a preprocessor argument using -DK=num. e.g. -DK=1
 *       K is equal to num_channels / num_groups.
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U8/S8/QS8/QASYMM8/U16/S16/QS16/F16/U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the first source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void channel_shuffle_nchw(TENSOR3D_DECLARATION(src),
                                   TENSOR3D_DECLARATION(dst))
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(dst);

    const uint curr_channel = get_global_id(2);          // channel id of input
    const uint group_id     = curr_channel / NUM_GROUPS; // group id
    const uint channel_id   = curr_channel % NUM_GROUPS; // channel id within the group

    const uint x = get_global_id(0) * BLOCK_SIZE;
    const uint y = get_global_id(1) * BLOCK_SIZE;
    const uint z = channel_id * K + group_id;

    // Load the NxN block
    TYPE u0 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 0, 0));
    TYPE u1 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 1, 0));
    TYPE u2 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 2, 0));
    TYPE u3 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 3, 0));
#if BLOCK_SIZE > 4
    TYPE u4 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 4, 0));
    TYPE u5 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 5, 0));
    TYPE u6 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 6, 0));
    TYPE u7 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 7, 0));
#if BLOCK_SIZE == 16
    TYPE u8  = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 8, 0));
    TYPE u9  = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 9, 0));
    TYPE u10 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 10, 0));
    TYPE u11 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 11, 0));
    TYPE u12 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 12, 0));
    TYPE u13 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 13, 0));
    TYPE u14 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 14, 0));
    TYPE u15 = VLOAD(BLOCK_SIZE)(0, (__global DATA_TYPE *)tensor3D_offset(&src, 0, 15, 0));
#endif /* BLOCK_SIZE == 16 */
#endif /* BLOCK_SIZE > 4 */

    // Store blocks
    VSTORE(BLOCK_SIZE)
    (u0, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 0, z));
    VSTORE(BLOCK_SIZE)
    (u1, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 1, z));
    VSTORE(BLOCK_SIZE)
    (u2, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 2, z));
    VSTORE(BLOCK_SIZE)
    (u3, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 3, z));
#if BLOCK_SIZE > 4
    VSTORE(BLOCK_SIZE)
    (u4, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 4, z));
    VSTORE(BLOCK_SIZE)
    (u5, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 5, z));
    VSTORE(BLOCK_SIZE)
    (u6, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 6, z));
    VSTORE(BLOCK_SIZE)
    (u7, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 7, z));
#if BLOCK_SIZE == 16
    VSTORE(BLOCK_SIZE)
    (u8, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 8, z));
    VSTORE(BLOCK_SIZE)
    (u9, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 9, z));
    VSTORE(BLOCK_SIZE)
    (u10, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 10, z));
    VSTORE(BLOCK_SIZE)
    (u11, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 11, z));
    VSTORE(BLOCK_SIZE)
    (u12, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 12, z));
    VSTORE(BLOCK_SIZE)
    (u13, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 13, z));
    VSTORE(BLOCK_SIZE)
    (u14, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 14, z));
    VSTORE(BLOCK_SIZE)
    (u15, 0, (__global DATA_TYPE *)tensor3D_offset(&dst, x, y + 15, z));
#endif /* BLOCK_SIZE == 16 */
#endif /* BLOCK_SIZE > 4 */
}
#endif /* defined(DATA_TYPE) && defined(BLOCK_SIZE) && defined(NUM_GROUPS) && defined(K) */
