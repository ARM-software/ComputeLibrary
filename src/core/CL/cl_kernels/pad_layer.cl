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

#if defined(DATA_TYPE) && defined(VEC_SIZE) && defined(CONST_VAL)

#define VEC_TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE)
#define CONVERT_SELECT(x) CONVERT(x, VEC_DATA_TYPE(SELECT_DT, VEC_SIZE))

#if VEC_SIZE == 1
#define OFFSETS (int)0
#elif VEC_SIZE == 2
#define OFFSETS (int2)(0, 1)
#elif VEC_SIZE == 4
#define OFFSETS (int4)(0, 1, 2, 3)
#elif VEC_SIZE == 8
#define OFFSETS (int8)(0, 1, 2, 3, 4, 5, 6, 7)
#elif VEC_SIZE == 16
#define OFFSETS (int16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
#else // VEC_SIZE
#error "Only 1, 2, 3, 4, 8 and 16 vector sizes allowed"
#endif // VEC_SIZE

/** Perform a pad operation
 *
 * @note Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @note Vector size must be passed using the -DVEC_SIZE compile flag, e.g. -DVEC_SIZE=4
 * @note Constant value must be passed using the -DCONST_VAL compile flag, e.g. -DCONST_VAL=1.27
 * @note Pad to add to the left must be passed using the -DPAD_LEFT compile flag, e.g. -DPAD_LEFT=5
 * @note Input tensor's width must be passed using the -DSRC_WIDTH compile flag, e.g. -DSRC_WIDTH=224
 * @note Data type to use for the select instruction must be passed using the -DSELECT_DT compile flag, e.g. -DSELECT_DT=float
 * @note In case pad left is more than the vector size, the number of threads to skil alond the X axis must be passed using the
 *       -DTHREADS_TO_SKIP_X compile flag, e.g. -DTHREADS_TO_SKIP_X=1. This is defined as (PAD_LEFT / VEC_SIZE)
 * @note In pad also needs to be added to the top of the tensor, the following compile flags must be passed at compile time:
 *       -# -DPAD_TOP: Pad to add to the top of the input tensor (e.g. -DPAD_TOP=3)
 *       -# -DSRC_HEIGHT: Input tensor's height (e.g. -DSRC_HEIGHT=127)
 * @note In pad also needs to be added to the depth of the tensor, the following compile flags must be passed at compile time:
 *       -# -DPAD_NEAR: Pad to add before the first plane of the input tensor (e.g. -DPAD_NEAR=3)
 *       -# -DSRC_DEPTH: Input tensor's depth (e.g. -DSRC_DEPTH=32)
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8, S8, QASYMM8, U16, S16, U32, S32, F16, F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source image in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination image in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void pad_layer(TENSOR3D_DECLARATION(src),
                        TENSOR3D_DECLARATION(dst))
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

#if defined(PAD_NEAR)
    if(z < PAD_NEAR || z >= (PAD_NEAR + SRC_DEPTH))
    {
        Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);
        VSTORE(VEC_SIZE)
        ((VEC_TYPE)CONST_VAL, 0, (__global DATA_TYPE *)dst.ptr);
    }
    else
    {
#endif // defined(PAD_NEAR)

        Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
        Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

#if defined(THREADS_TO_SKIP_X)
        /* In case the pad left is greater than the vector size, and we are past the threads operating solely on pad values,
         * the input pointer must be brought back along the X axis to start from the first non-pad values.
         *
         * E.g. with VEC_SIZE=2, PAD_LEFT=5, CONST_VAL=0 and 1D input |1 2 3 4 5 6|:
         *  -# The first thread will compute the output values |0 0| since it detects (x_outs == (0, 1)) < PAD_LEFT
         *  -# The second thread will compute the output values |0 0| since it detects (x_outs == (2, 3)) < PAD_LEFT
         *  -# The third thread should compute |0 1|, however the input pointer is now ahead of ((x * VEC_SIZE) == 4) values, reading |4 5|
         *  -# To detect this, we use ((PAD_LEFT / VEC_SIZE) == THREADS_TO_SKIP_X == 2) and check that it is >= to the current x
         *  -# So, we bring the pointer back of THREADS_TO_SKIP_X threads, which means multiplying this constant by the input's step along the X axis
         *  -# Now that the pointer is back of ((THREADS_TO_SKIP_X * src_step_x) == 4) values, it will read the desired values |0 1|
         */
        src.ptr -= select(0u, THREADS_TO_SKIP_X * src_step_x, x >= THREADS_TO_SKIP_X);
#endif // defined(THREADS_TO_SKIP_X)
#if defined(PAD_NEAR)
        src.ptr -= PAD_NEAR * src_step_z;
#endif // defined(PAD_NEAR)

        VEC_TYPE src_vals = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src.ptr);

        VEC_INT xs_out = (VEC_INT)(x * VEC_SIZE) + OFFSETS;
        VEC_INT cond   = xs_out < (VEC_INT)PAD_LEFT || xs_out >= (VEC_INT)(PAD_LEFT + SRC_WIDTH);
#if defined(PAD_TOP)
        cond |= (VEC_INT)y < (VEC_INT)PAD_TOP || (VEC_INT)y >= (VEC_INT)(PAD_TOP + SRC_HEIGHT);
#endif // defined(PAD_TOP)
        VSTORE(VEC_SIZE)
        (select(src_vals, (VEC_TYPE)CONST_VAL, CONVERT_SELECT(cond)), 0, (__global DATA_TYPE *)dst.ptr);
#if defined(PAD_NEAR)
    }
#endif // defined(PAD_NEAR)
}
#endif // defined(DATA_TYPE) && defined(VEC_SIZE) && defined(CONST_VAL)
