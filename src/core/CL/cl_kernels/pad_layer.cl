/*
 * Copyright (c) 2019-2020 Arm Limited.
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

#if defined(DATA_TYPE) && defined(VEC_SIZE) && defined(PAD_X_BEFORE) && defined(SRC_WIDTH)

#define VEC_TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE)
#define VEC_SELECT SELECT_DATA_TYPE(DATA_TYPE, VEC_SIZE)
#define OFFSETS VEC_OFFS(VEC_SELECT, VEC_SIZE)

#if defined(CONST_VAL)
/** Perform a pad operation when PaddingMode is CONSTANT
 *
 * @note Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @note Vector size must be passed using the -DVEC_SIZE compile flag, e.g. -DVEC_SIZE=4
 * @note Constant value used to fill the pads must be passed using the -DCONST_VAL compile flag, e.g. -DCONST_VAL=1.27
 * @note Pad to add to the left must be passed using the -DPAD_X_BEFORE compile flag, e.g. -DPAD_X_BEFORE=5
 * @note Input tensor's width must be passed using the -DSRC_WIDTH compile flag, e.g. -DSRC_WIDTH=224
 * @note In case pad left is more than the vector size, the number of threads to skip along the X axis must be passed using the
 *       -DNUM_THREADS_TO_SKIP_X compile flag, e.g. -DNUM_THREADS_TO_SKIP_X=1. This is defined as (PAD_X_BEFORE / VEC_SIZE)
 * @note If pad also needs to be added to the top of the tensor, the following compile flags must be passed at compile time:
 *       -# -DPAD_Y_BEFORE: Pad to add to the top of the input tensor (e.g. -DPAD_Y_BEFORE=3)
 *       -# -DSRC_HEIGHT: Input tensor's height (e.g. -DSRC_HEIGHT=127)
 * @note If pad also needs to be added to the depth of the tensor, the following compile flags must be passed at compile time:
 *       -# -DPAD_Z_BEFORE: Pad to add before the first plane of the input tensor (e.g. -DPAD_Z_BEFORE=3)
 *       -# -DSRC_DEPTH: Input tensor's depth (e.g. -DSRC_DEPTH=32)
 * @note If pad also needs to be added to the batch of the tensor, the following compile flags must be passed at compile time:
 *       -# -DPAD_W_BEFORE: Pad to add before the first batch of the input tensor (e.g. -DPAD_W_BEFORE=3)
 *       -# -DSRC_BATCH: Input tensor's batch size (e.g. -DSRC_BATCH=4)
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: All
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
 * @param[in]  batch                             (Optional) Batch index if 4D pad must be applied
 */
__kernel void pad_layer_constant(TENSOR3D_DECLARATION(src),
                                 TENSOR3D_DECLARATION(dst)
#if defined(PAD_W_BEFORE)
                                 ,
                                 uint batch
#endif // defined(PAD_W_BEFORE)
                                )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    uint cond = 0;

#if defined(PAD_W_BEFORE)
    cond |= batch < PAD_W_BEFORE || batch >= (SRC_BATCH + PAD_W_BEFORE);
#endif // defined(PAD_W_BEFORE)
#if defined(PAD_Z_BEFORE)
    cond |= z < PAD_Z_BEFORE || z >= (SRC_DEPTH + PAD_Z_BEFORE);
#endif // defined(PAD_Z_BEFORE)

    if(cond)
    {
        Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);
        VSTORE(VEC_SIZE)
        ((VEC_TYPE)CONST_VAL, 0, (__global DATA_TYPE *)dst.ptr);
    }
    else
    {
        Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
        Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

#if defined(NUM_THREADS_TO_SKIP_X)
        /* In case the pad left is greater than the vector size, and we are past the threads operating solely on pad values,
         * the input pointer must be brought back along the X axis to start from the first non-pad values.
         *
         * E.g. with VEC_SIZE=2, PAD_X_BEFORE=5, CONST_VAL=0 and 1D input |1 2 3 4 5 6|:
         *  -# The first thread will compute the output values |0 0| since it detects (x_outs == (0, 1)) < PAD_X_BEFORE
         *  -# The second thread will compute the output values |0 0| since it detects (x_outs == (2, 3)) < PAD_X_BEFORE
         *  -# The third thread should compute |0 1|, however the input pointer is now ahead of ((x * VEC_SIZE) == 4) values, reading |4 5|
         *  -# To detect this, we use ((PAD_X_BEFORE / VEC_SIZE) == NUM_THREADS_TO_SKIP_X == 2) and check that it is >= to the current x
         *  -# So, we bring the pointer back of NUM_THREADS_TO_SKIP_X threads, which means multiplying this constant by the input's step along the X axis
         *  -# Now that the pointer is back of ((NUM_THREADS_TO_SKIP_X * src_step_x) == 4) values, it will read the desired values |0 1|
         */
        src.ptr -= select(0u, NUM_THREADS_TO_SKIP_X * src_step_x, x >= NUM_THREADS_TO_SKIP_X);
#endif // defined(NUM_THREADS_TO_SKIP_X)
#if defined(PAD_Z_BEFORE)
        src.ptr -= PAD_Z_BEFORE * src_step_z;
#endif // defined(PAD_Z_BEFORE)
#if defined(PAD_W_BEFORE)
        src.ptr -= PAD_W_BEFORE * SRC_DEPTH * src_step_z;
#endif // defined(PAD_W_BEFORE)

        VEC_TYPE src_vals = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src.ptr);

        VEC_INT xs_out = (VEC_INT)(x * VEC_SIZE) + CONVERT(OFFSETS, VEC_INT);
        VEC_INT cond   = xs_out < (VEC_INT)PAD_X_BEFORE || xs_out >= (VEC_INT)(SRC_WIDTH + PAD_X_BEFORE);
#if defined(PAD_Y_BEFORE)
        cond |= (VEC_INT)y < (VEC_INT)PAD_Y_BEFORE || (VEC_INT)y >= (VEC_INT)(SRC_HEIGHT + PAD_Y_BEFORE);
#endif // defined(PAD_Y_BEFORE)
        VSTORE(VEC_SIZE)
        (select(src_vals, (VEC_TYPE)CONST_VAL, CONVERT(cond, VEC_SELECT)), 0, (__global DATA_TYPE *)dst.ptr);
    }
}
#endif // defined(CONST_VAL)

#if defined(PAD_X_BEFORE_REMAINDER) && defined(PAD_X_AFTER_REMAINDER) && defined(PAD_X_BEFORE_REMAINDER_REFL) && defined(PAD_X_AFTER_REMAINDER_REFL) && defined(AFTER_PAD_FACT_X)

#define SCALAR_COND(x) (VEC_SELECT) x == (VEC_SELECT)1
#define ROTATE_REVERSE(x, n) ROTATE(REVERSE(x, VEC_SIZE), VEC_SIZE, n)
#define SYMM_REFL_LEFT(x, n0, n1) select(ROTATE_REVERSE(x, n1), ROTATE(x, VEC_SIZE, n0), OFFSETS >= (VEC_SELECT)n0)
#define SYMM_REFL_RIGHT(x, n0, n1) select(ROTATE(x, VEC_SIZE, n0), ROTATE_REVERSE(x, n1), OFFSETS >= (VEC_SELECT)n0)

/** Perform a pad operation when PaddingMode is SYMMETRIC
 *
 * @note Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @note Vector size must be passed using the -DVEC_SIZE compile flag, e.g. -DVEC_SIZE=4
 * @note Constant value must be passed using the -DCONST_VAL compile flag, e.g. -DCONST_VAL=1.27
 * @note Pad to add to the left must be passed using the -DPAD_X_BEFORE compile flag, e.g. -DPAD_X_BEFORE=5
 * @note Input tensor's width must be passed using the -DSRC_WIDTH compile flag, e.g. -DSRC_WIDTH=224
 * @note Number of values to the left when operating across left padding must be passed using the -DPAD_X_BEFORE_REMAINDER compile flag, e.g. -DPAD_X_BEFORE_REMAINDER=5
 * @note Number of values to the left when operating across right padding must be passed using the -DPAD_X_AFTER_REMAINDER compile flag, e.g. -DPAD_X_AFTER_REMAINDER=6
 * @note To rearrange the vectors properly, (PAD_X_BEFORE_REMAINDER + 1) must be passed when mode is REFLECT using the -DPAD_X_BEFORE_REMAINDER_REFL compile flag, e.g. -DPAD_X_BEFORE_REMAINDER=6
 * @note To rearrange the vectors properly, (PAD_X_AFTER_REMAINDER - 1) must be passed using the -DPAD_X_AFTER_REMAINDER_REFL compile flag, e.g. -DPAD_X_AFTER_REMAINDER=5
 * @note When after pad X, starting point to read backward from must be passed using the -DAFTER_PAD_FACT_X compile flag, e.g. -DAFTER_PAD_FACT_X=253
 * @note If padding mode is REFLECT, the -DIS_REFLECT compile flag must be set to 1, else it must be set to 0
 * @note If pad also needs to be added to the top of the tensor, the following compile flags must be passed at compile time:
 *       -# -DPAD_Y_BEFORE: Pad to add to the top of the input tensor (e.g. -DPAD_Y_BEFORE=3)
 *       -# -DSRC_HEIGHT: Input tensor's height (e.g. -DSRC_HEIGHT=127)
 * @note If pad also needs to be added to the depth of the tensor, the following compile flags must be passed at compile time:
 *       -# -DPAD_Z_BEFORE: Pad to add before the first plane of the input tensor (e.g. -DPAD_Z_BEFORE=3)
 *       -# -DSRC_DEPTH: Input tensor's depth (e.g. -DSRC_DEPTH=32)
 * @note If the starting point to read backward from is less than the output's last element accessed in the X, the following compile flags must be passed at compile time to avoid negative offsets:
 *       -# -DAFTER_PAD_REM: Defines how much to rotate the vector if the backward calculation attempted to read from a negative offset (e.g. -DAFTER_PAD_REM=3)
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: All
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
__kernel void pad_layer_symmetric_reflect(TENSOR3D_DECLARATION(src),
                                          TENSOR3D_DECLARATION(dst))
{
    // Get current thread position
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    // Define conditions based on the thread X position w.r.t. pad left and right
    const int x_out_first         = x * VEC_SIZE;
    const int x_out_last          = x_out_first + VEC_SIZE;
    const int is_before_pad_left  = (x_out_last <= PAD_X_BEFORE);
    const int is_across_pad_left  = (x_out_first < PAD_X_BEFORE) && (x_out_last > PAD_X_BEFORE);
    const int is_inside_input     = (x_out_first >= PAD_X_BEFORE) && (x_out_last <= (SRC_WIDTH + PAD_X_BEFORE));
    const int is_across_pad_right = (x_out_first < (SRC_WIDTH + PAD_X_BEFORE)) && (x_out_last > (SRC_WIDTH + PAD_X_BEFORE));
    const int is_after_pad_right  = (x_out_first >= (SRC_WIDTH + PAD_X_BEFORE));

    // Calculate base pointers
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes;
    Tensor3D        dst      = CONVERT_TO_TENSOR3D_STRUCT(dst);

    // Calculate input tensor's offset based on the defined conditions
    int x_offset = 0;
    x_offset     = select(x_offset, PAD_X_BEFORE - x_out_last + IS_REFLECT, is_before_pad_left);
    x_offset     = select(x_offset, x_out_first - PAD_X_BEFORE, is_inside_input);
    x_offset     = select(x_offset, SRC_WIDTH - VEC_SIZE, is_across_pad_right);
    x_offset     = select(x_offset, AFTER_PAD_FACT_X - x_out_last, is_after_pad_right);

#if defined(AFTER_PAD_REM)
    int neg_offs = x_offset < 0;
    x_offset     = max(x_offset, 0);
#endif // defined(AFTER_PAD_REM)

    // Load input values from the computed offset
    int y_in = y;
    int z_in = z;
#if defined(PAD_Y_BEFORE)
    y_in = select(y - PAD_Y_BEFORE, PAD_Y_BEFORE - y + IS_REFLECT - 1, y < PAD_Y_BEFORE);
    y_in = select(y_in, 2 * SRC_HEIGHT + PAD_Y_BEFORE - y - IS_REFLECT - 1, y >= (SRC_HEIGHT + PAD_Y_BEFORE));
#endif // defined(PAD_Y_BEFORE)
#if defined(PAD_Z_BEFORE)
    z_in = select(z - PAD_Z_BEFORE, PAD_Z_BEFORE - z + IS_REFLECT - 1, z < PAD_Z_BEFORE);
    z_in = select(z_in, 2 * SRC_DEPTH + PAD_Z_BEFORE - z - IS_REFLECT - 1, z >= (SRC_DEPTH + PAD_Z_BEFORE));
#endif // defined(PAD_Y_BEFORE)

    src_addr += x_offset * src_stride_x + y_in * src_step_y + z_in * src_step_z;

#if SRC_WIDTH == 1
    VSTORE(VEC_SIZE)
    ((VEC_TYPE)(*(__global DATA_TYPE *)src_addr), 0, (__global DATA_TYPE *)dst.ptr);
#else // SRC_WIDTH == 1

    VEC_TYPE src_vals = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src_addr);

    // Choose rearrangement policy based on the defined conditions
    src_vals = select(src_vals, SYMM_REFL_LEFT(src_vals, PAD_X_BEFORE_REMAINDER, PAD_X_BEFORE_REMAINDER_REFL), SCALAR_COND(is_across_pad_left));
    src_vals = select(src_vals, SYMM_REFL_RIGHT(src_vals, PAD_X_AFTER_REMAINDER, PAD_X_AFTER_REMAINDER_REFL), SCALAR_COND(is_across_pad_right));
    src_vals = select(src_vals, REVERSE(src_vals, VEC_SIZE), SCALAR_COND((is_before_pad_left || is_after_pad_right)));
#if defined(AFTER_PAD_REM)
    src_vals = select(src_vals, ROTATE(src_vals, VEC_SIZE, AFTER_PAD_REM), SCALAR_COND(neg_offs));
#endif // defined(AFTER_PAD_REM)

    // Store
    VSTORE(VEC_SIZE)
    (src_vals, 0, (__global DATA_TYPE *)dst.ptr);
#endif // SRC_WIDTH == 1
}
#endif // defined(PAD_X_BEFORE_REMAINDER) && defined(PAD_X_AFTER_REMAINDER) && defined(PAD_X_BEFORE_REMAINDER_REFL) && defined(PAD_X_AFTER_REMAINDER_REFL) && defined(AFTER_PAD_FACT_X)
#endif // defined(DATA_TYPE) && defined(VEC_SIZE) && defined(PAD_X_BEFORE) && defined(SRC_WIDTH)
