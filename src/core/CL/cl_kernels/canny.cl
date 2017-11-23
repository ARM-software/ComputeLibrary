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

/** Calculate the magnitude and phase from horizontal and vertical result of sobel result.
 *
 * @note The calculation of gradient uses level 1 normalisation.
 * @attention The input and output data types need to be passed at compile time using -DDATA_TYPE_IN and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN=uchar -DDATA_TYPE_OUT=short
 *
 * @param[in]  src1_ptr                            Pointer to the source image (Vertical result of Sobel). Supported data types: S16, S32
 * @param[in]  src1_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  src1_step_x                         src1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  src1_step_y                         src1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  src2_ptr                            Pointer to the source image (Vertical result of Sobel). Supported data types: S16, S32
 * @param[in]  src2_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  src2_step_x                         src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  src2_step_y                         src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] grad_ptr                            Pointer to the gradient output. Supported data types: U16, U32
 * @param[in]  grad_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  grad_step_x                         grad_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  grad_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  grad_step_y                         grad_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  grad_offset_first_element_in_bytes  The offset of the first element of the output
 * @param[out] angle_ptr                           Pointer to the angle output. Supported data types: U8
 * @param[in]  angle_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  angle_step_x                        angle_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  angle_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  angle_step_y                        angle_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  angle_offset_first_element_in_bytes The offset of the first element of the output
 */
__kernel void combine_gradients_L1(
    IMAGE_DECLARATION(src1),
    IMAGE_DECLARATION(src2),
    IMAGE_DECLARATION(grad),
    IMAGE_DECLARATION(angle))
{
    // Construct images
    Image src1  = CONVERT_TO_IMAGE_STRUCT(src1);
    Image src2  = CONVERT_TO_IMAGE_STRUCT(src2);
    Image grad  = CONVERT_TO_IMAGE_STRUCT(grad);
    Image angle = CONVERT_TO_IMAGE_STRUCT(angle);

    // Load sobel horizontal and vertical values
    VEC_DATA_TYPE(DATA_TYPE_IN, 4)
    h = vload4(0, (__global DATA_TYPE_IN *)src1.ptr);
    VEC_DATA_TYPE(DATA_TYPE_IN, 4)
    v = vload4(0, (__global DATA_TYPE_IN *)src2.ptr);

    /* Calculate the gradient, using level 1 normalisation method */
    VEC_DATA_TYPE(DATA_TYPE_OUT, 4)
    m = CONVERT_SAT((abs(h) + abs(v)), VEC_DATA_TYPE(DATA_TYPE_OUT, 4));

    /* Calculate the angle */
    float4 p = atan2pi(convert_float4(v), convert_float4(h));

    /* Remap angle to range [0, 256) */
    p = select(p, p + 2, p < 0.0f) * 128.0f;

    /* Store results */
    vstore4(m, 0, (__global DATA_TYPE_OUT *)grad.ptr);
    vstore4(convert_uchar4_sat_rte(p), 0, angle.ptr);
}

/** Calculate the gradient and angle from horizontal and vertical result of sobel result.
 *
 * @note The calculation of gradient uses level 2 normalisation
 * @attention The input and output data types need to be passed at compile time using -DDATA_TYPE_IN and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN=uchar -DDATA_TYPE_OUT=short
 *
 * @param[in]  src1_ptr                            Pointer to the source image (Vertical result of Sobel). Supported data types: S16, S32
 * @param[in]  src1_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  src1_step_x                         src1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  src1_step_y                         src1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  src2_ptr                            Pointer to the source image (Vertical result of Sobel). Supported data types: S16, S32
 * @param[in]  src2_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  src2_step_x                         src2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  src2_step_y                         src2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] grad_ptr                            Pointer to the gradient output. Supported data types: U16, U32
 * @param[in]  grad_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  grad_step_x                         grad_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  grad_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  grad_step_y                         grad_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  grad_offset_first_element_in_bytes  The offset of the first element of the output
 * @param[out] angle_ptr                           Pointer to the angle output. Supported data types: U8
 * @param[in]  angle_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  angle_step_x                        angle_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  angle_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  angle_step_y                        angle_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  angle_offset_first_element_in_bytes The offset of the first element of the output
 */
__kernel void combine_gradients_L2(
    IMAGE_DECLARATION(src1),
    IMAGE_DECLARATION(src2),
    IMAGE_DECLARATION(grad),
    IMAGE_DECLARATION(angle))
{
    // Construct images
    Image src1  = CONVERT_TO_IMAGE_STRUCT(src1);
    Image src2  = CONVERT_TO_IMAGE_STRUCT(src2);
    Image grad  = CONVERT_TO_IMAGE_STRUCT(grad);
    Image angle = CONVERT_TO_IMAGE_STRUCT(angle);

    // Load sobel horizontal and vertical values
    float4 h = convert_float4(vload4(0, (__global DATA_TYPE_IN *)src1.ptr));
    float4 v = convert_float4(vload4(0, (__global DATA_TYPE_IN *)src2.ptr));

    /* Calculate the gradient, using level 2 normalisation method */
    float4 m = sqrt(h * h + v * v);

    /* Calculate the angle */
    float4 p = atan2pi(v, h);

    /* Remap angle to range [0, 256) */
    p = select(p, p + 2, p < 0.0f) * 128.0f;

    /* Store results */
    vstore4(CONVERT_SAT_ROUND(m, VEC_DATA_TYPE(DATA_TYPE_OUT, 4), rte), 0, (__global DATA_TYPE_OUT *)grad.ptr);
    vstore4(convert_uchar4_sat_rte(p), 0, angle.ptr);
}

/** Array that holds the relative coordinates offset for the neighbouring pixels.
 */
__constant short4 neighbours_coords[] =
{
    { -1, 0, 1, 0 },  // 0
    { -1, 1, 1, -1 }, // 45
    { 0, 1, 0, -1 },  // 90
    { 1, 1, -1, -1 }, // 135
    { 1, 0, -1, 0 },  // 180
    { 1, -1, -1, 1 }, // 225
    { 0, 1, 0, -1 },  // 270
    { -1, -1, 1, 1 }, // 315
    { -1, 0, 1, 0 },  // 360
};

/** Perform non maximum suppression.
 *
 * @attention The input and output data types need to be passed at compile time using -DDATA_TYPE_IN and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN=uchar -DDATA_TYPE_OUT=short
 *
 * @param[in]  grad_ptr                              Pointer to the gradient output. Supported data types: S16, S32
 * @param[in]  grad_stride_x                         Stride of the source image in X dimension (in bytes)
 * @param[in]  grad_step_x                           grad_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  grad_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  grad_step_y                           grad_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  grad_offset_first_element_in_bytes    The offset of the first element of the output
 * @param[in]  angle_ptr                             Pointer to the angle output. Supported data types: U8
 * @param[in]  angle_stride_x                        Stride of the source image in X dimension (in bytes)
 * @param[in]  angle_step_x                          angle_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  angle_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  angle_step_y                          angle_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  angle_offset_first_element_in_bytes   TThe offset of the first element of the output
 * @param[out] non_max_ptr                           Pointer to the non maximum suppressed output. Supported data types: U16, U32
 * @param[in]  non_max_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  non_max_step_x                        non_max_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  non_max_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  non_max_step_y                        non_max_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  non_max_offset_first_element_in_bytes The offset of the first element of the output
 * @param[in]  lower_thr                             The low threshold
 */
__kernel void suppress_non_maximum(
    IMAGE_DECLARATION(grad),
    IMAGE_DECLARATION(angle),
    IMAGE_DECLARATION(non_max),
    uint lower_thr)
{
    // Construct images
    Image grad    = CONVERT_TO_IMAGE_STRUCT(grad);
    Image angle   = CONVERT_TO_IMAGE_STRUCT(angle);
    Image non_max = CONVERT_TO_IMAGE_STRUCT(non_max);

    // Get gradient and angle
    DATA_TYPE_IN gradient = *((__global DATA_TYPE_IN *)grad.ptr);
    uchar an              = convert_ushort(*angle.ptr);

    if(gradient <= lower_thr)
    {
        return;
    }

    // Divide the whole round into 8 directions
    uchar         ang  = 127 - an;
    DATA_TYPE_OUT q_an = (ang + 16) >> 5;

    // Find the two pixels in the perpendicular direction
    short2       x_p = neighbours_coords[q_an].s02;
    short2       y_p = neighbours_coords[q_an].s13;
    DATA_TYPE_IN g1  = *((global DATA_TYPE_IN *)offset(&grad, x_p.x, y_p.x));
    DATA_TYPE_IN g2  = *((global DATA_TYPE_IN *)offset(&grad, x_p.y, y_p.y));

    if((gradient > g1) && (gradient > g2))
    {
        *((global DATA_TYPE_OUT *)non_max.ptr) = gradient;
    }
}

#define EDGE 255
#define hysteresis_local_stack_L1 8  // The size of level 1 stack. This has to agree with the host side
#define hysteresis_local_stack_L2 16 // The size of level 2 stack, adjust this can impact the match rate with VX implementation

/** Check whether pixel is valid
 *
 * Skip the pixel if the early_test fails.
 * Otherwise, it tries to add the pixel coordinate to the stack, and proceed to popping the stack instead if the stack is full
 *
 * @param[in] early_test Boolean condition based on the minv check and visited buffer check
 * @param[in] x_pos      X-coordinate of pixel that is going to be recorded, has to be within the boundary
 * @param[in] y_pos      Y-coordinate of pixel that is going to be recorded, has to be within the boundary
 * @param[in] x_cur      X-coordinate of current central pixel
 * @param[in] y_cur      Y-coordinate of current central pixel
 */
#define check_pixel(early_test, x_pos, y_pos, x_cur, y_cur)                               \
    {                                                                                     \
        if(!early_test)                                                                   \
        {                                                                                 \
            /* Number of elements in the local stack 1, points to next available entry */ \
            c = *((__global char *)offset(&l1_stack_counter, x_cur, y_cur));              \
            \
            if(c > (hysteresis_local_stack_L1 - 1)) /* Stack level 1 is full */           \
                goto pop_stack;                                                           \
            \
            /* The pixel that has already been recorded is ignored */                     \
            if(!atomic_or((__global uint *)offset(&recorded, x_pos, y_pos), 1))           \
            {                                                                             \
                l1_ptr[c] = (short2)(x_pos, y_pos);                                       \
                *((__global char *)offset(&l1_stack_counter, x_cur, y_cur)) += 1;         \
            }                                                                             \
        }                                                                                 \
    }

/** Perform hysteresis.
 *
 * @attention The input data_type needs to be passed at compile time using -DDATA_TYPE_IN: e.g. -DDATA_TYPE_IN=short
 *
 * @param[in]  src_ptr                                        Pointer to the input image. Supported data types: U8
 * @param[in]  src_stride_x                                   Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                                     src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                                   Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                                     src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes              The offset of the first element of the output
 * @param[out] out_ptr                                        Pointer to the output image. Supported data types: U8
 * @param[in]  out_stride_x                                   Stride of the source image in X dimension (in bytes)
 * @param[in]  out_step_x                                     out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                                   Stride of the source image in Y dimension (in bytes)
 * @param[in]  out_step_y                                     out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes              The offset of the first element of the output
 * @param[out] visited_ptr                                    Pointer to the visited buffer, where pixels are marked as visited. Supported data types: U32
 * @param[in]  visited_stride_x                               Stride of the source image in X dimension (in bytes)
 * @param[in]  visited_step_x                                 visited_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  visited_stride_y                               Stride of the source image in Y dimension (in bytes)
 * @param[in]  visited_step_y                                 visited_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  visited_offset_first_element_in_bytes          The offset of the first element of the output
 * @param[out] recorded_ptr                                   Pointer to the recorded buffer, where pixels are marked as recorded. Supported data types: U32
 * @param[in]  recorded_stride_x                              Stride of the source image in X dimension (in bytes)
 * @param[in]  recorded_step_x                                recorded_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  recorded_stride_y                              Stride of the source image in Y dimension (in bytes)
 * @param[in]  recorded_step_y                                recorded_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  recorded_offset_first_element_in_bytes         The offset of the first element of the output
 * @param[out] l1_stack_ptr                                   Pointer to the l1 stack of a pixel. Supported data types: S32
 * @param[in]  l1_stack_stride_x                              Stride of the source image in X dimension (in bytes)
 * @param[in]  l1_stack_step_x                                l1_stack_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  l1_stack_stride_y                              Stride of the source image in Y dimension (in bytes)
 * @param[in]  l1_stack_step_y                                l1_stack_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  l1_stack_offset_first_element_in_bytes         The offset of the first element of the output
 * @param[out] l1_stack_counter_ptr                           Pointer to the l1 stack counters of an image. Supported data types: U8
 * @param[in]  l1_stack_counter_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  l1_stack_counter_step_x                        l1_stack_counter_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  l1_stack_counter_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  l1_stack_counter_step_y                        l1_stack_counter_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  l1_stack_counter_offset_first_element_in_bytes The offset of the first element of the output
 * @param[in]  low_thr                                        The lower threshold
 * @param[in]  up_thr                                         The upper threshold
 * @param[in]  width                                          The width of the image.
 * @param[in]  height                                         The height of the image
 */
kernel void hysteresis(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(out),
    IMAGE_DECLARATION(visited),
    IMAGE_DECLARATION(recorded),
    IMAGE_DECLARATION(l1_stack),
    IMAGE_DECLARATION(l1_stack_counter),
    uint low_thr,
    uint up_thr,
    int  width,
    int  height)
{
    // Create images
    Image src              = CONVERT_TO_IMAGE_STRUCT_NO_STEP(src);
    Image out              = CONVERT_TO_IMAGE_STRUCT_NO_STEP(out);
    Image visited          = CONVERT_TO_IMAGE_STRUCT_NO_STEP(visited);
    Image recorded         = CONVERT_TO_IMAGE_STRUCT_NO_STEP(recorded);
    Image l1_stack         = CONVERT_TO_IMAGE_STRUCT_NO_STEP(l1_stack);
    Image l1_stack_counter = CONVERT_TO_IMAGE_STRUCT_NO_STEP(l1_stack_counter);

    // Index
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Load value
    DATA_TYPE_IN val = *((__global DATA_TYPE_IN *)offset(&src, x, y));

    // If less than upper threshold set to NO_EDGE and return
    if(val <= up_thr)
    {
        *offset(&out, x, y) = 0;
        return;
    }

    // Init local stack 2
    short2 stack_L2[hysteresis_local_stack_L2] = { 0 };
    int    L2_counter                          = 0;

    // Perform recursive hysteresis
    while(true)
    {
        // Get L1 stack pointer
        __global short2 *l1_ptr = (__global short2 *)(l1_stack.ptr + y * l1_stack.stride_y + x * hysteresis_local_stack_L1 * l1_stack.stride_x);

        // If the pixel has already been visited, proceed with the items in the stack instead
        if(atomic_or((__global uint *)offset(&visited, x, y), 1) != 0)
        {
            goto pop_stack;
        }

        // Set strong edge
        *offset(&out, x, y) = EDGE;

        // If it is the top of stack l2, we don't need check the surrounding pixels
        if(L2_counter > (hysteresis_local_stack_L2 - 1))
        {
            goto pop_stack2;
        }

        // Points to the start of the local stack;
        char c;

        VEC_DATA_TYPE(DATA_TYPE_IN, 4)
        x_tmp;
        uint4 v_tmp;

        // Get direction pixel indices
        int N = max(y - 1, 0), S = min(y + 1, height - 2), W = max(x - 1, 0), E = min(x + 1, width - 2);

        // Check 8 pixels around for week edges where low_thr < val <= up_thr
        x_tmp = vload4(0, (__global DATA_TYPE_IN *)offset(&src, W, N));
        v_tmp = vload4(0, (__global uint *)offset(&visited, W, N));
        check_pixel(((x_tmp.s0 <= low_thr) || v_tmp.s0 || (x_tmp.s0 > up_thr)), W, N, x, y); // NW
        check_pixel(((x_tmp.s1 <= low_thr) || v_tmp.s1 || (x_tmp.s1 > up_thr)), x, N, x, y); // N
        check_pixel(((x_tmp.s2 <= low_thr) || v_tmp.s2 || (x_tmp.s2 > up_thr)), E, N, x, y); // NE

        x_tmp = vload4(0, (__global DATA_TYPE_IN *)offset(&src, W, y));
        v_tmp = vload4(0, (__global uint *)offset(&visited, W, y));
        check_pixel(((x_tmp.s0 <= low_thr) || v_tmp.s0 || (x_tmp.s0 > up_thr)), W, y, x, y); // W
        check_pixel(((x_tmp.s2 <= low_thr) || v_tmp.s2 || (x_tmp.s2 > up_thr)), E, y, x, y); // E

        x_tmp = vload4(0, (__global DATA_TYPE_IN *)offset(&src, W, S));
        v_tmp = vload4(0, (__global uint *)offset(&visited, W, S));
        check_pixel(((x_tmp.s0 <= low_thr) || v_tmp.s0 || (x_tmp.s0 > up_thr)), W, S, x, y); // SW
        check_pixel(((x_tmp.s1 <= low_thr) || v_tmp.s1 || (x_tmp.s1 > up_thr)), x, S, x, y); // S
        check_pixel(((x_tmp.s2 <= low_thr) || v_tmp.s2 || (x_tmp.s2 > up_thr)), E, S, x, y); // SE

#undef check_pixel

pop_stack:
        c = *((__global char *)offset(&l1_stack_counter, x, y));

        if(c >= 1)
        {
            *((__global char *)offset(&l1_stack_counter, x, y)) -= 1;
            int2 l_c = convert_int2(l1_ptr[c - 1]);

            // Push the current position into level 2 stack
            stack_L2[L2_counter].x = x;
            stack_L2[L2_counter].y = y;

            x = l_c.x;
            y = l_c.y;

            L2_counter++;

            continue;
        }

        if(L2_counter > 0)
        {
            goto pop_stack2;
        }
        else
        {
            return;
        }

pop_stack2:
        L2_counter--;
        x = stack_L2[L2_counter].x;
        y = stack_L2[L2_counter].y;
    };
}
