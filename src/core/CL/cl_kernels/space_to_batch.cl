/*
 * Copyright (c) 2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software withoutput restriction, including withoutput limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KOUTD, EXPRESS OR
 * IMPLIED, OUTCLUDOUTG BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONOUTFROUTGEMENT. OUT NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER OUT AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISOUTG FROM,
 * OUT OF OR OUT CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALOUTGS OUT THE
 * SOFTWARE.
 */
#include "helpers.h"

#if defined(BATCH_SIZE) && defined(DATA_TYPE) && defined(WIDTH_IN) && defined(HEIGHT_IN)
/** Calculate the space to batch conversion.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The block shape tensor rank must be passed at compile time using -DBLOCK_SHAPE_DIM. e.g. -DBLOCK_SHAPE_DIM=2
 *
 * @param[in]  input_ptr                                 Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                            Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                              input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                            Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                              input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                            Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                              input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes       The offset of the first element in the first source image
 * @param[in]  paddings_ptr                              Pointer to the second source image. Supported data types: S32
 * @param[in]  paddings_stride_x                         Stride of the paddinds tensor in X dimension (in bytes)
 * @param[in]  paddings_step_x                           paddings_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  paddings_stride_y                         Stride of the paddinds tensor in Y dimension (in bytes)
 * @param[in]  paddings_step_y                           paddings_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  paddingse_offset_first_element_in_bytes   The offset of the first element in the second source image
 * @param[in]  block_shape_ptr                           Pointer to the block shape tensor. Supported data types: S32
 * @param[in]  block_shape_stride_x                      Stride of the block shape tensor in X dimension (in bytes)
 * @param[in]  block_shape_step_x                        block_shape_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  block_shape_stride_y                      Stride of the block shape tensor in Y dimension (in bytes)
 * @param[in]  block_shape_step_y                        block_shape_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  block_shape_offset_first_element_in_bytes The offset of the first element in the block shapetensor
 * @param[in]  batch_id                                  The output tensor batch id
 * @param[out] output_ptr                                Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                           Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                             output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                           Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                             output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                           Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                             output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes      The offset of the first element in the destination image
 */
__kernel void space_to_batch_nchw(
    TENSOR4D_DECLARATION(input),
    IMAGE_DECLARATION(paddings),
    VECTOR_DECLARATION(block_shape),
    const int batch_id,
    TENSOR3D_DECLARATION(output))
{
    Tensor4D in    = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
    Image    pad   = CONVERT_TO_IMAGE_STRUCT_NO_STEP(paddings);
    Vector   block = CONVERT_TO_VECTOR_STRUCT_NO_STEP(block_shape);
    Tensor3D out   = CONVERT_TO_TENSOR3D_STRUCT(output);

    const int pad_left_x  = *((__global int *)offset(&pad, 0, 0));
    const int pad_right_x = *((__global int *)offset(&pad, 1, 0));
    const int pad_left_y  = *((__global int *)offset(&pad, 0, 1));
    const int pad_right_y = *((__global int *)offset(&pad, 1, 1));

    int block_x = *((__global int *)vector_offset(&block, 0));
    int block_y = *((__global int *)vector_offset(&block, 1));

    const int out_x = get_global_id(0);
    const int out_y = get_global_id(1);
    const int z     = get_global_id(2);

    const int pos_x = out_x * block_x + ((batch_id / BATCH_IN) % block_x);
    const int pos_y = out_y * block_y + ((batch_id / BATCH_IN) / block_x);

    if(((pos_y >= pad_left_y) && (pos_y < pad_left_y + HEIGHT_IN) && (pos_x >= pad_left_x) && (pos_x < pad_left_x + WIDTH_IN)))
    {
        const int w    = batch_id % BATCH_IN;
        const int in_x = pos_x - pad_left_x;
        const int in_y = pos_y - pad_left_y;

        *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, in_x, in_y, z, w));
    }
}
/** Calculate the space to batch conversion. (NHWC)
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The block shape tensor rank must be passed at compile time using -DBLOCK_SHAPE_DIM. e.g. -DBLOCK_SHAPE_DIM=2
 *
 * @param[in]  input_ptr                                 Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                            Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                              input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                            Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                              input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                            Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                              input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes       The offset of the first element in the first source image
 * @param[in]  paddings_ptr                              Pointer to the second source image. Supported data types: S32
 * @param[in]  paddings_stride_x                         Stride of the paddinds tensor in X dimension (in bytes)
 * @param[in]  paddings_step_x                           paddings_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  paddings_stride_y                         Stride of the paddinds tensor in Y dimension (in bytes)
 * @param[in]  paddings_step_y                           paddings_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  paddingse_offset_first_element_in_bytes   The offset of the first element in the second source image
 * @param[in]  block_shape_ptr                           Pointer to the block shape tensor. Supported data types: S32
 * @param[in]  block_shape_stride_x                      Stride of the block shape tensor in X dimension (in bytes)
 * @param[in]  block_shape_step_x                        block_shape_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  block_shape_stride_y                      Stride of the block shape tensor in Y dimension (in bytes)
 * @param[in]  block_shape_step_y                        block_shape_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  block_shape_offset_first_element_in_bytes The offset of the first element in the block shapetensor
 * @param[in]  batch_id                                  The output tensor batch id
 * @param[out] output_ptr                                Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                           Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                             output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                           Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                             output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                           Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                             output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes      The offset of the first element in the destination image
 */
__kernel void space_to_batch_nhwc(
    TENSOR4D_DECLARATION(input),
    IMAGE_DECLARATION(paddings),
    VECTOR_DECLARATION(block_shape),
    const int batch_id,
    TENSOR3D_DECLARATION(output))
{
    Tensor4D in    = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
    Image    pad   = CONVERT_TO_IMAGE_STRUCT_NO_STEP(paddings);
    Vector   block = CONVERT_TO_VECTOR_STRUCT_NO_STEP(block_shape);
    Tensor3D out   = CONVERT_TO_TENSOR3D_STRUCT(output);

    const int pad_left_x  = *((__global int *)offset(&pad, 0, 0));
    const int pad_right_x = *((__global int *)offset(&pad, 1, 0));
    const int pad_left_y  = *((__global int *)offset(&pad, 0, 1));
    const int pad_right_y = *((__global int *)offset(&pad, 1, 1));

    int block_x = *((__global int *)vector_offset(&block, 0));
    int block_y = *((__global int *)vector_offset(&block, 1));

    const int out_x = get_global_id(1);
    const int out_y = get_global_id(2);
    const int z     = get_global_id(0);

    const int pos_x = out_x * block_x + ((batch_id / BATCH_IN) % block_x);
    const int pos_y = out_y * block_y + ((batch_id / BATCH_IN) / block_x);

    if(((pos_y >= pad_left_y) && (pos_y < pad_left_y + HEIGHT_IN) && (pos_x >= pad_left_x) && (pos_x < pad_left_x + WIDTH_IN)))
    {
        const int w    = batch_id % BATCH_IN;
        const int in_x = pos_x - pad_left_x;
        const int in_y = pos_y - pad_left_y;

        *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, z, in_x, in_y, w));
    }
}
#endif // defined(BATCH_SIZE) && defined(DATA_TYPE)  && defined(WIDTH_IN) && defined(HEIGHT_IN)

#if defined(BATCH_SIZE) && defined(DATA_TYPE) && defined(BLOCK_SHAPE_X) && defined(BLOCK_SHAPE_Y) && defined(PAD_LEFT_X) && defined(PAD_RIGHT_X) && defined(PAD_LEFT_Y) && defined(PAD_RIGHT_Y) && defined(WIDTH_IN) && defined(HEIGHT_IN)
/** Calculate the space to batch conversion.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The input tensor batch size must be passed at compile time using -DBATCH_SIZE. e.g. -DBATCH_SIZE=2
 * @note The block shape x must be passed at compile time using -DBLOCK_SHAPE_X. e.g. -DBLOCK_SHAPE_X=2
 * @note The block shape y must be passed at compile time using -DBLOCK_SHAPE_Y. e.g. -DBLOCK_SHAPE_Y=2
 * @note The starting pad value of x must be passed at compile time using -DPAD_LEFT_X. e.g. -DPAD_LEFT_X=2
 * @note The ending pad value of x must be passed at compile time using -DPAD_RIGHT_X. e.g. -DPAD_RIGHT_X=2
 * @note The starting pad value of y must be passed at compile time using -DPAD_LEFT_Y. e.g. -DPAD_LEFT_Y=2
 * @note The ending pad value of  y must be passed at compile time using -DPAD_RIGHT_Y. e.g. -DPAD_RIGHT_X=2
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source image
 * @param[in]  batch_id                             The output tensor batch id
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void space_to_batch_static_nchw(
    TENSOR4D_DECLARATION(input),
    const int batch_id,
    TENSOR3D_DECLARATION(output))
{
    Tensor4D in  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(output);

    int block_x = BLOCK_SHAPE_X;
    int block_y = BLOCK_SHAPE_Y;

    const int out_x = get_global_id(0);
    const int out_y = get_global_id(1);
    const int z     = get_global_id(2);

    const int pos_x = out_x * block_x + ((batch_id / BATCH_IN) % block_x);
    const int pos_y = out_y * block_y + ((batch_id / BATCH_IN) / block_x);

    if(pos_y >= PAD_LEFT_Y && pos_y < PAD_LEFT_Y + HEIGHT_IN && pos_x >= PAD_LEFT_X && pos_x < PAD_LEFT_X + WIDTH_IN)
    {
        const int w    = batch_id % BATCH_IN;
        const int in_x = pos_x - PAD_LEFT_X;
        const int in_y = pos_y - PAD_LEFT_Y;

        *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, in_x, in_y, z, w));
    }
}
/** Calculate the space to batch conversion. (NHWC)
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The input tensor batch size must be passed at compile time using -DBATCH_SIZE. e.g. -DBATCH_SIZE=2
 * @note The block shape x must be passed at compile time using -DBLOCK_SHAPE_X. e.g. -DBLOCK_SHAPE_X=2
 * @note The block shape y must be passed at compile time using -DBLOCK_SHAPE_Y. e.g. -DBLOCK_SHAPE_Y=2
 * @note The starting pad value of x must be passed at compile time using -DPAD_LEFT_X. e.g. -DPAD_LEFT_X=2
 * @note The ending pad value of x must be passed at compile time using -DPAD_RIGHT_X. e.g. -DPAD_RIGHT_X=2
 * @note The starting pad value of y must be passed at compile time using -DPAD_LEFT_Y. e.g. -DPAD_LEFT_Y=2
 * @note The ending pad value of  y must be passed at compile time using -DPAD_RIGHT_Y. e.g. -DPAD_RIGHT_X=2
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source image
 * @param[in]  batch_id                             The output tensor batch id
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void space_to_batch_static_nhwc(
    TENSOR4D_DECLARATION(input),
    const int batch_id,
    TENSOR3D_DECLARATION(output))
{
    Tensor4D in  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(output);

    int block_x = BLOCK_SHAPE_X;
    int block_y = BLOCK_SHAPE_Y;

    const int out_x = get_global_id(1);
    const int out_y = get_global_id(2);
    const int z     = get_global_id(0);

    const int pos_x = out_x * block_x + ((batch_id / BATCH_IN) % block_x);
    const int pos_y = out_y * block_y + ((batch_id / BATCH_IN) / block_x);

    if(pos_y >= PAD_LEFT_Y && pos_y < PAD_LEFT_Y + HEIGHT_IN && pos_x >= PAD_LEFT_X && pos_x < PAD_LEFT_X + WIDTH_IN)
    {
        const int w    = batch_id % BATCH_IN;
        const int in_x = pos_x - PAD_LEFT_X;
        const int in_y = pos_y - PAD_LEFT_Y;

        *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, z, in_x, in_y, w));
    }
}
#endif // defined(BATCH_SIZE) && defined(DATA_TYPE) && defined(BLOCK_SHAPE_X) && defined(BLOCK_SHAPE_Y) && defined(PAD_LEFT_X) && defined(PAD_RIGHT_X) && defined(PAD_LEFT_Y) && defined(PAD_RIGHT_Y)  && defined(WIDTH_IN) && defined(HEIGHT_IN)
