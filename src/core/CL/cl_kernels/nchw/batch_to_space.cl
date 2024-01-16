/*
 * Copyright (c) 2018-2021, 2023-2024 Arm Limited.
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

#if defined(DATA_TYPE) && defined(BATCH_SIZE)
/** Batch to space transformation. (NCHW)
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The input tensor batch size must be passed at compile time using -DBATCH_SIZE. e.g. -DBATCH_SIZE=2
 *
 * @deprecated This method for dynamic block shape is not fully mature and will be removed in 23.08 release
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: All
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[in]  batch_id                             The input tensor batch id
 * @param[in]  block_shape_ptr                      Pointer to the source tensor. Supported data types: S32
 * @param[in]  block_shape_stride_x                 Stride of the source tensor in X dimension (in bytes)
 * @param[in]  block_shape_step_x                   block_shape_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  block_shape_stride_y                 Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  block_shape_step_y                   block_shape_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void batch_to_space_nchw(
    TENSOR4D_DECLARATION(input),
    const int batch_id,
    VECTOR_DECLARATION(block_shape),
    TENSOR3D_DECLARATION(output))
{
    Tensor4D in    = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input);
    Tensor3D out   = CONVERT_TO_TENSOR3D_STRUCT(output);
    Vector   block = CONVERT_TO_VECTOR_STRUCT_NO_STEP(block_shape);

    const int block_x = *((__global int *)vector_offset(&block, 0));
    const int block_y = *((__global int *)vector_offset(&block, 1));

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    const int in_batch = batch_id + ((x % block_x) + (y % block_y) * block_x) * BATCH_SIZE;
    const int in_x     = x / block_x;
    const int in_y     = y / block_y;

    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, in_x, in_y, z, in_batch));
}
#endif // defined(DATA_TYPE) && defined(BATCH_SIZE)

#if defined(DATA_TYPE) && defined(BATCH_SIZE) && defined(BLOCK_SHAPE_X) && defined(BLOCK_SHAPE_Y)
/** Batch to space transformation. (NCHW)
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The input tensor batch size must be passed at compile time using -DBATCH_SIZE. e.g. -DBATCH_SIZE=2
 * @note The block shape x must be passed at compile time using -DBLOCK_SHAPE_X. e.g. -DBLOCK_SHAPE_X=2
 * @note The block shape y must be passed at compile time using -DBLOCK_SHAPE_Y. e.g. -DBLOCK_SHAPE_Y=2
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: All
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[in]  batch_id                             The input tensor batch id
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void batch_to_space_static_nchw(
    TENSOR4D_DECLARATION(input),
    const int batch_id,
    TENSOR3D_DECLARATION(output))
{
    Tensor4D in  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(output);

    const int block_x = BLOCK_SHAPE_X;
    const int block_y = BLOCK_SHAPE_Y;

    const int x = get_global_id(0) + CROP_LEFT;
    const int y = get_global_id(1) + CROP_TOP;
    const int z = get_global_id(2);

    const int in_batch = batch_id + ((x % block_x) + (y % block_y) * block_x) * BATCH_SIZE;
    const int in_x     = x / block_x;
    const int in_y     = y / block_y;

    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, in_x, in_y, z, in_batch));
}
#endif // defined(DATA_TYPE) && defined(BATCH_SIZE) && defined(BLOCK_SHAPE_X) && defined(BLOCK_SHAPE_Y)
