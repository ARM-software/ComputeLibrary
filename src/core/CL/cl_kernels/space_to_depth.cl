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

#if defined(DATA_TYPE) && defined(BLOCK_SHAPE) && defined(CHANNEL_SIZE)
/** Space to depth transformation. (NCHW)
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The input tensor batch size must be passed at compile time using -DCHANNEL_SIZE. e.g. -DCHANNEL_SIZE=2
 * @note The block shape must be passed at compile time using -DBLOCK_SHAPE. e.g. -DBLOCK_SHAPE=2
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
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
__kernel void space_to_depth_nchw(
    TENSOR4D_DECLARATION(input),
    const int batch_id,
    TENSOR3D_DECLARATION(output))
{
    Tensor4D in  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(output);

    const int r = (CHANNEL_SIZE / (BLOCK_SHAPE * BLOCK_SHAPE));
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2) % r;

    const int in_x = x * BLOCK_SHAPE + (get_global_id(2) / r) % BLOCK_SHAPE;
    const int in_y = y * BLOCK_SHAPE + (get_global_id(2) / r) / BLOCK_SHAPE;

    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, in_x, in_y, z, batch_id));
}
/** Space to depth transformation. (NHWC)
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note The input tensor batch size must be passed at compile time using -DCHANNEL_SIZE. e.g. -DCHANNEL_SIZE=2
 * @note The block shape must be passed at compile time using -DBLOCK_SHAPE. e.g. -DBLOCK_SHAPE=2
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
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
__kernel void space_to_depth_nhwc(
    TENSOR4D_DECLARATION(input),
    const int batch_id,
    TENSOR3D_DECLARATION(output))
{
    Tensor4D in  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(output);

    const int r = (CHANNEL_SIZE / (BLOCK_SHAPE * BLOCK_SHAPE));
    const int x = get_global_id(1);
    const int y = get_global_id(2);
    const int z = get_global_id(0) % r;

    const int in_x = x * BLOCK_SHAPE + (get_global_id(0) / r) % BLOCK_SHAPE;
    const int in_y = y * BLOCK_SHAPE + (get_global_id(0) / r) / BLOCK_SHAPE;

    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, z, in_x, in_y, batch_id));
}
#endif // defined(DATA_TYPE) && defined(BLOCK_SHAPE) && defined(CHANNEL_SIZE)