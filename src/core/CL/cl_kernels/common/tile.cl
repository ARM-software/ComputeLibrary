/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#if defined(DATA_TYPE) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(SRC_DEPTH) && defined(DST_DEPTH)
/** Perform a floor operation on an input tensor.
 *
 * @attention Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Can only take floating point data types.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: All
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void tile(
    TENSOR4D_DECLARATION(input),
    TENSOR4D_DECLARATION(output))
{
    Tensor4D output = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(output, DST_DEPTH);
    Tensor4D input  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, SRC_DEPTH);

    // For all coordinates but x, each tile copies from the input
    const int y     = get_global_id(1);
    const int z     = get_global_id(2) % DST_DEPTH;
    const int batch = get_global_id(2) / DST_DEPTH;

#if defined(VEC_SIZE) && defined(OFFSET)
    // If we are loading/storing multiple elements at time, we need to
    // not exceed the input boundaries. The last threads need to backtrack
    // of OFFSET elements. Those elements cumulates for previous tiles

    const int id          = (int)(get_global_id(0));
    const int multiple_no = id / SRC_WIDTH_TILES;
    const int tile_no     = id % SRC_WIDTH_TILES;
    const int last_tile   = (int)(tile_no == SRC_WIDTH_TILES - 1);

    const int x_input  = tile_no * VEC_SIZE - last_tile * OFFSET;
    const int x_output = multiple_no * SRC_WIDTH + x_input;

    // Update the input and output pointers.
    input.ptr  = tensor4D_offset(&input, x_input, y % SRC_HEIGHT, z % SRC_DEPTH, batch % SRC_BATCHES);
    output.ptr = tensor4D_offset(&output, x_output, y, z, batch);

    // Copy the data
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input.ptr);

    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)output.ptr);
#else  // !defined(VEC_SIZE) || !defined(OFFSET)
    const int x = get_global_id(0);

    // Update the input and output pointers.
    input.ptr  = tensor4D_offset(&input, x % SRC_WIDTH, y % SRC_HEIGHT, z % SRC_DEPTH, batch % SRC_BATCHES);
    output.ptr = tensor4D_offset(&output, x, y, z, batch);

    *((__global DATA_TYPE *)(output.ptr)) = *((__global DATA_TYPE *)(input.ptr));
#endif // defined(VEC_SIZE) && defined(OFFSET)
}
#endif // defined(DATA_TYPE) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(SRC_DEPTH) && defined(DST_DEPTH)
