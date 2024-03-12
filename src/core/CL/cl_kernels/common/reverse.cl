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

#if defined(DATA_TYPE) && defined(NUM_REVERSE_DIMS)

#if NUM_REVERSE_DIMS > 4
#error("Reversing more than 4 dimensions is not currently supported")
#endif /* NUM_REVERSE_DIMS > 4 */

/** Performs reverse along the specified axis.
 *
 * @note The data type must be given as a preprocessor argument using -DDATA_TYPE=num. e.g. -DDATA_TYPE=uint
 * @note The number of dimensions to reverse must be given as a preprocessor argument using -DNUM_REVERSE_DIMS=num, e.g. -DNUM_REVERSE_DIMS=3
 * @note The number of dimensions of the source tensor must be given as a preprocessor argument using -DRANK=num, e.g. -DRANK=3
 * @note The values in axis_tensor must be within [-rank, rank-1].
 *
 * @param[in]  src_ptr                            Pointer to the source tensor. Supported data types: All
 * @param[in]  src_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                         src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_w                         src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[in]  axis_ptr                           Pointer to the source vector. Supported data types: U32
 * @param[in]  axis_stride_x                      Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  axis_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  axis_offset_first_element_in_bytes The offset of the first element in the first source tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_w                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_w                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 */
__kernel void reverse(TENSOR4D_DECLARATION(src),
                      VECTOR_DECLARATION(axis),
                      TENSOR4D_DECLARATION(dst),
                      const uint width,
                      const uint height,
                      const uint depth,
                      const uint batches)
{
    Tensor4D src  = CONVERT_TO_TENSOR4D_STRUCT(src, depth);
    Vector   axis = CONVERT_TO_VECTOR_STRUCT_NO_STEP(axis);
    Tensor4D dst  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(dst, depth);

    const uint x_in = get_global_id(0);
    const uint y_in = get_global_id(1);
    const uint z_in = get_global_id(2) % depth;
    const uint w_in = get_global_id(2) / depth;

    const uint4 dims       = (uint4)(0, 1, 2, 3);
    int4        to_reverse = (int4)(0, 0, 0, 0);

    VEC_DATA_TYPE(int, NUM_REVERSE_DIMS) indices =  VLOAD(NUM_REVERSE_DIMS)(0,(__global int *)axis.ptr);
#if defined(USE_INVERTED_AXIS)
    indices    = select((VEC_DATA_TYPE(int, NUM_REVERSE_DIMS)) RANK - 1, -1, indices < 0) - indices;
#else /* defined(USE_INVERTED_AXIS) */
    indices    = select(indices, indices + RANK, indices < 0);
#endif /* defined(USE_INVERTED_AXIS) */

#if NUM_REVERSE_DIMS == 1
    to_reverse = ((uint4)indices == dims);
#elif NUM_REVERSE_DIMS == 2
    to_reverse = ((uint4)indices.s0 == dims) || ((uint4)indices.s1 == dims);
#elif NUM_REVERSE_DIMS == 3
    to_reverse = ((uint4)indices.s0 == dims) || ((uint4)indices.s1 == dims) || ((uint4)indices.s2 == dims);
#else /* NUM_REVERSE_DIMS == 1 */
    to_reverse    = ((uint4)indices.s0 == dims) || ((uint4)indices.s1 == dims) || ((uint4)indices.s2 == dims) || ((uint4)indices.s3 == dims);
#endif /* NUM_REVERSE_DIMS == 1 */

    const uint x_out = to_reverse.s0 ? width - x_in - 1 : x_in;
    const uint y_out = to_reverse.s1 ? height - y_in - 1 : y_in;
    const uint z_out = to_reverse.s2 ? depth - z_in - 1 : z_in;
    const uint w_out = to_reverse.s3 ? batches - w_in - 1 : w_in;

    *((__global DATA_TYPE *)tensor4D_offset(&dst, x_out, y_out, z_out, w_out)) = *((__global DATA_TYPE *)src.ptr);
}
#endif // defined(DATA_TYPE) && defined(NUM_REVERSE_DIMS)
