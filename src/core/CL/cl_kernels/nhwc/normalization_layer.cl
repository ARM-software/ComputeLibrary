/*
 * Copyright (c) 2017-2021 Arm Limited.
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

#define MUL_OP(x, y) ((x) * (y))
#define ADD_OP(x, y) ((x) + (y))
#define DIV_OP(x, y) ((x) / (y))
#define POW_OP(x, y) pow((x), (y))
#define SQCVT_SAT(a) (a)

#if defined(WIDTH_SIZE)
/** Apply cross-map normalization.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size, e.g. -DVEC_SIZE=16
 * @note The radius should be given as a preprocessor argument using -DRADIUS=size. e.g. -DRADIUS=5
 * @note The number of slices should be given as a preprocessor argument using -DNUM_SLICES=size. e.g. -DNUM_SLICES=192
 * @note Scaling coefficient (= alpha/norm_size), beta and kappa need to be passed at compile time using -DCOEFF, -DALPHA and -DKAPPA
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void normalization_layer_cross_map_nhwc(TENSOR3D_DECLARATION(input),
                                                 TENSOR3D_DECLARATION(output))
{
    // Offset computation
    const uint x_offs = GET_SPATIAL_IDX(0, VEC_SIZE, VEC_SIZE_LEFTOVER);

    // Address computation
    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + get_global_id(1) * input_stride_y + get_global_id(2) * input_stride_z;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * output_stride_y + get_global_id(2) * output_stride_z;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    acc = 0;
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    coeff_v = SQCVT_SAT(COEFF);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    beta_v = SQCVT_SAT(BETA);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    kappa_v = SQCVT_SAT(KAPPA);

    const int left_slice  = max((int)0, (int)x_offs - (int)RADIUS);
    const int right_slice = min((int)WIDTH_SIZE - 1, (int)x_offs + (int)RADIUS);

    for(int i = left_slice; i <= right_slice; ++i)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + i * sizeof(DATA_TYPE)));
        acc    = ADD_OP(acc, MUL_OP(values, values));
    }

    acc = ADD_OP(MUL_OP(acc, coeff_v), kappa_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized = POW_OP(acc, beta_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized_pixel0 = DIV_OP(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + x_offs * sizeof(DATA_TYPE))), normalized);

    STORE_VECTOR_SELECT(normalized_pixel, DATA_TYPE, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif // defined(WIDTH_SIZE)

#if defined(NUM_SLICES) && defined(DIM1_SIZE)
/** Apply in-map normalization when tensors are in the NHWC data layout format.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size, e.g. -DVEC_SIZE=16
 * @note The radius should be given as a preprocessor argument using -DRADIUS=size. e.g. -DRADIUS=5
 * @note The number of slices should be given as a preprocessor argument using -DNUM_SLICES=size. e.g. -DNUM_SLICES=192
 * @note Scaling coefficient (= alpha/norm_size), beta and kappa need to be passed at compile time using -DCOEFF, -DALPHA and -DKAPPA
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the first destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void normalization_layer_in_map_nhwc(TENSOR3D_DECLARATION(input),
                                              TENSOR3D_DECLARATION(output))
{
    // Offset computation
    const uint x_offs       = GET_SPATIAL_IDX(0, VEC_SIZE, VEC_SIZE_LEFTOVER);
    const int  current_cols = get_global_id(1);
    const int  current_rows = get_global_id(2);

    // Address computation
    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE);
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + current_cols * output_stride_y + current_rows * output_stride_z;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    acc = 0;
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    coeff_v = SQCVT_SAT(COEFF);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    beta_v = SQCVT_SAT(BETA);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    kappa_v = SQCVT_SAT(KAPPA);

    const int first_col = max(0, current_cols - (int)RADIUS);
    const int last_col  = min((int)DIM1_SIZE - 1, current_cols + (int)RADIUS);

#if defined(IN_MAP_2D)
    const int first_row = max(0, current_rows - (int)RADIUS);
    const int last_row  = min((int)NUM_SLICES - 1, current_rows + (int)RADIUS);
#endif /* defined(IN_MAP_2D) */

#if defined(IN_MAP_2D)
    for(int j = first_row; j <= last_row; ++j)
    {
#else  // defined(IN_MAP_2D)
    const int j = current_rows;
#endif /* defined(IN_MAP_2D) */
        for(int i = first_col; i <= last_col; ++i)
        {
            VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
            values = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + i * input_stride_y + j * input_stride_z));
            acc    = ADD_OP(acc, MUL_OP(values, values));
        }
#if defined(IN_MAP_2D)
    }
#endif /* defined(IN_MAP_2D) */

    acc = ADD_OP(MUL_OP(acc, coeff_v), kappa_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized = POW_OP(acc, beta_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized_pixel0 = DIV_OP(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + current_cols * output_stride_y + current_rows *output_stride_z)), normalized);

    STORE_VECTOR_SELECT(normalized_pixel, DATA_TYPE, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif // defined(NUM_SLICES) && defined(DIM1_SIZE)