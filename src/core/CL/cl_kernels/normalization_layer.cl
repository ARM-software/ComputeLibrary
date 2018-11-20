/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#define MUL_OP(x, y) ((x) * (y))
#define ADD_OP(x, y) ((x) + (y))
#define DIV_OP(x, y) ((x) / (y))
#define POW_OP(x, y) pow((x), (y))
#define SQCVT_SAT(a) (a)

#define LOAD_OP(offset, ptr) vload4(offset, ptr)
#define STORE_OP(data, offset, ptr) vstore4(data, offset, ptr)

#if defined(NUM_SLICES)
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
__kernel void normalization_layer_cross_map(TENSOR3D_DECLARATION(input),
                                            TENSOR3D_DECLARATION(output))
{
    Tensor3D in  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    acc = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))0;
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    coeff_v = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(COEFF);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    beta_v = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(BETA);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    kappa_v = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(KAPPA);

    const int current_slice = get_global_id(2);
    const int left_slice    = max(-(int)RADIUS, -current_slice);
    const int right_slice   = min((int)RADIUS, (int)NUM_SLICES - 1 - current_slice);

    for(int i = left_slice; i <= right_slice; i++)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        values = LOAD_OP(0, (__global DATA_TYPE *)tensor3D_offset(&in, 0, 0, i));
        acc    = ADD_OP(acc, MUL_OP(values, values));
    }

    acc = ADD_OP(MUL_OP(acc, coeff_v), kappa_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized = POW_OP(acc, beta_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized_pixel = DIV_OP(LOAD_OP(0, (__global DATA_TYPE *)in.ptr), normalized);

    STORE_OP(normalized_pixel, 0, (__global DATA_TYPE *)out.ptr);
}
#endif /* defined(NUM_SLICES) */

#if defined(WIDTH_SIZE)
/** Apply in-map normalization when tensors are in the NCHW data layout format.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size, e.g. -DVEC_SIZE=16
 * @note The radius should be given as a preprocessor argument using -DRADIUS=size. e.g. -DRADIUS=5
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
__kernel void normalization_layer_in_map_nchw(TENSOR3D_DECLARATION(input),
                                              TENSOR3D_DECLARATION(output))
{
    Tensor3D in  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    acc = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))0;
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    coeff_v = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(COEFF);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    beta_v = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(BETA);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    kappa_v = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(KAPPA);

    const int current_col = get_global_id(0) << 2;
    const int left_pos    = max(-(int)RADIUS, -3 - current_col);
    const int right_pos   = min((int)RADIUS, (int)WIDTH_SIZE - 1 - current_col);

#if defined(IN_MAP_2D)
    const int current_row = get_global_id(1);
    const int first_row   = max(-(int)RADIUS, -current_row);
    const int last_row    = min((int)RADIUS, (int)get_global_size(1) - 1 - current_row);
#endif /* defined(IN_MAP_2D) */

#if defined(IN_MAP_2D)
    for(int j = first_row; j <= last_row; ++j)
    {
#endif /* defined(IN_MAP_2D) */
        for(int i = left_pos; i <= right_pos; ++i)
        {
#if defined(IN_MAP_2D)
            VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
            values = LOAD_OP(0, (__global DATA_TYPE *)tensor3D_offset(&in, i, j, 0));
#else  /* defined(IN_MAP_2D) */
            VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
            values = LOAD_OP(0, (__global DATA_TYPE *)tensor3D_offset(&in, i, 0, 0));
#endif /* defined(IN_MAP_2D) */
            acc = ADD_OP(acc, MUL_OP(values, values));
        }
#if defined(IN_MAP_2D)
    }
#endif /* defined(IN_MAP_2D) */

    acc = ADD_OP(MUL_OP(acc, coeff_v), kappa_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized = POW_OP(acc, beta_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized_pixel = DIV_OP(LOAD_OP(0, (__global DATA_TYPE *)in.ptr), normalized);

    STORE_OP(normalized_pixel, 0, (__global DATA_TYPE *)out.ptr);
}
#endif // defined(WIDTH_SIZE)

#if defined(NUM_SLICES)
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
    Tensor3D in  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    acc = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))0;
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    coeff_v = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(COEFF);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    beta_v = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(BETA);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    kappa_v = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(KAPPA);

    const int current_cols = get_global_id(1);
    const int first_col    = max(-(int)RADIUS, -current_cols);
    const int last_col     = min((int)RADIUS, (int)get_global_size(1) - 1 - current_cols);

#if defined(IN_MAP_2D)
    const int current_rows = get_global_id(2);
    const int first_row    = max(-(int)RADIUS, -current_rows);
    const int last_row     = min((int)RADIUS, (int)NUM_SLICES - 1 - current_rows);
#endif /* defined(IN_MAP_2D) */

#if defined(IN_MAP_2D)
    for(int j = first_row; j <= last_row; ++j)
    {
#endif /* defined(IN_MAP_2D) */
        for(int i = first_col; i <= last_col; ++i)
        {
#if defined(IN_MAP_2D)
            VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
            values = LOAD_OP(0, (__global DATA_TYPE *)tensor3D_offset(&in, 0, i, j));
#else  /* defined(IN_MAP_2D) */
            VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
            values = LOAD_OP(0, (__global DATA_TYPE *)tensor3D_offset(&in, 0, i, 0));
#endif /* defined(IN_MAP_2D) */
            acc = ADD_OP(acc, MUL_OP(values, values));
        }
#if defined(IN_MAP_2D)
    }
#endif /* defined(IN_MAP_2D) */

    acc = ADD_OP(MUL_OP(acc, coeff_v), kappa_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized = POW_OP(acc, beta_v);
    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    normalized_pixel = DIV_OP(LOAD_OP(0, (__global DATA_TYPE *)in.ptr), normalized);

    STORE_OP(normalized_pixel, 0, (__global DATA_TYPE *)out.ptr);
}
#endif /* defined(NUM_SLICES) */
