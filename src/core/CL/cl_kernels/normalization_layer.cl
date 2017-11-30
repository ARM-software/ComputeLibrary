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

#if defined(FIXED_POINT_POSITION)

#include "fixed_point.h"
#define MUL_OP(x, y) MUL_SAT_OP_EXPAND((x), (y), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)
#define ADD_OP(x, y) ADD_SAT_OP_EXPAND((x), (y), DATA_TYPE, VEC_SIZE)
#define DIV_OP(x, y) DIV_SAT_OP_VEC_EXPAND((x), (y), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)
#define EXP_OP(x) EXP_OP_EXPAND((x), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)
#define LOG_OP(x) LOG_OP_EXPAND((x), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)
#define POW_OP(x, y) EXP_OP(MUL_OP(LOG_OP((x)), (y)))
#define SQCVT_SAT(a) SQCVT_SAT_OP_EXPAND((a), DATA_TYPE, FIXED_POINT_POSITION)

#define LOAD_OP(offset, ptr) vload16(offset, ptr)
#define STORE_OP(data, offset, ptr) vstore16(data, offset, ptr)

#else // FIXED_POINT_POSITION

#define MUL_OP(x, y) ((x) * (y))
#define ADD_OP(x, y) ((x) + (y))
#define DIV_OP(x, y) ((x) / (y))
#define POW_OP(x, y) pow((x), (y))
#define SQCVT_SAT(a) (a)

#define LOAD_OP(offset, ptr) vload4(offset, ptr)
#define STORE_OP(data, offset, ptr) vstore4(data, offset, ptr)

#endif // FIXED_POINT_POSITION

/** Apply cross-map normalization.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size, e.g. -DVEC_SIZE=16
 * @note The radius should be given as a preprocessor argument using -DRADIUS=size. e.g. -DRADIUS=5
 * @note The number of slices should be given as a preprocessor argument using -DNUM_SLICES=size. e.g. -DNUM_SLICES=192
 * @note In case of fixed-point operation -DFIXED_POINT_POSITION=fixed_point_position must be provided: e.g. -DFIXED_POINT_POSITION=3
 * @note Scaling coefficient (= alpha/norm_size), beta and kappa need to be passed at compile time using -DCOEFF, -DALPHA and -DKAPPA
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: QS8/QS16/F16/F32
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

/** Apply in-map normalization.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size, e.g. -DVEC_SIZE=16
 * @note The radius should be given as a preprocessor argument using -DRADIUS=size. e.g. -DRADIUS=5
 * @note In case of fixed-point operation -DFIXED_POINT_POSITION=fixed_point_position must be provided: e.g. -DFIXED_POINT_POSITION=3
 * @note Scaling coefficient (= alpha/norm_size), beta and kappa need to be passed at compile time using -DCOEFF, -DALPHA and -DKAPPA
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: QS8/F16/F32
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
__kernel void normalization_layer_in_map(TENSOR3D_DECLARATION(input),
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
    const int right_pos   = min((int)RADIUS, (int)((get_global_size(0) << 2) + 3 - 1 - current_col));

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
