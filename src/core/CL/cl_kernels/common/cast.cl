/*
 * Copyright (c) 2016-2021 Arm Limited.
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

#ifdef SATURATE
#define CONVERT_DOWN(x, type) CONVERT_SAT(x, type)
#else /* SATURATE */
#define CONVERT_DOWN(x, type) CONVERT(x, type)
#endif /* SATURATE */

#define CONVERT_UP(x, type) CONVERT(x, type)

/** This function performs a down-casting
 *
 * @attention For QSYMM8_PER_CHANNEL -> QASYMM8, it is user's responsibility to keep track of the quantization info.
 *
 * @note The input and output data_types need to be passed at compile time using -DDATA_TYPE_IN and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN=uchar -DDATA_TYPE_OUT=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8/S8/QSYMM8_PER_CHANNEL/U16/S16/U32/S32/F16/F32
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         in_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  in_step_z                         in_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void cast_down(
    TENSOR3D_DECLARATION(in),
    TENSOR3D_DECLARATION(out))
{
    int x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);

    __global uchar *in_addr  = in_ptr + in_offset_first_element_in_bytes + sizeof(DATA_TYPE_IN) * x_offs + get_global_id(1) * in_stride_y + get_global_id(2) * in_stride_z;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + sizeof(DATA_TYPE_OUT) * x_offs + get_global_id(1) * out_stride_y + get_global_id(2) * out_stride_z;

    // Load data
    VEC_DATA_TYPE(DATA_TYPE_IN, VEC_SIZE)
    in_data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN *)in_addr);

#if defined(IS_DATA_TYPE_QUANTIZED)
    in_data ^= (VEC_DATA_TYPE(DATA_TYPE_IN, VEC_SIZE))0x80;
#endif // defined(IS_DATA_TYPE_QUANTIZED)

#if defined(IS_DATA_TYPE_FLOAT)
    VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)
    res0 = CONVERT_DOWN(in_data, VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE));
    STORE_VECTOR_SELECT(res, DATA_TYPE_OUT, out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#else  /* defined(IS_DATA_TYPE_FLOAT) */
    VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)
    res0 = CONVERT_DOWN(in_data, VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE));
    STORE_VECTOR_SELECT(res, DATA_TYPE_OUT, out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#endif /* defined(IS_DATA_TYPE_FLOAT) */
}

/** This function performs a up-casting
 *
 * @note The input and output data_types need to be passed at compile time using -DDATA_TYPE_IN and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN=uchar -DDATA_TYPE_OUT=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8/S8/U16/S16/U32/S32/F16/F32
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         in_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  in_step_z                         in_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8/U16/S16/U32/S32/F16/F32
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void cast_up(
    TENSOR3D_DECLARATION(in),
    TENSOR3D_DECLARATION(out))
{
    int x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);

    __global uchar *in_addr  = in_ptr + in_offset_first_element_in_bytes + sizeof(DATA_TYPE_IN) * x_offs + get_global_id(1) * in_stride_y + get_global_id(2) * in_stride_z;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + sizeof(DATA_TYPE_OUT) * x_offs + get_global_id(1) * out_stride_y + get_global_id(2) * out_stride_z;

    // Load data
    VEC_DATA_TYPE(DATA_TYPE_IN, VEC_SIZE)
    in_data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN *)in_addr);

#if defined(IS_DATA_TYPE_FLOAT)
    VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)
    res0 = CONVERT_UP(in_data, VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE));
    STORE_VECTOR_SELECT(res, DATA_TYPE_OUT, out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#else  /* defined(IS_DATA_TYPE_FLOAT) */
    VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)
    res0 = CONVERT_UP(in_data, VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE));
    STORE_VECTOR_SELECT(res, DATA_TYPE_OUT, out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#endif /* defined(IS_DATA_TYPE_FLOAT) */
}
