/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#if defined(VEC_SIZE) && defined(DATA_TYPE_SRC) && defined(DATA_TYPE_DST) && defined(SCALE) && defined(OFFSET)

/** This performs the dequantization of 8-bit unsigned integers to floating point.
 *
 * @note Source datatype should be given as a preprocessor argument using -DDATA_TYPE_SRC=type. e.g. -DDATA_TYPE_SRC=char
 * @note Destination datatype should be given as a preprocessor argument using -DDATA_TYPE_DST=type. e.g. -DDATA_TYPE_DST=float
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Quantization scale of input tensor is passed in with -DSCALE=scale.
 * @note Quantization offset of input tensor is passed in with -DOFFSET=offset.
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/QSYMM8
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: F16/F32
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void dequantization_layer(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

#if defined(LAST_ACCESSED_X)
    // Check if access on width gets out of bounds
    // If it does shift access vector to access elements within bounds
    const int xi = (int)(get_global_id(0) * VEC_SIZE);
    input.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * input_stride_x;
    output.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * output_stride_x;

    // Load data
    VEC_DATA_TYPE(int, VEC_SIZE)
    val = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_SRC *)input.ptr), VEC_DATA_TYPE(int, VEC_SIZE));

    // Create scale and offset vectors
    const VEC_DATA_TYPE(float, VEC_SIZE)
    vscale = SCALE;

    const VEC_DATA_TYPE(int, VEC_SIZE)
    voffset = OFFSET;

    // Dequantize
    VEC_DATA_TYPE(float, VEC_SIZE)
    res = vscale * CONVERT((val - voffset), VEC_DATA_TYPE(float, VEC_SIZE));

    // Store result
    VSTORE(VEC_SIZE)
    (CONVERT(res, VEC_DATA_TYPE(DATA_TYPE_DST, VEC_SIZE)), 0, (__global DATA_TYPE_DST *)output.ptr);
#else  // !defined(LAST_ACCESSED_X)
    *((__global DATA_TYPE_DST *)(output.ptr)) = (DATA_TYPE_DST)((float)((int)(*((__global DATA_TYPE_SRC *)(input.ptr))) - (int)(OFFSET)) * (float)(SCALE));
#endif // defined(LAST_ACCESSED_X)
}

#endif // defined(VEC_SIZE) && defined(DATA_TYPE_SRC) && defined(DATA_TYPE_DST) && defined(SCALE) && defined(OFFSET)
