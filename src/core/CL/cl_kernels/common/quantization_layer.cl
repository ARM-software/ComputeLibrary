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

#define CONVERT_RTE(x, type) (convert_##type##_rte((x)))
#define CONVERT_RTE_VEC_STR(x, type, size) (convert_##type##size##_rte((x)))
#define CONVERT_RTE_VEC(x, type, size) CONVERT_RTE_VEC_STR(x, type, size)

#if defined(VEC_SIZE) && defined(DATA_TYPE_IN) && defined(DATA_TYPE_OUT) && defined(SCALE) && defined(OFFSET) && defined(MIN_QUANT_VAL) && defined(MAX_QUANT_VAL)

/** This performs the quantization of floating point inputs or 8-bit quantized integers to 8-bit integers.
 *
 * @note Input data type should be given as a preprocessor argument using -DDATA_TYPE_IN=type. e.g. -DDATA_TYPE=short
 * @note Output data type should be given as a preprocessor argument using -DDATA_TYPE_OUT=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Quantization scale should be given as a preprocessor argument using -DSCALE=scale. e.g. -DSCALE=0.125
 * @note Quantization offset should be given as a preprocessor argument using -DOFFSET=offset. e.g. -DOFFSET=125
 * @note Minimum value for quantized type should be given as a preprocessor argument using -DMIN_QUANT_VAL=value. e.g. -DMIN_QUANT_VAL=0
 * @note Maximum value for quantized type should be given as a preprocessor argument using -DMAX_QUANT_VAL=value. e.g. -DMAXIN_QUANT_VAL=255
 * @note If the input data type if a floating point (F16 or F32) the preprocessor argument should be give as -DIS_FLOAT
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: QASYMM8/QASYMM8_SIGNED
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void quantization_layer(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output))
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

#if defined(VEC_SIZE) && defined(LAST_ACCESSED_X)
    // Check if access on width gets out of bounds
    // If it does shift access vector to access elements within bounds
    const int xi = (int)(get_global_id(0) * VEC_SIZE);
    input.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * input_stride_x;
    output.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * output_stride_x;

    // Load data
#if defined(IS_FLOAT)
    // Load data
    VEC_DATA_TYPE(DATA_TYPE_IN, VEC_SIZE)
    val_float = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN *)input.ptr);

    // Create scale and offset vectors
    const VEC_DATA_TYPE(DATA_TYPE_IN, VEC_SIZE) vscale = SCALE;
    const VEC_DATA_TYPE(int, VEC_SIZE) voffset         = OFFSET;
#else  // defined(IS_FLOAT)
    // Load data
    VEC_DATA_TYPE(DATA_TYPE_IN, VEC_SIZE)
    val = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN *)input.ptr);

    const VEC_DATA_TYPE(float, VEC_SIZE)
    val_float = CONVERT(val, VEC_DATA_TYPE(float, VEC_SIZE));

    // Create scale and offset vectors
    const VEC_DATA_TYPE(float, VEC_SIZE) vscale = SCALE;
    const VEC_DATA_TYPE(int, VEC_SIZE) voffset  = OFFSET;
#endif // defined(IS_FLOAT)

    // Quantize
    VEC_DATA_TYPE(int, VEC_SIZE)
    res = CLAMP(CONVERT_RTE_VEC(val_float / vscale, int, VEC_SIZE) + voffset, MIN_QUANT_VAL, MAX_QUANT_VAL);

    // Store result
    VSTORE(VEC_SIZE)
    (CONVERT(res, VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)), 0, (__global DATA_TYPE_OUT *)output.ptr);
#else  //!defined(VEC_SIZE) || !defined(LAST_ACCESSED_X)
    *((__global DATA_TYPE_OUT *)(output.ptr)) = (DATA_TYPE_OUT)CLAMP(CONVERT_RTE(((float) * (__global DATA_TYPE_IN *)input.ptr) / ((float)SCALE), int) + (int)OFFSET, MIN_QUANT_VAL, MAX_QUANT_VAL);
#endif // defined(VEC_SIZE) && defined(LAST_ACCESSED_X)
}
#endif // defined(VEC_SIZE) && defined(DATA_TYPE_IN) && defined(DATA_TYPE_OUT) && defined(SCALE) && defined(OFFSET) && defined(MIN_QUANT_VAL) && defined(MAX_QUANT_VAL)
