/*
 * Copyright (c) 2016, 2017 ARM Limited.
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

/** This performs an activation function floating point inputs.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Activation function should be given as a preprocessor argument using -DNAME. e.g. -DTANH
 * @note Distinction between floating point and integer is done using -DTYPE_FP and -DTYPE_INT preprocessor argument
 * @note A, B variables required by some activation functions are set using -DA= and -DB= respectively.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: F16, F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: F16, F32
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void activation_layer(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    // Get pixels pointer
    Image input  = CONVERT_TO_IMAGE_STRUCT(input);
    Image output = CONVERT_TO_IMAGE_STRUCT(output);

    // Load data
    VEC_DATA_TYPE(DATA_TYPE, 16)
    data = vload16(0, (__global DATA_TYPE *)input.ptr);

    // Perform activation
#if defined LOGISTIC
    data = 1 / (1 + exp(-data));
#elif defined TANH
    data = (VEC_DATA_TYPE(DATA_TYPE, 16))A * tanh((VEC_DATA_TYPE(DATA_TYPE, 16))B * data);
#elif defined RELU
    data = max(0, data);
#elif defined BRELU
    data = min((VEC_DATA_TYPE(DATA_TYPE, 16))A, max(0, data));
#elif defined SRELU
    data = log(1 + exp(data));
#elif defined ABS
#if defined   TYPE_INT
    data = abs(data);
#else
    data = fabs(data);
#endif
#elif defined SQUARE
    data = data * data;
#elif defined SQRT
    data = sqrt(data);
#elif defined LINEAR
    data = (VEC_DATA_TYPE(DATA_TYPE, 16))A * data + (VEC_DATA_TYPE(DATA_TYPE, 16))B;
#endif

    // Store result
    vstore16(data, 0, (__global DATA_TYPE *)output.ptr);
}
