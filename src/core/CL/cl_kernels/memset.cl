/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#if defined(DATA_TYPE) && defined(CONSTANT_VALUE) // Check for compile time constants

/** Fill the tensor's planes with all value
 * @attention The following variables must be passed at compile time:
 * -# -DDATA_TYPE = Tensor data type. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * -# -DCONSTANT_VALUE = The value use to fill the tensor's planes
 * -# -DVEC_SIZE = Vector size
 * -# -DLAST_ACCESSED_X = The element that is on the X border (threads trying to set this, might need to step back a bit)
 *
 * @param[in] tensor_ptr                           Pointer to the source image. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
 * @param[in] tensor_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] tensor_step_x                        tensor_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] tensor_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] tensor_step_y                        tensor_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] tensor_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] value                                The value used to fill the pages of the tensor
 */
__kernel void memset(
    TENSOR3D_DECLARATION(tensor))
{
    Tensor3D tensor = CONVERT_TO_TENSOR3D_STRUCT(tensor);

#if defined(VEC_SIZE)

#if defined(LAST_ACCESSED_X)
    // Check if access on width gets out of bounds
    // If it does shift access vector to access elements within bounds
    const int xi = (int)(get_global_id(0) * VEC_SIZE);
    tensor.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * tensor_stride_x;
#endif // defined(LAST_ACCESSED_X)

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    data = (DATA_TYPE)(CONSTANT_VALUE);

    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)tensor.ptr);
#else  // !defined(VEC_SIZE)
    *((__global DATA_TYPE *)(tensor.ptr)) = (DATA_TYPE)(CONSTANT_VALUE);
#endif // defined(VEC_SIZE)
}

#endif // Check for compile time constants
