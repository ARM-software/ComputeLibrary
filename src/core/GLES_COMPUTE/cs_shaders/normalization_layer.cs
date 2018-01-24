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

layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;

#include "helpers_cs.h"

/** Apply cross map normalization and in map normalization
 *
 * @note Alpha parameter / norm_size should be given as a preprocessor argument using "#define COEFF x"
 * @note BETA parameter in the normalization equation should be given as a preprocessor argument using "#define BETA x"
 * @note KAPPA parameter in the normalization equation should be given as a preprocessor argument using "#define KAPPA x"
 * @note Number of elements on the right or left side to normalize across should be given as a preprocessor argument using "#define RADIUS x"
 *
 * @param[in]  src1_ptr   Pointer to the first source tensor. Supported data types: F32
 * @param[in]  src1_attrs The attributes of the first source tensor
 * @param[in]  src2_ptr   Pointer to the second source tensor. Supported data types: Same as @p src1_ptr
 * @param[in]  src2_attrs The attributes of the second source tensor
 * @param[out] dst_ptr    Pointer to the destination tensor. Supported data types: Same as @p src1_ptr
 * @param[in]  dst_attrs  The attributes of the destination tensor
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src1_attrs;
    Tensor3DAttributes src2_attrs;
    Tensor3DAttributes dst_attrs;
};
TENSOR_DECLARATION(1, src1Buffer, float, src1_ptr, src1_shift, 2, readonly);
TENSOR_DECLARATION(2, src2Buffer, float, src2_ptr, src2_shift, 2, readonly);
TENSOR_DECLARATION(3, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

#ifdef CROSS_MAP
void main(void)
{
    Tensor3DIterator src1_iter = CONVERT_TO_TENSOR3D_ITERATOR(src1_attrs, src1_shift);
    Tensor3DIterator src2_iter = CONVERT_TO_TENSOR3D_ITERATOR(src2_attrs, src2_shift);
    Tensor3DIterator dst_iter  = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    float acc = 0.0;

    int num_of_slices = int(gl_NumWorkGroups.z * gl_WorkGroupSize.z);
    int current_slice = int(gl_GlobalInvocationID.z);

    int left_slice  = max(current_slice - int(RADIUS), int(0));
    int right_slice = min(current_slice + int(RADIUS), int(num_of_slices - 1));

    for(int i = left_slice; i <= right_slice; i++)
    {
        acc += LOAD(src2_ptr, TENSOR3D_OFFSET(src2_iter, 0, 0, i - current_slice));
    }

    float normalized = pow(float(KAPPA) + float(COEFF) * acc, float(BETA));

    float normalized_pixel = (LOAD_CURRENT_ITEM(src1_ptr, src1_iter)) / normalized;

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, normalized_pixel);
}

#elif defined(IN_MAP_1D)
void main(void)
{
    Tensor3DIterator src1_iter = CONVERT_TO_TENSOR3D_ITERATOR(src1_attrs, src1_shift);
    Tensor3DIterator src2_iter = CONVERT_TO_TENSOR3D_ITERATOR(src2_attrs, src2_shift);
    Tensor3DIterator dst_iter  = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    float acc = 0.0;

    int num_of_items_x = int(gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    int current_pos    = int(gl_GlobalInvocationID.x);

    int left_pos  = max(current_pos - int(RADIUS), int(0));
    int right_pos = min(current_pos + int(RADIUS), int(num_of_items_x + -1));

    for(int i = left_pos; i <= right_pos; i++)
    {
        acc += LOAD(src2_ptr, TENSOR3D_OFFSET(src2_iter, i - current_pos, 0, 0));
    }

    float normalized = pow(float(KAPPA) + float(COEFF) * acc, float(BETA));

    float normalized_pixel = (LOAD_CURRENT_ITEM(src1_ptr, src1_iter)) / normalized;

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, normalized_pixel);
}
#endif /*CROSS_MAP*/
