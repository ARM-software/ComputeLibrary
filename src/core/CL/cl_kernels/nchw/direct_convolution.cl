/*
 * Copyright (c) 2021 Arm Limited.
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
#include "helpers_asymm.h"

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data size must be passed at compile time using -DDATA_SIZE e.g. -DDATA_SIZE=32
 * @note The convolution stride x must be passed at compile time using -DSTRIDE_X e.g. -DSTRIDE_X=1
 * @note The third dimensions of the weights tensors must be passed at compile time using -DWEIGHTS_DEPTH
 * @note In case biases will be added to the convolution -DHAS_BIAS has to be passed to append the final matrix with 1 in each row.
 * @note The output quantization multiplier must be passed at compile time using -DOUTPUT_MULTIPLIER e.g. -DOUTPUT_MULTIPLIER=1234
 * @note The output quantization shift must be passed at compile time using -DOUTPUT_SHIFT e.g. -DOUTPUT_SHIFT=4
 * @note The input offset quantization parameter must be passed at compile time using -DINPUT_OFFSET e.g. -DINPUT_OFFSET=3
 * @note The weights offset quantization parameter must be passed at compile time using -DWEIGHTS_OFFSET e.g. -DWEIGHTS_OFFSET=3
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                            src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[out] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                            dst_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                            dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in]  weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in]  weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  weights_step_y                        weights_stride_y * number of elements along y processed per workitem(in bytes)
 * @param[in]  weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  weights_step_z                        weights_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in]  biases_ptr                            Pointer to the biases tensor. Same as @p src_ptr
 * @param[in]  biases_stride_x                       Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                         biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes  The offset of the first element in the biases tensor
 * @param[in]  weights_stride_w                      Stride of the weights tensor in the 4th dimension
 */
__kernel void direct_convolution_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#ifdef HAS_BIAS
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    unsigned int weights_stride_w)
{
    const int id0 = get_global_id(0);
    const int id1 = get_global_id(1);
    const int id2 = get_global_id(2);

    const int x_coords = (id0 * STRIDE_X) - PAD_LEFT;
    const int y_coords = (id1 * STRIDE_Y) - PAD_TOP;

    const int x_offs = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0) * sizeof(DATA_TYPE);

    __global uchar *src_addr     = (__global uchar *)(src_ptr + src_offset_first_element_in_bytes);
    __global uchar *weights_addr = (__global uchar *)(weights_ptr + weights_offset_first_element_in_bytes + id2 * weights_stride_w);
    __global uchar *dst_addr     = (__global uchar *)dst_ptr + dst_offset_first_element_in_bytes + x_offs + id1 * dst_stride_y + id2 * dst_stride_z;

#ifdef IS_QUANTIZED
    int acc_value = 0;
#else  /* IS_QUANTIZED */
    DATA_TYPE                 acc_value = 0;
#endif /* IS_QUANTIZED */
    for(volatile int d = 0; d < WEIGHTS_DEPTH; ++d)
    {
        for(int y = 0; y < WEI_HEIGHT; ++y)
        {
            for(int x = 0; x < WEI_WIDTH; ++x)
            {
                const int idx_x = (x_coords + x);
                const int idx_y = (y_coords + y);
                if((idx_x >= 0 && idx_x < SRC_WIDTH) && (idx_y >= 0 && idx_y < SRC_HEIGHT))
                {
                    const int weight_offset = x + (WEI_HEIGHT * y);
                    const int input_offset  = idx_x + SRC_WIDTH * idx_y;
#ifdef IS_QUANTIZED
                    int weight = convert_int(*((__global DATA_TYPE *)weights_addr + weight_offset));
                    int input  = convert_int(*((__global DATA_TYPE *)src_addr + input_offset));
                    acc_value += (input + INPUT_OFFSET) * (weight + WEIGHTS_OFFSET);
#else  /* IS_QUANTIZED */
                    DATA_TYPE weight    = *((__global DATA_TYPE *)weights_addr + weight_offset);
                    DATA_TYPE input     = *((__global DATA_TYPE *)src_addr + input_offset);
                    acc_value += input * weight;
#endif /* IS_QUANTIZED */
                }
            }
        }
        src_addr += src_stride_z;
        weights_addr += weights_stride_z;
    }

#ifdef HAS_BIAS

    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#ifdef IS_QUANTIZED
    int bias = *((__global int *)(vector_offset(&biases, id2)));
#else  /* IS_QUANTIZED */
    DATA_TYPE bias = *((__global DATA_TYPE *)(vector_offset(&biases, id2)));
#endif /* IS_QUANTIZED */
    acc_value += bias;

#endif /* defined(HAS_BIAS) */

#ifdef IS_QUANTIZED

#if OUTPUT_SHIFT < 0
    acc_value = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(acc_value, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 1);
#else  // OUTPUT_SHIFT < 0
    acc_value      = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(acc_value, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 1);
#endif // OUTPUT_SHIFT < 0
    acc_value = acc_value + OUTPUT_OFFSET;
#endif /* IS_QUANTIZED */

    *(__global DATA_TYPE *)dst_addr = CONVERT_SAT(acc_value, DATA_TYPE);
}