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

#if defined(SATURATE)
#define CONVERT_OP_INT_STR(x, type, size) (convert_##type##size##_sat(x))
#else // SATURATE
#define CONVERT_OP_INT_STR(x, type, size) (convert_##type##size(x))
#endif // SATURATE
#define CONVERT_OP_INT(x, type, size) CONVERT_OP_INT_STR(x, type, size)

#define MUL_OP(x, y, scale, type, size) CONVERT_OP_INT((x) * (y) >> scale, type, size)

#define CONVERT_RTE(x, type) (convert_##type##_rte((x)))
#define CONVERT_DOWN(x, type) CONVERT_RTE(x, type)

#if defined(DATA_TYPE_IN1) && defined(DATA_TYPE_IN2) && defined(ACC_DATA_TYPE) && defined(DATA_TYPE_OUT)

#define VEC_ACC_TYPE VEC_DATA_TYPE(ACC_DATA_TYPE, VEC_SIZE_OUT)
#define VEC_OUT_TYPE VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT)

/** Performs a pixelwise multiplication with integer scale of integer inputs.
 *
 * @attention The inputs and output data types need to be passed at compile time using -DDATA_TYPE_IN1, -DDATA_TYPE_IN2 and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN1=uchar -DDATA_TYPE_IN2=ushort -DDATA_TYPE_OUT=short
 * @attention The data_type of the intermediate result of the multiplication should passed as well using -DACC_DATA_TYPE.
 * e.g. If one of inputs is S16 -DACC_DATA_TYPE=int should be passed else -DACC_DATA_TYPE=short.
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: U8/S16
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_z                        in1_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: same as @p in1_ptr
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_z                        in2_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: same as @p in1_ptr
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  scale                             Integer scaling factor. Supported data types: S32.
 */
__kernel void pixelwise_mul_int(
    TENSOR3D_DECLARATION(in1),
    TENSOR3D_DECLARATION(in2),
    TENSOR3D_DECLARATION(out),
    const uint scale)
{
    size_t x = max((int)(get_global_id(0) * VEC_SIZE_OUT - (VEC_SIZE_OUT - VEC_SIZE_LEFTOVER) % VEC_SIZE_OUT), 0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);

    __global uchar *in1_addr = in1_ptr + in1_offset_first_element_in_bytes + x * in1_stride_x + y * in1_stride_y + z * in1_stride_z;
    __global uchar *in2_addr = in2_ptr + in2_offset_first_element_in_bytes + x * in2_stride_x + y * in2_stride_y + z * in2_stride_z;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + x * out_stride_x + y * out_stride_y + z * out_stride_z;

    // Load data
    VEC_ACC_TYPE in1_data = CONVERT((VEC_DATA_TYPE(DATA_TYPE_IN1, VEC_SIZE_OUT))VLOAD(VEC_SIZE_IN1)(0, (__global DATA_TYPE_IN1 *)in1_addr), VEC_ACC_TYPE);
    VEC_ACC_TYPE in2_data = CONVERT((VEC_DATA_TYPE(DATA_TYPE_IN2, VEC_SIZE_OUT))VLOAD(VEC_SIZE_IN2)(0, (__global DATA_TYPE_IN2 *)in2_addr), VEC_ACC_TYPE);

    // Perform multiplication and store result
    VEC_OUT_TYPE out_data0 = MUL_OP(in1_data, in2_data, scale, DATA_TYPE_OUT, VEC_SIZE_OUT);
    STORE_VECTOR_SELECT(out_data, DATA_TYPE_OUT, out_addr, VEC_SIZE_OUT, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif /* defined(DATA_TYPE_IN1) && defined(DATA_TYPE_IN2) && defined(ACC_DATA_TYPE) && defined(DATA_TYPE_OUT) */

#if defined(SCALE_IN1) && defined(SCALE_IN2) && defined(SCALE_OUT) && defined(DATA_TYPE_OUT) && defined(VEC_SIZE_OUT)

#define VEC_FLOAT VEC_DATA_TYPE(float, VEC_SIZE_OUT)
#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE_OUT)
#define VEC_TYPE VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE_OUT)

/** Performs a pixelwise multiplication with float scale of quantized inputs.
 *
 * @note The quantization offset of the first operand must be passed at compile time only if asymmetric using -DOFFSET_IN1, e.g. -DOFFSET_IN1=10
 * @note The quantization offset of the second operand must be passed at compile time only if asymmetric using -DOFFSET_IN2, e.g. -DOFFSET_IN2=10
 * @note The quantization offset of the output must be passed at compile time only if asymmetric using -DOFFSET_OUT, e.g. -DOFFSET_OUT=10
 * @note The quantization scale of the first operand must be passed at compile time using -DSCALE_IN1, e.g. -DSCALE_IN1=10
 * @note The quantization scale of the second operand must be passed at compile time using -DSCALE_IN2, e.g. -DSCALE_IN2=10
 * @note The quantization scale of the output must be passed at compile time using -DSCALE_OUT, e.g. -DSCALE_OUT=10
 * @note To perform saturating operation -DSATURATE has to be passed to the compiler otherwise wrapping policy will be used.
 * @attention The data type must be passed at compile time using -DDATA_TYPE_OUT, i.e. -DDATA_TYPE_OUT=uchar
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: QASYMM8/QASYMM8_SIGNED/QSYMM16
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_z                        in1_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: same as @p in1_ptr
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_stride_z                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_z                        in2_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: same as @p in1_ptr
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  scale                             Float scaling factor. Supported data types: F32
 */
__kernel void pixelwise_mul_quantized(
    TENSOR3D_DECLARATION(in1),
    TENSOR3D_DECLARATION(in2),
    TENSOR3D_DECLARATION(out),
    const float scale)
{
    size_t x = max((int)(get_global_id(0) * VEC_SIZE_OUT - (VEC_SIZE_OUT - VEC_SIZE_LEFTOVER) % VEC_SIZE_OUT), 0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);

    __global uchar *in1_addr = in1_ptr + in1_offset_first_element_in_bytes + x * in1_stride_x + y * in1_stride_y + z * in1_stride_z;
    __global uchar *in2_addr = in2_ptr + in2_offset_first_element_in_bytes + x * in2_stride_x + y * in2_stride_y + z * in2_stride_z;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + x * out_stride_x + y * out_stride_y + z * out_stride_z;

    // Load data
    VEC_INT in_a = CONVERT((VEC_TYPE)(VLOAD(VEC_SIZE_OUT)(0, (__global DATA_TYPE_OUT *)in1_addr)), VEC_INT);
    VEC_INT in_b = CONVERT((VEC_TYPE)(VLOAD(VEC_SIZE_OUT)(0, (__global DATA_TYPE_OUT *)in2_addr)), VEC_INT);

    // Dequantize
#if defined(OFFSET_IN1)
    in_a -= (VEC_INT)((int)OFFSET_IN1);
#endif // defined(OFFSET_IN1)
#if defined(OFFSET_IN2)
    in_b -= (VEC_INT)((int)OFFSET_IN2);
#endif // defined(OFFSET_IN2)
    const VEC_FLOAT in1f32 = CONVERT(in_a, VEC_FLOAT) * (VEC_FLOAT)((float)SCALE_IN1);
    const VEC_FLOAT in2f32 = CONVERT(in_b, VEC_FLOAT) * (VEC_FLOAT)((float)SCALE_IN2);

#if defined(OFFSET_OUT)
    const VEC_FLOAT qresf32 = (in1f32 * in2f32 * scale) / ((VEC_FLOAT)(float)SCALE_OUT) + ((VEC_FLOAT)((float)OFFSET_OUT));
#else  // defined(OFFSET_OUT)
    const VEC_FLOAT qresf32 = (in1f32 * in2f32 * scale) / ((VEC_FLOAT)(float)SCALE_OUT);
#endif // defined(OFFSET_OUT)
    const VEC_TYPE res0 = CONVERT_SAT(CONVERT_DOWN(qresf32, VEC_INT), VEC_TYPE);

    // Store result
    STORE_VECTOR_SELECT(res, DATA_TYPE_OUT, out_addr, VEC_SIZE_OUT, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif /* defined(SCALE_IN1) && defined(SCALE_IN2) && defined(SCALE_OUT) && defined(DATA_TYPE_OUT) && defined(VEC_SIZE_OUT) */
