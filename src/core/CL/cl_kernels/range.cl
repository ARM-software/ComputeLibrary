/*
 * Copyright (c) 2018-2021 Arm Limited.
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

#if defined(VECTOR_SIZE) && defined(START) && defined(STEP) && defined(DATA_TYPE) && defined(VEC_SIZE_LEFTOVER)

#if !defined(OFFSET_OUT) && !defined(SCALE_OUT)

#if VECTOR_SIZE == 2
#define STEP_VEC ((VEC_DATA_TYPE(DATA_TYPE, 2))(0, STEP))
#elif VECTOR_SIZE == 3
#define STEP_VEC ((VEC_DATA_TYPE(DATA_TYPE, 3))(0, STEP, 2 * STEP))
#elif VECTOR_SIZE == 4
#define STEP_VEC ((VEC_DATA_TYPE(DATA_TYPE, 4))(0, STEP, 2 * STEP, 3 * STEP))
#elif VECTOR_SIZE == 8
#define STEP_VEC ((VEC_DATA_TYPE(DATA_TYPE, 8))(0, STEP, 2 * STEP, 3 * STEP, 4 * STEP, 5 * STEP, 6 * STEP, 7 * STEP))
#elif VECTOR_SIZE == 16
#define STEP_VEC ((VEC_DATA_TYPE(DATA_TYPE, 16))(0, STEP, 2 * STEP, 3 * STEP, 4 * STEP, 5 * STEP, 6 * STEP, 7 * STEP, 8 * STEP, 9 * STEP, 10 * STEP, 11 * STEP, 12 * STEP, 13 * STEP, 14 * STEP, 15 * STEP))
#endif // VECTOR_SIZE == 2

/** Generates a sequence of numbers starting from START and extends by increments of 'STEP' up to but not including 'END'.
 *
 * @note starting value of the sequence must be given as a preprocessor argument using -DSTART=value. e.g. -DSTART=0
 * @note difference between consequtive elements of the sequence must be given as a preprocessor argument using -DSTEP=value. e.g. -DSTEP=1
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note vector size supported by the device must be given as a preprocessor argument using -DVECTOR_SIZE=value. e.g. -DDATA_TYPE=4
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVECTOR_SIZE=3. It is defined as the remainder between the input's first dimension and VECTOR_SIZE
 *
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: U8/S8/U16/S16/U32/S32/F16/F32.
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  out_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void range(
    VECTOR_DECLARATION(out))
{
    uint     id             = max((int)(get_global_id(0) * VECTOR_SIZE - (VECTOR_SIZE - VEC_SIZE_LEFTOVER) % VECTOR_SIZE), 0);
    __global uchar *dst_ptr = out_ptr + out_offset_first_element_in_bytes + id * sizeof(DATA_TYPE);
#if VECTOR_SIZE == 1
    DATA_TYPE seq;
    seq = (DATA_TYPE)START + (DATA_TYPE)id * (DATA_TYPE)STEP;

    *dst_ptr = seq;
#else  // VECTOR_SIZE == 1
    VEC_DATA_TYPE(DATA_TYPE, VECTOR_SIZE)
    seq0 = ((DATA_TYPE)START + (DATA_TYPE)id * (DATA_TYPE)STEP);
    seq0 = seq0 + STEP_VEC;
    STORE_VECTOR_SELECT(seq, DATA_TYPE, dst_ptr, VECTOR_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#endif //VECTOR_SIZE == 1
}

#else // !defined(OFFSET_OUT) && !defined(SCALE_OUT)

#if VECTOR_SIZE == 2
#define STEP_VEC ((VEC_DATA_TYPE(float, 2))(0, STEP))
#elif VECTOR_SIZE == 3
#define STEP_VEC ((VEC_DATA_TYPE(float, 3))(0, STEP, 2 * STEP))
#elif VECTOR_SIZE == 4
#define STEP_VEC ((VEC_DATA_TYPE(float, 4))(0, STEP, 2 * STEP, 3 * STEP))
#elif VECTOR_SIZE == 8
#define STEP_VEC ((VEC_DATA_TYPE(float, 8))(0, STEP, 2 * STEP, 3 * STEP, 4 * STEP, 5 * STEP, 6 * STEP, 7 * STEP))
#elif VECTOR_SIZE == 16
#define STEP_VEC ((VEC_DATA_TYPE(float, 16))(0, STEP, 2 * STEP, 3 * STEP, 4 * STEP, 5 * STEP, 6 * STEP, 7 * STEP, 8 * STEP, 9 * STEP, 10 * STEP, 11 * STEP, 12 * STEP, 13 * STEP, 14 * STEP, 15 * STEP))
#endif // VECTOR_SIZE == 2

#define CONVERT_RTE(x, type) (convert_##type##_rte((x)))
#define CONVERT_DOWN(x, type) CONVERT_RTE(x, type)

/** Generates a sequence of numbers starting from START and extends by increments of 'STEP' up to but not including 'END'.
 *
 * @note starting value of the sequence must be given as a preprocessor argument using -DSTART=value. e.g. -DSTART=0
 * @note difference between consequtive elements of the sequence must be given as a preprocessor argument using -DSTEP=value. e.g. -DSTEP=1
 * @note Datatype must be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note vector size supported by the device must be given as a preprocessor argument using -DVECTOR_SIZE=vector_size. e.g. -DDATA_TYPE=4
 * @note The quantization offset of the output must be passed at compile time using -DOFFSET_OUT, i.e. -DOFFSET_OUT=10
 * @note The quantization scale of the output must be passed at compile time using -DSCALE_OUT, i.e. -DSCALE_OUT=10
 *
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: QASYMM8.
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  out_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void range_quantized(
    VECTOR_DECLARATION(out))
{
    uint     id             = max((int)(get_global_id(0) * VECTOR_SIZE - (VECTOR_SIZE - VEC_SIZE_LEFTOVER) % VECTOR_SIZE), 0);
    __global uchar *dst_ptr = out_ptr + out_offset_first_element_in_bytes + id * sizeof(DATA_TYPE);
#if VECTOR_SIZE == 1
    float           seq;
    seq      = (float)START + (float)id * (float)STEP;
    seq      = (DATA_TYPE)(int)(seq / ((float)SCALE_OUT) + (float)OFFSET_OUT);
    seq      = max(0.0f, min(seq, 255.0f));
    *dst_ptr = CONVERT_SAT(CONVERT_DOWN(seq, int), uchar);
#else  // VECTOR_SIZE == 1
    VEC_DATA_TYPE(float, VECTOR_SIZE)
    seq = (float)START + id * (float)STEP;
    seq = seq + STEP_VEC;
    seq = seq / ((VEC_DATA_TYPE(float, VECTOR_SIZE))((float)SCALE_OUT)) + ((VEC_DATA_TYPE(float, VECTOR_SIZE))((float)OFFSET_OUT));
    seq = max((VEC_DATA_TYPE(float, VECTOR_SIZE))(0.0f), min(seq, (VEC_DATA_TYPE(float, VECTOR_SIZE))(255.0f)));
    VEC_DATA_TYPE(uchar, VECTOR_SIZE)
    res0 = CONVERT_SAT(CONVERT_DOWN(seq, VEC_DATA_TYPE(int, VECTOR_SIZE)), VEC_DATA_TYPE(uchar, VECTOR_SIZE));
    STORE_VECTOR_SELECT(res, DATA_TYPE, dst_ptr, VECTOR_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
#endif // VECTOR_SIZE == 1
}
#endif // !defined(OFFSET_OUT) && !defined(SCALE_OUT)

#endif // defined(VECTOR_SIZE) && defined(START) && defined(STEP) && defined(DATA_TYPE) && defined(VEC_SIZE_LEFTOVER)
