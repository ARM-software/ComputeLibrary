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

/** Calculates L1 normalization between two inputs.
 *
 * @param[in] a First input. Supported data types: S16, S32
 * @param[in] b Second input. Supported data types: S16, S32
 *
 * @return L1 normalization magnitude result. Supported data types: S16, S32
 */
inline VEC_DATA_TYPE(DATA_TYPE, 16) magnitude_l1(VEC_DATA_TYPE(DATA_TYPE, 16) a, VEC_DATA_TYPE(DATA_TYPE, 16) b)
{
    return CONVERT_SAT(add_sat(abs(a), abs(b)), VEC_DATA_TYPE(DATA_TYPE, 16));
}

/** Calculates L2 normalization between two inputs.
 *
 * @param[in] a First input. Supported data types: S16, S32
 * @param[in] b Second input. Supported data types: S16, S32
 *
 * @return L2 normalization magnitude result. Supported data types: S16, S32
 */
inline VEC_DATA_TYPE(DATA_TYPE, 16) magnitude_l2(int16 a, int16 b)
{
    return CONVERT_SAT((sqrt(convert_float16((convert_uint16(a * a) + convert_uint16(b * b)))) + 0.5f),
                       VEC_DATA_TYPE(DATA_TYPE, 16));
}

/** Calculates unsigned phase between two inputs.
 *
 * @param[in] a First input. Supported data types: S16, S32
 * @param[in] b Second input. Supported data types: S16, S32
 *
 * @return Unsigned phase mapped in the interval [0, 180]. Supported data types: U8
 */
inline uchar16 phase_unsigned(VEC_DATA_TYPE(DATA_TYPE, 16) a, VEC_DATA_TYPE(DATA_TYPE, 16) b)
{
    float16 angle_deg_f32 = atan2pi(convert_float16(b), convert_float16(a)) * (float16)180.0f;
    angle_deg_f32         = select(angle_deg_f32, (float16)180.0f + angle_deg_f32, angle_deg_f32 < (float16)0.0f);
    return convert_uchar16(angle_deg_f32);
}

/** Calculates signed phase between two inputs.
 *
 * @param[in] a First input. Supported data types: S16, S32
 * @param[in] b Second input. Supported data types: S16, S32
 *
 * @return Signed phase mapped in the interval [0, 256). Supported data types: U8
 */
inline uchar16 phase_signed(VEC_DATA_TYPE(DATA_TYPE, 16) a, VEC_DATA_TYPE(DATA_TYPE, 16) b)
{
    float16 arct = atan2pi(convert_float16(b), convert_float16(a));
    arct         = select(arct, arct + 2, arct < 0.0f);

    return convert_uchar16(convert_int16(mad(arct, 128, 0.5f)) & (int16)0xFFu);
}

#if(1 == MAGNITUDE)
#define MAGNITUDE_OP(x, y) magnitude_l1((x), (y))
#elif(2 == MAGNITUDE)
#define MAGNITUDE_OP(x, y) magnitude_l2(convert_int16(x), convert_int16(y))
#else /* MAGNITUDE */
#define MAGNITUDE_OP(x, y)
#endif /* MAGNITUDE */

#if(1 == PHASE)
#define PHASE_OP(x, y) phase_unsigned((x), (y))
#elif(2 == PHASE)
#define PHASE_OP(x, y) phase_signed((x), (y))
#else /* PHASE */
#define PHASE_OP(x, y)
#endif /* PHASE */

/** Calculate the magnitude and phase of given the gradients of an image.
 *
 * @note Magnitude calculation supported: L1 normalization(type = 1) and L2 normalization(type = 2).
 * @note Phase calculation supported: Unsigned(type = 1) [0,128] and Signed(type = 2) [0,256).
 *
 * @attention To enable phase calculation -DPHASE="phase_calculation_type_id" must be provided at compile time. eg -DPHASE=1
 * @attention To enable magnitude calculation -DMAGNITUDE="magnitude_calculation_type_id" must be provided at compile time. eg -DMAGNITUDE=1
 * @attention Datatype of the two inputs is passed at compile time using -DDATA_TYPE. e.g -DDATA_TYPE=short. Supported data_types are: short and int
 *
 * @param[in]  gx_ptr                                  Pointer to the first source image (gradient X). Supported data types: S16, S32
 * @param[in]  gx_stride_x                             Stride of the source image in X dimension (in bytes)
 * @param[in]  gx_step_x                               gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  gx_stride_y                             Stride of the source image in Y dimension (in bytes)
 * @param[in]  gx_step_y                               gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  gx_offset_first_element_in_bytes        The offset of the first element in the source image
 * @param[in]  gy_ptr                                  Pointer to the second source image (gradient Y) . Supported data types: S16, S32
 * @param[in]  gy_stride_x                             Stride of the destination image in X dimension (in bytes)
 * @param[in]  gy_step_x                               gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  gy_stride_y                             Stride of the destination image in Y dimension (in bytes)
 * @param[in]  gy_step_y                               gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  gy_offset_first_element_in_bytes        The offset of the first element in the destination image
 * @param[out] magnitude_ptr                           Pointer to the magnitude destination image. Supported data types: S16, S32
 * @param[in]  magnitude_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  magnitude_step_x                        magnitude_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  magnitude_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  magnitude_step_y                        magnitude_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  magnitude_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] phase_ptr                               Pointer to the phase destination image. Supported data types: U8
 * @param[in]  phase_stride_x                          Stride of the destination image in X dimension (in bytes)
 * @param[in]  phase_step_x                            phase_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  phase_stride_y                          Stride of the destination image in Y dimension (in bytes)
 * @param[in]  phase_step_y                            phase_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  phase_offset_first_element_in_bytes     The offset of the first element in the destination image
 * */
__kernel void magnitude_phase(
    IMAGE_DECLARATION(gx),
    IMAGE_DECLARATION(gy)
#ifdef MAGNITUDE
    ,
    IMAGE_DECLARATION(magnitude)
#endif /* MAGNITUDE */
#ifdef PHASE
    ,
    IMAGE_DECLARATION(phase)
#endif /* PHASE */
)
{
    // Get pixels pointer
    Image gx = CONVERT_TO_IMAGE_STRUCT(gx);
    Image gy = CONVERT_TO_IMAGE_STRUCT(gy);

    // Load values
    VEC_DATA_TYPE(DATA_TYPE, 16)
    in_a = vload16(0, (__global DATA_TYPE *)gx.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 16)
    in_b = vload16(0, (__global DATA_TYPE *)gy.ptr);

    // Calculate and store the results
#ifdef MAGNITUDE
    Image magnitude = CONVERT_TO_IMAGE_STRUCT(magnitude);
    vstore16(MAGNITUDE_OP(in_a, in_b), 0, (__global DATA_TYPE *)magnitude.ptr);
#endif /* MAGNITUDE */
#ifdef PHASE
    Image phase = CONVERT_TO_IMAGE_STRUCT(phase);
    vstore16(PHASE_OP(in_a, in_b), 0, phase.ptr);
#endif /* PHASE */
}
