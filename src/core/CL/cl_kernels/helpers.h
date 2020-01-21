/*
 * Copyright (c) 2016-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_HELPER_H
#define ARM_COMPUTE_HELPER_H

#if defined(ARM_COMPUTE_OPENCL_FP16_ENABLED) && defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif // defined(ARM_COMPUTE_OPENCL_FP16_ENABLED) && defined(cl_khr_fp16)

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#if defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)

#if defined(ARM_COMPUTE_DEBUG_ENABLED) && defined(cl_arm_printf)
#pragma OPENCL EXTENSION cl_arm_printf : enable
#endif // defined(ARM_COMPUTE_DEBUG_ENABLED) && defined(cl_arm_printf)

#define GPU_ARCH_MIDGARD 0x100
#define GPU_ARCH_BIFROST 0x200

/** Concatenate two inputs.
 *
 * @param[in] a The first input to be concatenated
 * @param[in] b The second input to be concatenated
 *
 * @return The concatenated output
 */
#define CONCAT(a, b) a##b

/** Expand the given vector
 *
 * @param[in] x The vector to be expanded
 *
 * @return The expanded output
 */
#define EXPAND(x) x

/** Clamp the given value between an upper and lower bound.
 *
 * @param[in] x       The value to be clamped
 * @param[in] min_val The lower bound
 * @param[in] max_val The upper bound
 *
 * @return The clamped value.
 */
#define CLAMP(x, min_val, max_val) min(max(x, min_val), max_val)

/** REVn reverses the given vector whose size is n.
 * @name REVn
 *
 * @param[in] x The vector to be reversed
 *
 * @return The reversed vector
 * @{
 */
#define REV1(x) ((x))
#define REV2(x) ((x).s10)
#define REV3(x) ((x).s210)
#define REV4(x) ((x).s3210)
#define REV8(x) ((x).s76543210)
#define REV16(x) ((x).sFEDCBA9876543210)
/** @} */ // end of group REVn

/** Reverse the given vector.
 * @name REVERSE
 *
 * @param[in] x The vector to be reversed
 * @param[in] s The size of the vector
 *
 * @return The reversed vector
 * @{
 */
#define REVERSE_STR(x, s) REV##s((x))
#define REVERSE(x, s) REVERSE_STR(x, s)
/** @} */ // end of group REVERSE

/** Circular-right-shift (rotate-right) the vector of size s by the amount of n.
 * @name ROTs_n
 *
 * @param[in] x The vector to be shifted
 *
 * @return The shifted vector
 * @{
 */
#define ROT1_0(x) ((x))

#define ROT2_0(x) ((x))
#define ROT2_1(x) ((x).s10)

#define ROT3_0(x) ((x))
#define ROT3_1(x) ((x).s201)
#define ROT3_2(x) ((x).s120)

#define ROT4_0(x) ((x))
#define ROT4_1(x) ((x).s3012)
#define ROT4_2(x) ((x).s2301)
#define ROT4_3(x) ((x).s1230)

#define ROT8_0(x) ((x))
#define ROT8_1(x) ((x).s70123456)
#define ROT8_2(x) ((x).s67012345)
#define ROT8_3(x) ((x).s56701234)
#define ROT8_4(x) ((x).s45670123)
#define ROT8_5(x) ((x).s34567012)
#define ROT8_6(x) ((x).s23456701)
#define ROT8_7(x) ((x).s12345670)

#define ROT16_0(x) ((x))
#define ROT16_1(x) ((x).sF0123456789ABCDE)
#define ROT16_2(x) ((x).sEF0123456789ABCD)
#define ROT16_3(x) ((x).sDEF0123456789ABC)
#define ROT16_4(x) ((x).sCDEF0123456789AB)
#define ROT16_5(x) ((x).sBCDEF0123456789A)
#define ROT16_6(x) ((x).sABCDEF0123456789)
#define ROT16_7(x) ((x).s9ABCDEF012345678)
#define ROT16_8(x) ((x).s89ABCDEF01234567)
#define ROT16_9(x) ((x).s789ABCDEF0123456)
#define ROT16_10(x) ((x).s6789ABCDEF012345)
#define ROT16_11(x) ((x).s56789ABCDEF01234)
#define ROT16_12(x) ((x).s456789ABCDEF0123)
#define ROT16_13(x) ((x).s3456789ABCDEF012)
#define ROT16_14(x) ((x).s23456789ABCDEF01)
#define ROT16_15(x) ((x).s123456789ABCDEF0)
/** @} */ // end of group ROTs_n

/** Circular-right-shift (rotate-right) the given vector by the given amount.
 * @name ROTATE
 *
 * @param[in] x The vector to be shifted
 * @param[in] s The size of the vector
 * @param[in] n The amount to be shifted
 *
 * @return The shifted vector
 * @{
 */
#define ROTATE_STR(x, s, n) ROT##s##_##n(x)
#define ROTATE(x, s, n) ROTATE_STR(x, s, n)
/** @} */ // end of group ROTATE

/** Creates a vector of size n filled with offset values corresponding to the location of each element.
 * @name V_OFFSn
 *
 * @param[in] dt The data type of the output vector
 *
 * @return The vector filled with offset values
 * @{
 */
#define V_OFFS1(dt) (dt)(0)
#define V_OFFS2(dt) (dt)(0, 1)
#define V_OFFS3(dt) (dt)(0, 1, 3)
#define V_OFFS4(dt) (dt)(0, 1, 2, 3)
#define V_OFFS8(dt) (dt)(0, 1, 2, 3, 4, 5, 6, 7)
#define V_OFFS16(dt) (dt)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
/** @} */ // end of group V_OFFSn

/** Create a vector filled with offset values corresponding to the location of each element.
 * @name VEC_OFFS
 *
 * @param[in] dt The data type of the output vector
 * @param[in] s  The size of the output vector
 *
 * @return The vector filled with offset values
 * @{
 */
#define VEC_OFFS_STR(dt, s) V_OFFS##s(dt)
#define VEC_OFFS(dt, s) VEC_OFFS_STR(dt, s)
/** @} */ // end of group VEC_OFFS

#define VLOAD_STR(size) vload##size
#define VLOAD(size) VLOAD_STR(size)

#define VSTORE_STR(size) vstore##size
#define VSTORE(size) VSTORE_STR(size)

#define float1 float
#define half1 half
#define char1 char
#define uchar1 uchar
#define short1 short
#define ushort1 ushort
#define int1 int
#define uint1 uint
#define long1 long
#define ulong1 ulong
#define double1 double

#define vload1(OFFSET, PTR) *(OFFSET + PTR)
#define vstore1(DATA, OFFSET, PTR) *(OFFSET + PTR) = DATA

// Convert built-in functions with _sat modifier are not supported in floating point so we create defines
// without _sat to overcome this issue
#define convert_float_sat convert_float
#define convert_float1_sat convert_float
#define convert_float2_sat convert_float2
#define convert_float3_sat convert_float3
#define convert_float4_sat convert_float4
#define convert_float8_sat convert_float8
#define convert_float16_sat convert_float16
#define convert_half_sat convert_float
#define convert_half1_sat convert_half
#define convert_half2_sat convert_half2
#define convert_half3_sat convert_half3
#define convert_half4_sat convert_half4
#define convert_half8_sat convert_half8
#define convert_half16_sat convert_half16

#define convert_float1 convert_float
#define convert_half1 convert_half
#define convert_char1 convert_char
#define convert_uchar1 convert_uchar
#define convert_short1 convert_short
#define convert_ushort1 convert_ushort
#define convert_int1 convert_int
#define convert_uint1 convert_uint
#define convert_long1 convert_long
#define convert_ulong1 convert_ulong
#define convert_double1 convert_double

#define convert_char1_sat convert_char_sat
#define convert_uchar1_sat convert_uchar_sat
#define convert_short1_sat convert_short_sat
#define convert_ushort1_sat convert_ushort_sat
#define convert_int1_sat convert_int_sat
#define convert_uint1_sat convert_uint_sat
#define convert_long1_sat convert_long_sat
#define convert_ulong1_sat convert_ulong_sat
#define convert_double1_sat convert_double_sat

#define VEC_DATA_TYPE_STR(type, size) type##size
#define VEC_DATA_TYPE(type, size) VEC_DATA_TYPE_STR(type, size)

#define CL_VEC_DATA_TYPE_STR(type, size) type##size
#define CL_VEC_DATA_TYPE(type, size) CL_VEC_DATA_TYPE_STR(type, size)

#define CONVERT_STR(x, type) (convert_##type((x)))
#define CONVERT(x, type) CONVERT_STR(x, type)

#define CONVERT_SAT_STR(x, type) (convert_##type##_sat((x)))
#define CONVERT_SAT(x, type) CONVERT_SAT_STR(x, type)

#define CONVERT_SAT_ROUND_STR(x, type, round) (convert_##type##_sat_##round((x)))
#define CONVERT_SAT_ROUND(x, type, round) CONVERT_SAT_ROUND_STR(x, type, round)

#define VECTOR_DECLARATION(name)     \
    __global uchar *name##_ptr,      \
    uint        name##_stride_x, \
    uint        name##_step_x,   \
    uint        name##_offset_first_element_in_bytes

#define IMAGE_DECLARATION(name)      \
    __global uchar *name##_ptr,      \
    uint        name##_stride_x, \
    uint        name##_step_x,   \
    uint        name##_stride_y, \
    uint        name##_step_y,   \
    uint        name##_offset_first_element_in_bytes

#define TENSOR3D_DECLARATION(name)   \
    __global uchar *name##_ptr,      \
    uint        name##_stride_x, \
    uint        name##_step_x,   \
    uint        name##_stride_y, \
    uint        name##_step_y,   \
    uint        name##_stride_z, \
    uint        name##_step_z,   \
    uint        name##_offset_first_element_in_bytes

#define TENSOR4D_DECLARATION(name)   \
    __global uchar *name##_ptr,      \
    uint        name##_stride_x, \
    uint        name##_step_x,   \
    uint        name##_stride_y, \
    uint        name##_step_y,   \
    uint        name##_stride_z, \
    uint        name##_step_z,   \
    uint        name##_stride_w, \
    uint        name##_step_w,   \
    uint        name##_offset_first_element_in_bytes

#define CONVERT_TO_VECTOR_STRUCT(name) \
    update_vector_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x)

#define CONVERT_TO_VECTOR_STRUCT_NO_STEP(name) \
    update_vector_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, 0)

#define CONVERT_TO_IMAGE_STRUCT(name) \
    update_image_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y)

#define CONVERT_TO_IMAGE_STRUCT_NO_STEP(name) \
    update_image_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, 0, name##_stride_y, 0)

#define CONVERT_TENSOR3D_TO_IMAGE_STRUCT(name) \
    update_image_from_tensor3D_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, name##_stride_z, name##_step_z)

#define CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(name) \
    update_image_from_tensor3D_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, 0, name##_stride_y, 0, name##_stride_z, name##_step_z)

#define CONVERT_TENSOR3D_TO_IMAGE_STRUCT(name) \
    update_image_from_tensor3D_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, name##_stride_z, name##_step_z)

#define CONVERT_TO_TENSOR3D_STRUCT(name)                                                                                                           \
    update_tensor3D_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, \
                                 name##_stride_z, name##_step_z)

#define CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(name) \
    update_tensor3D_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, 0, name##_stride_y, 0, name##_stride_z, 0)

#define CONVERT_TO_TENSOR4D_STRUCT(name, mod_size)                                                                                                 \
    update_tensor4D_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, \
                                 name##_stride_z, name##_step_z, name##_stride_w, name##_step_w, mod_size)

#define CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(name, mod_size) \
    update_tensor4D_workitem_ptr(name##_ptr, name##_offset_first_element_in_bytes, name##_stride_x, 0, name##_stride_y, 0, name##_stride_z, 0, name##_stride_w, 0, mod_size)

/** Structure to hold Vector information */
typedef struct Vector
{
    __global uchar *ptr;                           /**< Pointer to the starting postion of the buffer */
    int             offset_first_element_in_bytes; /**< The offset of the first element in the source image */
    int             stride_x;                      /**< Stride of the image in X dimension (in bytes) */
} Vector;

/** Structure to hold Image information */
typedef struct Image
{
    __global uchar *ptr;                           /**< Pointer to the starting postion of the buffer */
    int             offset_first_element_in_bytes; /**< The offset of the first element in the source image */
    int             stride_x;                      /**< Stride of the image in X dimension (in bytes) */
    int             stride_y;                      /**< Stride of the image in Y dimension (in bytes) */
} Image;

/** Structure to hold 3D tensor information */
typedef struct Tensor3D
{
    __global uchar *ptr;                           /**< Pointer to the starting postion of the buffer */
    int             offset_first_element_in_bytes; /**< The offset of the first element in the source image */
    int             stride_x;                      /**< Stride of the image in X dimension (in bytes) */
    int             stride_y;                      /**< Stride of the image in Y dimension (in bytes) */
    int             stride_z;                      /**< Stride of the image in Z dimension (in bytes) */
} Tensor3D;

/** Structure to hold 4D tensor information */
typedef struct Tensor4D
{
    __global uchar *ptr;                           /**< Pointer to the starting postion of the buffer */
    int             offset_first_element_in_bytes; /**< The offset of the first element in the source image */
    int             stride_x;                      /**< Stride of the image in X dimension (in bytes) */
    int             stride_y;                      /**< Stride of the image in Y dimension (in bytes) */
    int             stride_z;                      /**< Stride of the image in Z dimension (in bytes) */
    int             stride_w;                      /**< Stride of the image in W dimension (in bytes) */
} Tensor4D;

/** Wrap vector information into an Vector structure, and make the pointer point at this workitem's data.
 *
 * @param[in] ptr                           Pointer to the starting postion of the buffer
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source vector
 * @param[in] stride_x                      Stride of the vector in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 *
 * @return An image object
 */
inline Vector update_vector_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x)
{
    Vector vector =
    {
        .ptr                           = ptr,
        .offset_first_element_in_bytes = offset_first_element_in_bytes,
        .stride_x                      = stride_x,
    };
    vector.ptr += vector.offset_first_element_in_bytes + get_global_id(0) * step_x;
    return vector;
}

/** Wrap image information into an Image structure, and make the pointer point at this workitem's data.
 *
 * @param[in] ptr                           Pointer to the starting postion of the buffer
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] stride_x                      Stride of the image in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] stride_y                      Stride of the image in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem(in bytes)
 *
 * @return An image object
 */
inline Image update_image_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y)
{
    Image img =
    {
        .ptr                           = ptr,
        .offset_first_element_in_bytes = offset_first_element_in_bytes,
        .stride_x                      = stride_x,
        .stride_y                      = stride_y
    };
    img.ptr += img.offset_first_element_in_bytes + get_global_id(0) * step_x + get_global_id(1) * step_y;
    return img;
}

/** Wrap 3D tensor information into an image structure, and make the pointer point at this workitem's data.
 *
 * @param[in] ptr                           Pointer to the starting postion of the buffer
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] stride_x                      Stride of the image in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] stride_y                      Stride of the image in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] stride_z                      Stride of the image in Z dimension (in bytes)
 * @param[in] step_z                        stride_z * number of elements along Z processed per workitem(in bytes)
 *
 * @return A 3D tensor object
 */
inline Image update_image_from_tensor3D_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Image img =
    {
        .ptr                           = ptr,
        .offset_first_element_in_bytes = offset_first_element_in_bytes,
        .stride_x                      = stride_x,
        .stride_y                      = stride_y
    };
    img.ptr += img.offset_first_element_in_bytes + get_global_id(0) * step_x + get_global_id(1) * step_y + get_global_id(2) * step_z;
    return img;
}

/** Wrap 3D tensor information into an tensor structure, and make the pointer point at this workitem's data.
 *
 * @param[in] ptr                           Pointer to the starting postion of the buffer
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] stride_x                      Stride of the image in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] stride_y                      Stride of the image in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] stride_z                      Stride of the image in Z dimension (in bytes)
 * @param[in] step_z                        stride_z * number of elements along Z processed per workitem(in bytes)
 *
 * @return A 3D tensor object
 */
inline Tensor3D update_tensor3D_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Tensor3D tensor =
    {
        .ptr                           = ptr,
        .offset_first_element_in_bytes = offset_first_element_in_bytes,
        .stride_x                      = stride_x,
        .stride_y                      = stride_y,
        .stride_z                      = stride_z
    };
    tensor.ptr += tensor.offset_first_element_in_bytes + get_global_id(0) * step_x + get_global_id(1) * step_y + get_global_id(2) * step_z;
    return tensor;
}

inline Tensor4D update_tensor4D_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z, uint stride_w,
                                             uint step_w,
                                             uint mod_size)
{
    Tensor4D tensor =
    {
        .ptr                           = ptr,
        .offset_first_element_in_bytes = offset_first_element_in_bytes,
        .stride_x                      = stride_x,
        .stride_y                      = stride_y,
        .stride_z                      = stride_z,
        .stride_w                      = stride_w
    };

    tensor.ptr += tensor.offset_first_element_in_bytes + get_global_id(0) * step_x + get_global_id(1) * step_y + (get_global_id(2) % mod_size) * step_z + (get_global_id(2) / mod_size) * step_w;
    return tensor;
}

/** Get the pointer position of a Vector
 *
 * @param[in] vec Pointer to the starting position of the buffer
 * @param[in] x   Relative X position
 */
inline __global const uchar *vector_offset(const Vector *vec, int x)
{
    return vec->ptr + x * vec->stride_x;
}

/** Get the pointer position of a Image
 *
 * @param[in] img Pointer to the starting position of the buffer
 * @param[in] x   Relative X position
 * @param[in] y   Relative Y position
 */
inline __global uchar *offset(const Image *img, int x, int y)
{
    return img->ptr + x * img->stride_x + y * img->stride_y;
}

/** Get the pointer position of a Tensor3D
 *
 * @param[in] tensor Pointer to the starting position of the buffer
 * @param[in] x      Relative X position
 * @param[in] y      Relative Y position
 * @param[in] z      Relative Z position
 */
inline __global const uchar *tensor3D_offset(const Tensor3D *tensor, int x, int y, int z)
{
    return tensor->ptr + x * tensor->stride_x + y * tensor->stride_y + z * tensor->stride_z;
}

/** Get the pointer position of a Tensor4D
 *
 * @param[in] tensor Pointer to the starting position of the buffer
 * @param[in] x      Relative X position
 * @param[in] y      Relative Y position
 * @param[in] z      Relative Z position
 * @param[in] w      Relative W position
 */
inline __global const uchar *tensor4D_offset(const Tensor4D *tensor, int x, int y, int z, int w)
{
    return tensor->ptr + x * tensor->stride_x + y * tensor->stride_y + z * tensor->stride_z + w * tensor->stride_w;
}

#endif // _HELPER_H
