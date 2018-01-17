/*
 * Copyright (c) 2016-2018 ARM Limited.
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

#if defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif // defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)

#if defined(ARM_COMPUTE_DEBUG_ENABLED)
#pragma OPENCL EXTENSION cl_arm_printf : enable
#endif // defined(ARM_COMPUTE_DEBUG_ENABLED)

#define EXPAND(x) x

#define CLAMP(x, min_val, max_val) min(max(x, min_val), max_val)

#define VLOAD_STR(size) vload##size
#define VLOAD(size) VLOAD_STR(size)

#define VSTORE_STR(size) vstore##size
#define VSTORE(size) VSTORE_STR(size)

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
Vector inline update_vector_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x)
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
Image inline update_image_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y)
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
Image inline update_image_from_tensor3D_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
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
Tensor3D inline update_tensor3D_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
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

Tensor4D inline update_tensor4D_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z, uint stride_w,
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
__global inline const uchar *vector_offset(const Vector *vec, int x)
{
    return vec->ptr + x * vec->stride_x;
}

/** Get the pointer position of a Image
 *
 * @param[in] img Pointer to the starting position of the buffer
 * @param[in] x   Relative X position
 * @param[in] y   Relative Y position
 */
__global inline uchar *offset(const Image *img, int x, int y)
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
__global inline const uchar *tensor3D_offset(const Tensor3D *tensor, int x, int y, int z)
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
__global inline const uchar *tensor4D_offset(const Tensor4D *tensor, int x, int y, int z, int w)
{
    return tensor->ptr + x * tensor->stride_x + y * tensor->stride_y + z * tensor->stride_z + w * tensor->stride_w;
}

#endif // _HELPER_H
