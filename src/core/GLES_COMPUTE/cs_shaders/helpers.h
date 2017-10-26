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

#ifndef ARM_COMPUTE_HELPER_H
#define ARM_COMPUTE_HELPER_H

#define CLAMP(x, min_val, max_val) min(max(x, min_val), max_val)

#define VEC_DATA_TYPE_STR(type, size) type##size
#define VEC_DATA_TYPE(type, size) VEC_DATA_TYPE_STR(type, size)

#define CONVERT(x, type) type(x)

#define PACK(value, stype, dtype) \
    pack_##stype##_##dtype(value)

#define UNPACK(value, stype, dtype) \
    unpack_##stype##_##dtype(value)

#define BUFFER_DECLARATION(name, location, type, access)          \
    layout(std430, binding = location) access buffer name##Buffer \
    {                                                             \
        type name##_ptr[];                                        \
    }

#define VECTOR_PARAM_DECLARATION(name)         \
    uint name##_stride_x;                      \
    uint name##_step_x;                        \
    uint name##_offset_first_element_in_bytes; \
    uint name##_buffer_data_type_size

#define IMAGE_PARAM_DECLARATION(name)          \
    uint name##_stride_x;                      \
    uint name##_step_x;                        \
    uint name##_stride_y;                      \
    uint name##_step_y;                        \
    uint name##_offset_first_element_in_bytes; \
    uint name##_buffer_data_type_size

#define TENSOR3D_PARAM_DECLARATION(name)       \
    uint name##_stride_x;                      \
    uint name##_step_x;                        \
    uint name##_stride_y;                      \
    uint name##_step_y;                        \
    uint name##_stride_z;                      \
    uint name##_step_z;                        \
    uint name##_offset_first_element_in_bytes; \
    uint name##_buffer_data_type_size

/** Structure to hold Vector information */
struct Vector
{
    uint current_offset;                /**< Current offset of vector */
    uint offset_first_element_in_bytes; /**< The offset of the first element in the source image */
    uint stride_x;                      /**< Stride of the image in X dimension (in bytes) */
};

/** Structure to hold Image information */
struct Image
{
    uint current_offset;                /**< Current offset of image */
    uint offset_first_element_in_bytes; /**< The offset of the first element in the source image */
    uint stride_x;                      /**< Stride of the image in X dimension (in bytes) */
    uint stride_y;                      /**< Stride of the image in Y dimension (in bytes) */
};

/** Structure to hold 3D tensor information */
struct Tensor3D
{
    uint current_offset;                /**< Current offset of tensor */
    uint offset_first_element_in_bytes; /**< The offset of the first element in the source image */
    uint stride_x;                      /**< Stride of the image in X dimension (in bytes) */
    uint stride_y;                      /**< Stride of the image in Y dimension (in bytes) */
    uint stride_z;                      /**< Stride of the image in Z dimension (in bytes) */
};

/////////////////////////////////////////////////////////////
// TODO: old to be removed

#define CONVERT_TO_VECTOR_STRUCT(name) \
    update_vector_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x)

#define CONVERT_TO_VECTOR_STRUCT_FP16(name) \
    update_vector_workitem_offset_fp16(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x)

#define CONVERT_TO_VECTOR_STRUCT_NO_STEP(name) \
    update_vector_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, uint(0))

#define CONVERT_TO_VECTOR_STRUCT_NO_STEP_FP16(name) \
    update_vector_workitem_offset_fp16(name##_offset_first_element_in_bytes, name##_stride_x, uint(0))

#define CONVERT_TO_IMAGE_STRUCT(name) \
    update_image_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y)

#define CONVERT_TO_IMAGE_STRUCT_FP16(name) \
    update_image_workitem_offset_fp16(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y)

#define CONVERT_TO_IMAGE_STRUCT_NO_STEP(name) \
    update_image_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, uint(0), name##_stride_y, uint(0))

#define CONVERT_TO_IMAGE_STRUCT_NO_STEP_FP16(name) \
    update_image_workitem_offset_fp16(name##_offset_first_element_in_bytes, name##_stride_x, uint(0), name##_stride_y, uint(0))

#define CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(name) \
    update_image_from_tensor3D_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, uint(0), name##_stride_y, uint(0), name##_stride_z, name##_step_z)

#define CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP_FP16(name) \
    update_image_from_tensor3D_workitem_offset_fp16(name##_offset_first_element_in_bytes, name##_stride_x, uint(0), name##_stride_y, uint(0), name##_stride_z, name##_step_z)

#define CONVERT_TENSOR3D_TO_IMAGE_STRUCT(name) \
    update_image_from_tensor3D_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, name##_stride_z, name##_step_z)

#define CONVERT_TENSOR3D_TO_IMAGE_STRUCT_FP16(name) \
    update_image_from_tensor3D_workitem_offset_fp16(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, name##_stride_z, name##_step_z)

#define CONVERT_TO_TENSOR3D_STRUCT(name)                                                                                                  \
    update_tensor3D_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, \
                                    name##_stride_z, name##_step_z)

#define CONVERT_TO_TENSOR3D_STRUCT_FP16(name)                                                                                                  \
    update_tensor3D_workitem_offset_fp16(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, \
                                         name##_stride_z, name##_step_z)

#define CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(name) \
    update_tensor3D_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, uint(0), name##_stride_y, uint(0), name##_stride_z, uint(0))

#define CONVERT_TO_TENSOR3D_STRUCT_NO_STEP_FP16(name) \
    update_tensor3D_workitem_offset_fp16(name##_offset_first_element_in_bytes, name##_stride_x, uint(0), name##_stride_y, uint(0), name##_stride_z, uint(0))

// FIXME: Redesign the macros if different data types are supported.
#define LOAD4(name, offset) \
    name##_ptr[offset]

#define STORE4(name, offset, value) \
    name##_ptr[offset] = value

// Load 1 element, which size is determined by ssbo type.
#define LOAD1(r, name, offset) \
    r = name##_ptr[offset]

#define STORE1(name, offset, value) \
    name##_ptr[offset] = value

#define LOAD2(r, name, offset) \
    LOAD1(r[0], name, offset); \
    LOAD1(r[1], name, (offset) + uint(1))

#define STORE2(name, offset, value)            \
    name##_ptr[offset]             = value[0]; \
    name##_ptr[(offset) + uint(1)] = value[1]

#define LOAD3(r, name, offset)             \
    LOAD1(r[0], name, offset);             \
    LOAD1(r[1], name, (offset) + uint(1)); \
    LOAD1(r[2], name, (offset) + uint(2))

#define CURRENT_OFFSET(name) \
    name.current_offset

/** Wrap vector information into an Vector structure, and make the offset to be this workitem's position.
 *
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source vector
 * @param[in] stride_x                      Stride of the vector in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 *
 * @return An vector object
 */
Vector update_vector_workitem_offset(uint offset_first_element_in_bytes, uint stride_x, uint step_x)
{
    Vector vector;
    vector.offset_first_element_in_bytes = offset_first_element_in_bytes;
    vector.stride_x                      = stride_x;
    vector.current_offset                = (vector.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x) >> 2;

    return vector;
}

/** Wrap vector information into an Vector structure, and make the offset to be this workitem's position.
 *
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source vector
 * @param[in] stride_x                      Stride of the vector in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 *
 * @return An vector object
 */
Vector update_vector_workitem_offset_fp16(uint offset_first_element_in_bytes, uint stride_x, uint step_x)
{
    Vector vector;
    vector.offset_first_element_in_bytes = offset_first_element_in_bytes;
    vector.stride_x                      = stride_x;
    vector.current_offset                = vector.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x;

    return vector;
}

/** Wrap image information into an Image structure, and make the offset to be this workitem's position.
 *
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] stride_x                      Stride of the image in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] stride_y                      Stride of the image in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem(in bytes)
 *
 * @return An image object
 */
Image update_image_workitem_offset(uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y)
{
    Image img;
    img.offset_first_element_in_bytes = offset_first_element_in_bytes;
    img.stride_x                      = stride_x;
    img.stride_y                      = stride_y;
    img.current_offset                = (img.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y) >> 2;

    return img;
}

/** Wrap image information into an Image structure, and make the offset to be this workitem's position.
 *
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] stride_x                      Stride of the image in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] stride_y                      Stride of the image in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem(in bytes)
 *
 * @return An image object
 */
Image update_image_workitem_offset_fp16(uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y)
{
    Image img;
    img.offset_first_element_in_bytes = offset_first_element_in_bytes;
    img.stride_x                      = stride_x;
    img.stride_y                      = stride_y;
    img.current_offset                = img.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y;

    return img;
}

/** Wrap 3D tensor information into an image structure, and make the offset to be this workitem's position.
 *
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] stride_x                      Stride of the image in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] stride_y                      Stride of the image in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] stride_z                      Stride of the image in Z dimension (in bytes)
 * @param[in] step_z                        stride_z * number of elements along Z processed per workitem(in bytes)
 *
 * @return A 2D Image object
 */
Image update_image_from_tensor3D_workitem_offset(uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Image img;
    img.offset_first_element_in_bytes = offset_first_element_in_bytes;
    img.stride_x                      = stride_x;
    img.stride_y                      = stride_y;
    img.current_offset                = (img.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y + gl_GlobalInvocationID.z * step_z) >> 2;

    return img;
}

/** Wrap 3D tensor information into an image structure, and make the offset to be this workitem's position.
 *
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] stride_x                      Stride of the image in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] stride_y                      Stride of the image in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] stride_z                      Stride of the image in Z dimension (in bytes)
 * @param[in] step_z                        stride_z * number of elements along Z processed per workitem(in bytes)
 *
 * @return A 2D Image object
 */
Image update_image_from_tensor3D_workitem_offset_fp16(uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Image img;
    img.offset_first_element_in_bytes = offset_first_element_in_bytes;
    img.stride_x                      = stride_x;
    img.stride_y                      = stride_y;
    img.current_offset                = img.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y + gl_GlobalInvocationID.z * step_z;

    return img;
}

/** Wrap 3D tensor information into an tensor structure, and make the offset to be this workitem's position.
 *
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
Tensor3D update_tensor3D_workitem_offset(uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Tensor3D tensor;
    tensor.offset_first_element_in_bytes = offset_first_element_in_bytes;
    tensor.stride_x                      = stride_x;
    tensor.stride_y                      = stride_y;
    tensor.stride_z                      = stride_z;
    tensor.current_offset                = (tensor.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y + gl_GlobalInvocationID.z * step_z) >> 2;

    return tensor;
}

/** Wrap 3D tensor information into an tensor structure, and make the offset to be this workitem's position.
 *
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
Tensor3D update_tensor3D_workitem_offset_fp16(uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Tensor3D tensor;
    tensor.offset_first_element_in_bytes = offset_first_element_in_bytes;
    tensor.stride_x                      = stride_x;
    tensor.stride_y                      = stride_y;
    tensor.stride_z                      = stride_z;
    tensor.current_offset                = tensor.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y + gl_GlobalInvocationID.z * step_z;

    return tensor;
}

/** Get the pointer position of a Vector
 *
 * @param[in] vec Pointer to the starting position of the buffer
 * @param[in] x   Relative X position
 */
uint vector_offset(Vector vec, int x)
{
    return CONVERT(CONVERT(vec.current_offset << 2, int) + x * CONVERT(vec.stride_x, int), uint) >> 2;
}

/** Get the pointer position of a Vector
 *
 * @param[in] vec Pointer to the starting position of the buffer
 * @param[in] x   Relative X position
 */
uint vector_offset_fp16(Vector vec, int x)
{
    return CONVERT(CONVERT(vec.current_offset, int) + x * CONVERT(vec.stride_x, int), uint);
}

/** Get the pointer position of a Image
 *
 * @param[in] img Pointer to the starting position of the buffer
 * @param[in] x   Relative X position
 * @param[in] y   Relative Y position
 */
uint offset(Image img, int x, int y)
{
    return CONVERT(CONVERT(img.current_offset << 2, int) + x * CONVERT(img.stride_x, int) + y * CONVERT(img.stride_y, int), uint) >> 2;
}

/** Get the pointer position of a Image
 *
 * @param[in] img Pointer to the starting position of the buffer
 * @param[in] x   Relative X position
 * @param[in] y   Relative Y position
 */
uint offset_fp16(Image img, int x, int y)
{
    return CONVERT(CONVERT(img.current_offset, int) + x * CONVERT(img.stride_x, int) + y * CONVERT(img.stride_y, int), uint);
}

/** Get the pointer position of a Tensor3D
 *
 * @param[in] tensor Pointer to the starting postion of the buffer
 * @param[in] x      Relative X position
 * @param[in] y      Relative Y position
 * @param[in] z      Relative Z position
 */
uint tensor3D_offset(Tensor3D tensor, int x, int y, int z)
{
    return CONVERT(CONVERT(tensor.current_offset << 2, int) + x * CONVERT(tensor.stride_x, int) + y * CONVERT(tensor.stride_y, int) + z * CONVERT(tensor.stride_z, int), uint) >> 2;
}

/** Get the pointer position of a Tensor3D
 *
 * @param[in] tensor Pointer to the starting postion of the buffer
 * @param[in] x      Relative X position
 * @param[in] y      Relative Y position
 * @param[in] z      Relative Z position
 */
uint tensor3D_offset_fp16(Tensor3D tensor, int x, int y, int z)
{
    return CONVERT(CONVERT(tensor.current_offset, int) + x * CONVERT(tensor.stride_x, int) + y * CONVERT(tensor.stride_y, int) + z * CONVERT(tensor.stride_z, int), uint);
}

/////////////////////////////////////////////////////////////
// new one

#define GC_CONVERT_TO_VECTOR_STRUCT(name) \
    gc_update_vector_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x)

#define GC_CONVERT_TO_VECTOR_STRUCT_NO_STEP(name) \
    gc_update_vector_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, uint(0))

#define GC_CONVERT_TO_IMAGE_STRUCT(name) \
    gc_update_image_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y)

#define GC_CONVERT_TO_IMAGE_STRUCT_NO_STEP(name) \
    gc_update_image_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, uint(0), name##_stride_y, uint(0))

#define GC_CONVERT_TO_TENSOR3D_STRUCT(name)                                                                                                  \
    gc_update_tensor3D_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, \
                                       name##_stride_z, name##_step_z)

#define GC_CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(name) \
    gc_update_tensor3D_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, uint(0), name##_stride_y, uint(0), name##_stride_z, uint(0))

#define GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT(name) \
    gc_update_image_from_tensor3D_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, name##_step_x, name##_stride_y, name##_step_y, name##_stride_z, name##_step_z)

#define GC_CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(name) \
    gc_update_image_from_tensor3D_workitem_offset(name##_offset_first_element_in_bytes, name##_stride_x, uint(0), name##_stride_y, uint(0), name##_stride_z, name##_step_z)

Vector gc_update_vector_workitem_offset(uint offset_first_element_in_bytes, uint stride_x, uint step_x)
{
    Vector vector;
    vector.offset_first_element_in_bytes = offset_first_element_in_bytes;
    vector.stride_x                      = stride_x;
    vector.current_offset                = vector.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x;

    return vector;
}

Image gc_update_image_workitem_offset(uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y)
{
    Image img;
    img.offset_first_element_in_bytes = offset_first_element_in_bytes;
    img.stride_x                      = stride_x;
    img.stride_y                      = stride_y;
    img.current_offset                = img.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y;

    return img;
}

Tensor3D gc_update_tensor3D_workitem_offset(uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Tensor3D tensor;
    tensor.offset_first_element_in_bytes = offset_first_element_in_bytes;
    tensor.stride_x                      = stride_x;
    tensor.stride_y                      = stride_y;
    tensor.stride_z                      = stride_z;
    tensor.current_offset                = tensor.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y + gl_GlobalInvocationID.z * step_z;

    return tensor;
}

Image gc_update_image_from_tensor3D_workitem_offset(uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Image img;
    img.offset_first_element_in_bytes = offset_first_element_in_bytes;
    img.stride_x                      = stride_x;
    img.stride_y                      = stride_y;
    img.current_offset                = img.offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y + gl_GlobalInvocationID.z * step_z;

    return img;
}

#define GC_CURRENT_OFFSET(name) \
    name.current_offset

uint gc_vector_offset(Vector vec, int x)
{
    return CONVERT(CONVERT(vec.current_offset, int) + x * CONVERT(vec.stride_x, int), uint);
}

uint gc_image_offset(Image img, int x, int y)
{
    return CONVERT(CONVERT(img.current_offset, int) + x * CONVERT(img.stride_x, int) + y * CONVERT(img.stride_y, int), uint);
}

uint gc_tensor3D_offset(Tensor3D tensor, int x, int y, int z)
{
    return CONVERT(CONVERT(tensor.current_offset, int) + x * CONVERT(tensor.stride_x, int) + y * CONVERT(tensor.stride_y, int) + z * CONVERT(tensor.stride_z, int), uint);
}

// load/store number of element depends on buffer type
#define GC_LOAD1(r, name, offset) \
    r = name##_ptr[offset]

#define GC_LOAD2(r, name, offset) \
    GC_LOAD1(r[0], name, offset); \
    GC_LOAD1(r[1], name, (offset) + uint(1))

#define GC_LOAD3(r, name, offset)             \
    GC_LOAD1(r[0], name, offset);             \
    GC_LOAD1(r[1], name, (offset) + uint(1)); \
    GC_LOAD1(r[2], name, (offset) + uint(2))

#define GC_STORE1(value, name, offset) \
    name##_ptr[offset] = value

#define GC_STORE2(value, name, offset) \
    GC_STORE1(value[0], name, offset); \
    GC_STORE1(value[1], name, (offset) + uint(1))

#define GC_STORE3(value, name, offset)             \
    GC_STORE1(value[0], name, offset);             \
    GC_STORE1(value[1], name, (offset) + uint(1)); \
    GC_STORE1(value[2], name, (offset) + uint(2))

// has to manually expand them since not supported by compiler
#define GC_LOAD1_1D_OFFSET(r, name, x) \
    GC_LOAD1(r, name, gc_vector_offset(name, int(x)) >> name##_buffer_data_type_size)

#define GC_LOAD1_2D_OFFSET(r, name, x, y) \
    GC_LOAD1(r, name, gc_image_offset(name, int(x), int(y)) >> name##_buffer_data_type_size)

#define GC_LOAD1_3D_OFFSET(r, name, x, y, z) \
    GC_LOAD1(r, name, gc_tensor3D_offset(name, int(x), int(y), int(z)) >> name##_buffer_data_type_size)

#define GC_STORE1_1D_OFFSET(value, name, x) \
    GC_STORE1(value, name, gc_vector_offset(name, int(x)) >> name##_buffer_data_type_size)

#define GC_STORE1_2D_OFFSET(value, name, x, y) \
    GC_STORE1(value, name, gc_image_offset(name, int(x), int(y)) >> name##_buffer_data_type_size)

#define GC_STORE1_3D_OFFSET(value, name, x, y, z) \
    GC_STORE1(value, name, gc_tensor3D_offset(name, int(x), int(y), int(z)) >> name##_buffer_data_type_size)

#define GC_LOAD2_1D_OFFSET(r, name, x) \
    GC_LOAD2(r, name, gc_vector_offset(name, int(x)) >> name##_buffer_data_type_size)

#define GC_LOAD2_2D_OFFSET(r, name, x, y) \
    GC_LOAD2(r, name, gc_image_offset(name, int(x), int(y)) >> name##_buffer_data_type_size)

#define GC_LOAD2_3D_OFFSET(r, name, x, y, z) \
    GC_LOAD2(r, name, gc_tensor3D_offset(name, int(x), int(y), int(z)) >> name##_buffer_data_type_size)

#define GC_STORE2_1D_OFFSET(value, name, x) \
    GC_STORE2(value, name, gc_vector_offset(name, int(x)) >> name##_buffer_data_type_size)

#define GC_STORE2_2D_OFFSET(value, name, x, y) \
    GC_STORE2(value, name, gc_image_offset(name, int(x), int(y)) >> name##_buffer_data_type_size)

#define GC_STORE2_3D_OFFSET(value, name, x, y, z) \
    GC_STORE2(value, name, gc_tensor3D_offset(name, int(x), int(y), int(z)) >> name##_buffer_data_type_size)

#define GC_LOAD3_1D_OFFSET(r, name, x) \
    GC_LOAD3(r, name, gc_vector_offset(name, int(x)) >> name##_buffer_data_type_size)

#define GC_LOAD3_2D_OFFSET(r, name, x, y) \
    GC_LOAD3(r, name, gc_image_offset(name, int(x), int(y)) >> name##_buffer_data_type_size)

#define GC_LOAD3_3D_OFFSET(r, name, x, y, z) \
    GC_LOAD3(r, name, gc_tensor3D_offset(name, int(x), int(y), int(z)) >> name##_buffer_data_type_size)

/////////////////////////////////////////////////////////////

#endif // _HELPER_H
