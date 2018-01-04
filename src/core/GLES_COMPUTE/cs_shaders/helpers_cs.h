/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#ifndef ARM_COMPUTE_HELPER_CS_H
#define ARM_COMPUTE_HELPER_CS_H

#define SHADER_PARAMS_DECLARATION \
    layout(std140, binding = 0) uniform shader_params

#define TENSOR_DECLARATION(location, buffer_type, type, ptr_name, shift_name, element_shift, access) \
    layout(std430, binding = location) access buffer buffer_type                                     \
    {                                                                                                \
        type ptr_name[];                                                                             \
    };                                                                                               \
    const uint shift_name = uint(element_shift)

struct VectorAttributes
{
    uint stride_x;                      /**< Stride of the vector in X dimension (in bytes) */
    uint step_x;                        /**< stride_x * number of elements along X processed per workitem (in bytes) */
    uint offset_first_element_in_bytes; /**< The offset of the first element in the vector (in bytes) */
    uint padding;                       /**< The padding to rounding up the structure to a multiple of a vec4 */
};

struct ImageAttributes
{
    uint stride_x;                      /**< Stride of the image in X dimension (in bytes) */
    uint step_x;                        /**< stride_x * number of elements along X processed per workitem (in bytes) */
    uint stride_y;                      /**< Stride of the image in Y dimension (in bytes) */
    uint step_y;                        /**< stride_y * number of elements along Y processed per workitem (in bytes) */
    uint offset_first_element_in_bytes; /**< The offset of the first element in the image (in bytes) */
    uint padding1;                      /**< The padding to rounding up the structure to a multiple of a vec4 */
    uint padding2;                      /**< The padding to rounding up the structure to a multiple of a vec4 */
    uint padding3;                      /**< The padding to rounding up the structure to a multiple of a vec4 */
};

struct Tensor3DAttributes
{
    uint stride_x;                      /**< Stride of the tensor in X dimension (in bytes) */
    uint step_x;                        /**< stride_x * number of elements along X processed per workitem (in bytes) */
    uint stride_y;                      /**< Stride of the tensor in Y dimension (in bytes) */
    uint step_y;                        /**< stride_y * number of elements along Y processed per workitem (in bytes) */
    uint stride_z;                      /**< Stride of the tensor in Z dimension (in bytes) */
    uint step_z;                        /**< stride_z * number of elements along Z processed per workitem (in bytes) */
    uint offset_first_element_in_bytes; /**< The offset of the first element in the tensor (in bytes) */
    uint padding;                       /**< The padding to rounding up the structure to a multiple of a vec4 */
};

struct VectorIterator
{
    int current_offset_in_bytes; /**< Current offset of vector (in bytes) */
    int stride_x;                /**< Stride of the vector in X dimension (in bytes) */
    int element_shift;           /**< The number of bits to shift by for one element */
};

struct ImageIterator
{
    int current_offset_in_bytes; /**< Current offset of image (in bytes) */
    int stride_x;                /**< Stride of the image in X dimension (in bytes) */
    int stride_y;                /**< Stride of the image in Y dimension (in bytes) */
    int element_shift;           /**< The number of bits to shift by for one element */
};

struct Tensor3DIterator
{
    int current_offset_in_bytes; /**< Current offset of tensor (in bytes) */
    int stride_x;                /**< Stride of the tensor in X dimension (in bytes) */
    int stride_y;                /**< Stride of the tensor in Y dimension (in bytes) */
    int stride_z;                /**< Stride of the tensor in Z dimension (in bytes) */
    int element_shift;           /**< The number of bits to shift by for one element */
};

#define CONVERT_TO_VECTOR_ITERATOR(attrs, element_shift)                          \
    update_vector_iter_offset(element_shift, attrs.offset_first_element_in_bytes, \
                              attrs.stride_x, attrs.step_x)

#define CONVERT_TO_VECTOR_ITERATOR_NO_STEP(attrs, element_shift)                  \
    update_vector_iter_offset(element_shift, attrs.offset_first_element_in_bytes, \
                              attrs.stride_x, uint(0))

#define CONVERT_TO_IMAGE_ITERATOR(attrs, element_shift)                          \
    update_image_iter_offset(element_shift, attrs.offset_first_element_in_bytes, \
                             attrs.stride_x, attrs.step_x, attrs.stride_y, attrs.step_y)

#define CONVERT_TO_IMAGE_ITERATOR_NO_STEP(attrs, element_shift)                  \
    update_image_iter_offset(element_shift, attrs.offset_first_element_in_bytes, \
                             attrs.stride_x, uint(0), attrs.stride_y, uint(0))

#define CONVERT_TO_TENSOR3D_ITERATOR(attrs, element_shift)                          \
    update_tensor3D_iter_offset(element_shift, attrs.offset_first_element_in_bytes, \
                                attrs.stride_x, attrs.step_x, attrs.stride_y, attrs.step_y, attrs.stride_z, attrs.step_z)

#define CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(attrs, element_shift)                  \
    update_tensor3D_iter_offset(element_shift, attrs.offset_first_element_in_bytes, \
                                attrs.stride_x, uint(0), attrs.stride_y, uint(0), attrs.stride_z, uint(0))

#define CONVERT_TENSOR3D_TO_IMAGE_ITERATOR(attrs, element_shift)                               \
    update_image_from_tensor3D_iter_offset(element_shift, attrs.offset_first_element_in_bytes, \
                                           attrs.stride_x, attrs.step_x, attrs.stride_y, attrs.step_y, attrs.stride_z, attrs.step_z)

#define CONVERT_TENSOR3D_TO_IMAGE_ITERATOR_NO_STEP(attrs, element_shift)                       \
    update_image_from_tensor3D_iter_offset(element_shift, attrs.offset_first_element_in_bytes, \
                                           attrs.stride_x, uint(0), attrs.stride_y, uint(0), attrs.stride_z, attrs.step_z)

/** Wrap vector information into a VectorIterator structure, and make the offset to be this workitem's position.
 *
 * @param[in] element_shift                 The number of bits to shift by for one element
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source vector
 * @param[in] stride_x                      Stride of the vector in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem (in bytes)
 *
 * @return A VectorIterator object
 */
VectorIterator update_vector_iter_offset(uint element_shift, uint offset_first_element_in_bytes, uint stride_x, uint step_x)
{
    VectorIterator vector_iter;
    vector_iter.element_shift           = int(element_shift);
    vector_iter.stride_x                = int(stride_x);
    vector_iter.current_offset_in_bytes = int(offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x);

    return vector_iter;
}

/** Wrap image information into an ImageIterator structure, and make the offset to be this workitem's position.
 *
 * @param[in] element_shift                 The number of bits to shift by for one element
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] stride_x                      Stride of the image in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem (in bytes)
 * @param[in] stride_y                      Stride of the image in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem (in bytes)
 *
 * @return An ImageIterator object
 */
ImageIterator update_image_iter_offset(uint element_shift, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y)
{
    ImageIterator image_iter;
    image_iter.element_shift           = int(element_shift);
    image_iter.stride_x                = int(stride_x);
    image_iter.stride_y                = int(stride_y);
    image_iter.current_offset_in_bytes = int(offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y);

    return image_iter;
}

/** Wrap 3D tensor information into a Tensor3DIterator structure, and make the offset to be this workitem's position.
 *
 * @param[in] element_shift                 The number of bits to shift by for one element
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source tersor
 * @param[in] stride_x                      Stride of the tersor in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem (in bytes)
 * @param[in] stride_y                      Stride of the tersor in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem (in bytes)
 * @param[in] stride_z                      Stride of the tersor in Z dimension (in bytes)
 * @param[in] step_z                        stride_z * number of elements along Z processed per workitem (in bytes)
 *
 * @return A 3D Tensor3DIterator object
 */
Tensor3DIterator update_tensor3D_iter_offset(uint element_shift, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Tensor3DIterator tensor_iter;
    tensor_iter.element_shift           = int(element_shift);
    tensor_iter.stride_x                = int(stride_x);
    tensor_iter.stride_y                = int(stride_y);
    tensor_iter.stride_z                = int(stride_z);
    tensor_iter.current_offset_in_bytes = int(offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y + gl_GlobalInvocationID.z * step_z);

    return tensor_iter;
}

/** Wrap 3D tensor information into an ImageIterator structure, and make the offset to be this workitem's position.
 *
 * @param[in] element_shift                 The number of bits to shift by for one element
 * @param[in] offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] stride_x                      Stride of the tensor in X dimension (in bytes)
 * @param[in] step_x                        stride_x * number of elements along X processed per workitem (in bytes)
 * @param[in] stride_y                      Stride of the tensor in Y dimension (in bytes)
 * @param[in] step_y                        stride_y * number of elements along Y processed per workitem (in bytes)
 * @param[in] stride_z                      Stride of the tensor in Z dimension (in bytes)
 * @param[in] step_z                        stride_z * number of elements along Z processed per workitem (in bytes)
 *
 * @return An ImageIterator object
 */
ImageIterator update_image_from_tensor3D_iter_offset(uint element_shift, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    ImageIterator image_iter;
    image_iter.element_shift           = int(element_shift);
    image_iter.stride_x                = int(stride_x);
    image_iter.stride_y                = int(stride_y);
    image_iter.current_offset_in_bytes = int(offset_first_element_in_bytes + gl_GlobalInvocationID.x * step_x + gl_GlobalInvocationID.y * step_y + gl_GlobalInvocationID.z * step_z);

    return image_iter;
}

#define VECTOR_OFFSET(tensor_iter, x) \
    uint(vector_offset_in_bytes(tensor_iter, int(x)) >> tensor_iter.element_shift)

#define IMAGE_OFFSET(tensor_iter, x, y) \
    uint(image_offset_in_bytes(tensor_iter, int(x), int(y)) >> tensor_iter.element_shift)

#define TENSOR3D_OFFSET(tensor_iter, x, y, z) \
    uint(tensor3D_offset_in_bytes(tensor_iter, int(x), int(y), int(z)) >> tensor_iter.element_shift)

#define TENSOR_OFFSET_ADVANCE(tensor_iter, n) \
    uint((tensor_iter.current_offset_in_bytes >> tensor_iter.element_shift) + int(n))

#define TENSOR_OFFSET_ADVANCE_IN_BYTES(tensor_iter, n) \
    uint((tensor_iter.current_offset_in_bytes + int(n)) >> tensor_iter.element_shift)

#define CURRENT_ITEM_OFFSET(tensor_iter) \
    uint(tensor_iter.current_offset_in_bytes >> tensor_iter.element_shift)

#define CURRENT_ITEM_OFFSET_IN_BYTES(tensor_iter) \
    uint(tensor_iter.current_offset_in_bytes)

#define TENSOR_ITERATOR_ADVANCE(tensor_iter, n) \
    tensor_iter.current_offset_in_bytes += (int(n) << tensor_iter.element_shift)

#define TENSOR_ITERATOR_ADVANCE_IN_BYTES(tensor_iter, n) \
    tensor_iter.current_offset_in_bytes += int(n)

#define SET_TENSOR_ITERATOR_OFFSET_IN_BYTES(tensor_iter, n) \
    tensor_iter.current_offset_in_bytes = int(n)

/** Get the offset of a VectorIterator
 *
 * @param[in] vector_iter The VectorIterator object pointed to the starting position of the buffer
 * @param[in] x           Relative X position
 *
 * @return The relative offset of the VectorIterator object (in bytes)
 */
uint vector_offset_in_bytes(VectorIterator vector_iter, int x)
{
    return uint(vector_iter.current_offset_in_bytes + x * vector_iter.stride_x);
}

/** Get the offset of an ImageIterator
 *
 * @param[in] vector_iter The ImageIterator object pointed to the starting position of the buffer
 * @param[in] x           Relative X position
 * @param[in] y           Relative Y position
 *
 * @return The relative offset of the ImageIterator object (in bytes)
 */
uint image_offset_in_bytes(ImageIterator image_iter, int x, int y)
{
    return uint(image_iter.current_offset_in_bytes + x * image_iter.stride_x + y * image_iter.stride_y);
}

/** Get the offset of a Tensor3DIterator
 *
 * @param[in] vector_iter The Tensor3DIterator object pointed to the starting position of the buffer
 * @param[in] x           Relative X position
 * @param[in] y           Relative Y position
 * @param[in] z           Relative Z position
 *
 * @return The relative offset of the Tensor3DIterator object (in bytes)
 */
uint tensor3D_offset_in_bytes(Tensor3DIterator tensor_iter, int x, int y, int z)
{
    return uint(tensor_iter.current_offset_in_bytes + x * tensor_iter.stride_x + y * tensor_iter.stride_y + z * tensor_iter.stride_z);
}

#define LOAD(tensor_ptr, offset) tensor_ptr[offset]
#define STORE(tensor_ptr, offset, data) tensor_ptr[offset] = data
#define LOAD_CURRENT_ITEM(tensor_ptr, tensor_iter) tensor_ptr[CURRENT_ITEM_OFFSET(tensor_iter)]
#define STORE_CURRENT_ITEM(tensor_ptr, tensor_iter, data) tensor_ptr[CURRENT_ITEM_OFFSET(tensor_iter)] = data

#define VLOAD2(return_type, tensor_ptr, offset) \
    return_type(LOAD(tensor_ptr, offset),       \
                LOAD(tensor_ptr, (offset) + uint(1)))

#define VSTORE2(tensor_ptr, offset, data) \
    STORE(tensor_ptr, offset, data[0]);   \
    STORE(tensor_ptr, (offset) + uint(1), data[1])

#define VLOAD2_CURRENT_ITEM(return_type, tensor_ptr, tensor_iter) VLOAD2(return_type, tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define VSTORE2_CURRENT_ITEM(tensor_ptr, tensor_iter, data) VSTORE2(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

#define VLOAD3(return_type, tensor_ptr, offset)       \
    return_type(LOAD(tensor_ptr, offset),             \
                LOAD(tensor_ptr, (offset) + uint(1)), \
                LOAD(tensor_ptr, (offset) + uint(2)))

#define VSTORE3(tensor_ptr, offset, data)           \
    STORE(tensor_ptr, offset, data[0]);             \
    STORE(tensor_ptr, (offset) + uint(1), data[1]); \
    STORE(tensor_ptr, (offset) + uint(2), data[2])

#define VLOAD3_CURRENT_ITEM(return_type, tensor_ptr, tensor_iter) VLOAD3(return_type, tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define VSTORE3_CURRENT_ITEM(tensor_ptr, tensor_iter, data) VSTORE3(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

#define VLOAD4(return_type, tensor_ptr, offset)       \
    return_type(LOAD(tensor_ptr, offset),             \
                LOAD(tensor_ptr, (offset) + uint(1)), \
                LOAD(tensor_ptr, (offset) + uint(2)), \
                LOAD(tensor_ptr, (offset) + uint(3)))

#define VSTORE4(tensor_ptr, offset, data)           \
    STORE(tensor_ptr, offset, data[0]);             \
    STORE(tensor_ptr, (offset) + uint(1), data[1]); \
    STORE(tensor_ptr, (offset) + uint(2), data[2]); \
    STORE(tensor_ptr, (offset) + uint(3), data[3])

#define VLOAD4_CURRENT_ITEM(return_type, tensor_ptr, tensor_iter) VLOAD4(return_type, tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define VSTORE4_CURRENT_ITEM(tensor_ptr, tensor_iter, data) VSTORE4(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

#define VLOAD5(return_type, tensor_ptr, offset)       \
    return_type(LOAD(tensor_ptr, offset),             \
                LOAD(tensor_ptr, (offset) + uint(1)), \
                LOAD(tensor_ptr, (offset) + uint(2)), \
                LOAD(tensor_ptr, (offset) + uint(3)), \
                LOAD(tensor_ptr, (offset) + uint(4)))

#define VSTORE5(tensor_ptr, offset, data)           \
    STORE(tensor_ptr, offset, data[0]);             \
    STORE(tensor_ptr, (offset) + uint(1), data[1]); \
    STORE(tensor_ptr, (offset) + uint(2), data[2]); \
    STORE(tensor_ptr, (offset) + uint(3), data[3]); \
    STORE(tensor_ptr, (offset) + uint(4), data[4])

#define VLOAD5_CURRENT_ITEM(return_type, tensor_ptr, tensor_iter) VLOAD5(return_type, tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define VSTORE5_CURRENT_ITEM(tensor_ptr, tensor_iter, data) VSTORE5(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

/** Converting the vec4 object to 4 half-precision (16-bits) floating point values and packing into a uvec2 object
 *
 * @param[in] data The vec4 object to be packed
 *
 * @return The packed uvec2 object
 */
highp uvec2 pack4_half(mediump vec4 data)
{
    return uvec2(packHalf2x16(data.xy), packHalf2x16(data.zw));
}

/** Unpacking the uvec2 object to 4 half-precision (16-bits) floating point values and converting to a vec4 object
 *
 * @param[in] packed_data The uvec2 object to be unpacked
 *
 * @return The unpacked vec4 object
 */
mediump vec4 unpack4_half(highp uvec2 packed_data)
{
    return vec4(unpackHalf2x16(packed_data.x), unpackHalf2x16(packed_data.y));
}

/** Unpacking the uvec3 object to 6 half-precision (16-bits) floating point values and converting to a vec2[3] object
 *
 * @param[in] packed_data The uvec3 object to be unpacked
 *
 * @return The unpacked vec2[3] object
 */
mediump vec2[3] unpack6_half(highp uvec3 packed_data)
{
    return vec2[3](unpackHalf2x16(packed_data[0]),
                   unpackHalf2x16(packed_data[1]),
                   unpackHalf2x16(packed_data[2]));
}

/** Converting the vec4[2] object to 8 half-precision (16-bits) floating point values and packing into a uvec4 object
 *
 * @param[in] data The vec4[2] object to be packed
 *
 * @return The packed uvec4 object
 */
highp uvec4 pack8_half(mediump vec4 data[2])
{
    return uvec4(packHalf2x16(data[0].xy), packHalf2x16(data[0].zw),
                 packHalf2x16(data[1].xy), packHalf2x16(data[1].zw));
}

/** Unpacking the uvec4 object to 8 half-precision (16-bits) floating point values and converting to a vec4[2] object
 *
 * @param[in] packed_data The uvec4 object to be unpacked
 *
 * @return The unpacked vec4[2] object
 */
mediump vec4[2] unpack8_half(highp uvec4 packed_data)
{
    return vec4[2](vec4(unpackHalf2x16(packed_data.x), unpackHalf2x16(packed_data.y)),
                   vec4(unpackHalf2x16(packed_data.z), unpackHalf2x16(packed_data.w)));
}

/** Unpacking the uvec2[3] object to 12 half-precision (16-bits) floating point values and converting to a vec4[3] object
 *
 * @param[in] packed_data The uvec2[3] object to be unpacked
 *
 * @return The unpacked vec4[3] object
 */
mediump vec4[3] unpack12_half(highp uvec2[3] packed_data)
{
    return vec4[3](vec4(unpackHalf2x16(packed_data[0].x), unpackHalf2x16(packed_data[0].y)),
                   vec4(unpackHalf2x16(packed_data[1].x), unpackHalf2x16(packed_data[1].y)),
                   vec4(unpackHalf2x16(packed_data[2].x), unpackHalf2x16(packed_data[2].y)));
}

// For half-precision (16-bits) floating point packed into a "uint" element
#define LOAD_UNPACK2_HALF(tensor_ptr, offset) unpackHalf2x16(uint(LOAD(tensor_ptr, offset)))
#define STORE_PACK2_HALF(tensor_ptr, offset, data) STORE(tensor_ptr, offset, packHalf2x16(data))
#define LOAD_UNPACK2_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter) LOAD_UNPACK2_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define STORE_PACK2_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter, data) STORE_PACK2_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

#define VLOAD2_UNPACK4_HALF(tensor_ptr, offset) unpack4_half(VLOAD2(uvec2, tensor_ptr, offset))
#define VSTORE2_PACK4_HALF(tensor_ptr, offset, data) VSTORE2(tensor_ptr, offset, pack4_half(data))
#define VLOAD2_UNPACK4_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter) VLOAD2_UNPACK4_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define VSTORE2_PACK4_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter, data) VSTORE2_PACK4_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

#define VLOAD3_UNPACK6_HALF(tensor_ptr, offset) unpack6_half(VLOAD3(uvec3, tensor_ptr, offset))
#define VLOAD3_UNPACK6_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter) VLOAD3_UNPACK6_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))

#define VLOAD4_UNPACK8_HALF(tensor_ptr, offset) unpack8_half(VLOAD4(uvec4, tensor_ptr, offset))
#define VSTORE4_PACK8_HALF(tensor_ptr, offset, data) VSTORE4(tensor_ptr, offset, pack8_half(data))
#define VLOAD4_UNPACK8_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter) VLOAD4_UNPACK8_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define VSTORE4_PACK8_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter, data) VSTORE4_PACK8_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

// For half-precision (16-bits) floating point packed into a "uvec2" element
#define LOAD_UNPACK4_HALF(tensor_ptr, offset) unpack4_half(uvec2(LOAD(tensor_ptr, offset)))
#define STORE_PACK4_HALF(tensor_ptr, offset, data) STORE(tensor_ptr, offset, pack4_half(data))
#define LOAD_UNPACK4_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter) LOAD_UNPACK4_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define STORE_PACK4_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter, data) STORE_PACK4_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

#define VLOAD2_UNPACK8_HALF(tensor_ptr, offset) unpack8_half(VLOAD2(uvec4, tensor_ptr, offset))
#define VSTORE2_PACK8_HALF(tensor_ptr, offset, data) VSTORE2(tensor_ptr, offset, pack8_half(data))
#define VLOAD2_UNPACK8_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter) VLOAD2_UNPACK8_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define VSTORE2_PACK8_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter, data) VSTORE2_PACK8_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

#define VLOAD3_UNPACK12_HALF(tensor_ptr, offset) unpack12_half(VLOAD3(uvec2[3], tensor_ptr, offset))
#define VLOAD3_UNPACK12_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter) VLOAD3_UNPACK12_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))

// For half-precision (16-bits) floating point packed into a "uvec4" element
#define LOAD_UNPACK8_HALF(tensor_ptr, offset) unpack8_half(uvec4(LOAD(tensor_ptr, offset)))
#define STORE_PACK8_HALF(tensor_ptr, offset, data) STORE(tensor_ptr, offset, pack8_half(data))
#define LOAD_UNPACK8_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter) LOAD_UNPACK8_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define STORE_PACK8_CURRENT_ITEM_HALF(tensor_ptr, tensor_iter, data) STORE_PACK8_HALF(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

/** Converting the uvec4 object to 4 low-precision uint values and packing into a uint object
 *
 * @param[in] data The uvec4 object to be packed
 *
 * @return The packed uint object
 */
highp uint pack4_u8(lowp uvec4 data)
{
    highp uint r = uint(0);

    for(int i = 0; i < 4; i++)
    {
        r |= data[i] << uint(i * 8);
    }

    return r;
}

/** Unpacking the uint object to 4 low-precision uint values and converting to a uvec4 object
 *
 * @param[in] packed_data The uint object to be unpacked
 *
 * @return The unpacked uvec4 object
 */
lowp uvec4 unpack4_u8(highp uint packed_data)
{
    lowp uvec4 uvec;

    for(int i = 0; i < 4; i++)
    {
        uvec[i] = (packed_data >> uint(i * 8)) & uint(0xFF);
    }

    return uvec;
}

#define LOAD_UNPACK4_U8(tensor_ptr, offset) unpack4_u8(uint(LOAD(tensor_ptr, offset)))
#define STORE_PACK4_U8(tensor_ptr, offset, data) STORE(tensor_ptr, offset, pack4_u8(data))
#define LOAD_UNPACK4_CURRENT_ITEM_U8(tensor_ptr, tensor_iter) LOAD_UNPACK4_U8(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter))
#define STORE_PACK4_CURRENT_ITEM_U8(tensor_ptr, tensor_iter, data) STORE_PACK4_U8(tensor_ptr, CURRENT_ITEM_OFFSET(tensor_iter), data)

#endif // ARM_COMPUTE_HELPER_CS_H
