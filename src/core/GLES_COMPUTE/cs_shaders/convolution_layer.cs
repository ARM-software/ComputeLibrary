/*
 * Copyright (c) 2017, 2018 ARM Limited.
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

#if defined(DATA_TYPE_FP16)
precision mediump float;
#endif // DATA_TYPE_FP16

#ifdef IM2COL_GENERIC
/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 * @note In case biases will be added to the convolution "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr      Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs    The attributes of the source tensor
 * @param[out] dst_ptr      Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs    The attributes of the destination tensor
 * @param[in]  filter_depth The depth of the used filter
 * @param[in]  src_stride_w Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w Stride of the destination tensor in W dimension (in bytes).
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    ImageAttributes    dst_attrs;
    uint               filter_depth;
    uint               src_stride_w;
    uint               dst_stride_w;
};

#ifdef DATA_TYPE_FP32
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, restrict);
void main(void)
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(src_attrs, src_shift);
    ImageIterator    dst_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    uint xc    = gl_GlobalInvocationID.x;                // x coordinate in the convolved tensor
    uint yc    = gl_GlobalInvocationID.y;                // y coordinate in the convolved tensor
    uint ch    = gl_GlobalInvocationID.z % filter_depth; // input feature map
    uint batch = gl_GlobalInvocationID.z / filter_depth; // the batch

    // Calculate input indeces
    uint xi           = xc * uint(STRIDE_X) - uint(PAD_X);
    uint yi           = yc * uint(STRIDE_Y) - uint(PAD_Y);
    uint input_offset = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (ch * src_attrs.stride_z) + (batch * src_stride_w));

    // Calculate output indeces
    uint xo            = ch * uint(KERNEL_WIDTH) * uint(KERNEL_HEIGHT);
    uint yo            = xc + yc * uint(CONVOLVED_WIDTH); // Index of the convolution
    uint output_offset = TENSOR_OFFSET_ADVANCE_IN_BYTES(dst_iter, (yo * dst_attrs.stride_y) + (batch * dst_stride_w) + xo);

    // Linearize convolution elements
    for(uint y = yi, y_e = yi + uint(KERNEL_HEIGHT); y < y_e; ++y)
    {
        for(uint x = xi, x_e = xi + uint(KERNEL_WIDTH); x < x_e; ++x)
        {
#if PAD_X == 0 && PAD_Y == 0
            output_offset = input_offset + ((x * src_attrs.stride_x + y * src_attrs.stride_y) >> uint(2));
            STORE(dst_ptr, output_offset, LOAD(src_ptr, input_offset));

#else  // PAD_X == 0 && PAD_Y == 0
            if(x < 0 || x >= SRC_WIDTH || y < 0 || y >= SRC_HEIGHT)
            {
                STORE(dst_ptr, output_offset, 0.0f);
            }
            else
            {
                output_offset = input_offset + (x * srcs_attrs.stride_x + y * src_attrs.stride_y) >> uint(2));
                STORE(dst_ptr, output_offset, LOAD(src_ptr, input_offset));
            }
#endif // PAD_X == 0 && PAD_Y == 0
        }
    }

#ifdef HAS_BIAS
    if(ch == (uint(KERNEL_DEPTH) - 1))
    {
        STORE(dst_ptr, output_offset, 1.0f);
    }
#endif // HAS_BIAS
}

#elif defined(DATA_TYPE_FP16)
TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
}

#else /* DATA_TYPE_FP32 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
#endif /* IM2COL_GENERIC */

#ifdef IM2COL_REDUCED
/** This kernel reshapes the tensor's low three dimensions to single row for GEMM operation
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP16"
 * @note In case biases will be added in late stage, "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr   Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs The attributes of the source tensor
 * @param[out] dst_ptr   Pointer to the destination tensor. Same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination tensor
 * @param[in]  width     The width of the input tensor
 * @param[in]  height    The height of the input tensor
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    VectorAttributes   dst_attrs;
    uint               width;
    uint               height;
};

#ifdef DATA_TYPE_FP32
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, restrict);

void main(void)
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    VectorIterator   dst_iter = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    uvec3 pos            = uvec3(gl_GlobalInvocationID.xyz);
    uvec3 size           = uvec3(gl_WorkGroupSize.xyz);
    uint  image_size     = width * height;
    uint  tmp_out_offset = VECTOR_OFFSET(dst_iter, pos.x + pos.y * width + pos.z * image_size);

    STORE(dst_ptr, tmp_out_offset, LOAD_CURRENT_ITEM(src_ptr, src_iter));

#ifdef HAS_BIAS
    // If it is the last thread in the 3 dimensional workgroup
    if(pos.x == (size.x - 1) && pos.y == (size.y - 1) && pos.z == (size.z - 1))
    {
        tmp_out_offset += (dst_attrs.stride_x >> uint(2));
        STORE(dst_ptr, tmp_out_offset, 1.f);
    }
#endif // HAS_BIAS
}

#elif defined(DATA_TYPE_FP16)

#if defined(IM2COL_REDUCED_8X)
TENSOR_DECLARATION(1, srcBuffer, uvec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec4, dst_ptr, dst_shift, 4, restrict);
#elif defined(IM2COL_REDUCED_4X) /* IM2COL_REDUCED_8X */
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, restrict);
#else                            /* IM2COL_REDUCED_8X */
TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, uint, dst_ptr, dst_shift, 2, restrict);
#endif                           /* IM2COL_REDUCED_8X */

#if defined(IM2COL_REDUCED_GENERIC)
void main(void)
{
    Tensor3DIterator src_iter        = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator src_nostep_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(src_attrs, src_shift);
    VectorIterator   dst_iter        = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    uvec3 pos            = uvec3(gl_GlobalInvocationID.xyz);
    uvec3 size           = uvec3(gl_WorkGroupSize.xyz);
    uint  image_size     = width * height;
    uint  element_count  = src_attrs.step_x / src_attrs.stride_x;
    uint  tmp_out_offset = VECTOR_OFFSET(dst_iter, pos.x * element_count + pos.y * width + pos.z * image_size);
    uint  width_fp16     = (width + uint(1)) >> uint(1);
    uint  tmp;

    // odd width
    if(width % uint(2) != uint(0))
    {
        // even row
        if((pos.y + pos.z * height) % uint(2) == uint(0))
        {
            tmp = LOAD_CURRENT_ITEM(src_ptr, src_iter);
            STORE(dst_ptr, tmp_out_offset, tmp);
        }
        else
        {
            // special op
            uint tmpleft  = uint(0);
            uint tmpright = uint(0);
            tmpright      = LOAD_CURRENT_ITEM(src_ptr, src_iter); //right half
            if(pos.x == uint(0))
            {
                tmpleft  = LOAD(src_ptr, TENSOR3D_OFFSET(src_nostep_iter, int(width), int(pos.y) - 1, int(pos.z))); //left half
                tmpright = (tmpleft & uint(0xffff)) + (tmpright << uint(16));
            }
            else
            {
                tmpleft  = LOAD(src_ptr, TENSOR3D_OFFSET(src_nostep_iter, (int(pos.x) - 1) * int(element_count), int(pos.y), int(pos.z)));
                tmpright = ((tmpleft >> uint(16)) + (tmpright << uint(16)));
            }
            STORE(dst_ptr, tmp_out_offset, tmpright);
        }
    }
    else
    {
        tmp = LOAD_CURRENT_ITEM(src_ptr, src_iter);
        STORE(dst_ptr, tmp_out_offset, tmp);

#ifdef HAS_BIAS
        // If it is the last thread in the 3 dimensional workgroup
        if(pos.x == (size.x - 1) && pos.y == (size.y - 1) && pos.z == (size.z - 1))
        {
            tmp_out_offset += (dst_attrs.stride_x >> dst_shift);

            // FIXME: need odd/even detection for tmp_out_offset?
            mediump vec2 bias_vec = vec2(1.0f, 1.0f);
            STORE_PACK2_HALF(dst_ptr, tmp_out_offset, bias_vec);
        }
#endif // HAS_BIAS
    }
}

#else /* IM2COL_REDUCED_GENERIC */
void main(void)
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    VectorIterator   dst_iter = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    uvec3 pos            = uvec3(gl_GlobalInvocationID.xyz);
#if defined(IM2COL_REDUCED_8X)
    uint  tmp_out_offset = VECTOR_OFFSET(dst_iter, pos.x * uint(8) + pos.y * width + pos.z * uint(IMAGE_SIZE));
    uvec4 tmp            = LOAD_CURRENT_ITEM(src_ptr, src_iter);
    STORE(dst_ptr, tmp_out_offset, tmp);
#elif defined(IM2COL_REDUCED_4X) /* IM2COL_REDUCED_8X */
    uint  tmp_out_offset = VECTOR_OFFSET(dst_iter, pos.x * uint(4) + pos.y * width + pos.z * uint(IMAGE_SIZE));
    uvec2 tmp            = LOAD_CURRENT_ITEM(src_ptr, src_iter);
    STORE(dst_ptr, tmp_out_offset, tmp);
#else                            /* IM2COL_REDUCED_8X */
    uint tmp_out_offset = VECTOR_OFFSET(dst_iter, pos.x * uint(2) + pos.y * width + pos.z * uint(IMAGE_SIZE));
    uint tmp            = LOAD_CURRENT_ITEM(src_ptr, src_iter);
    STORE(dst_ptr, tmp_out_offset, tmp);
#endif                           /* IM2COL_REDUCED_8X */
}
#endif                           /* IM2COL_REDUCED_GENERIC */
#else                            /* DATA_TYPE_FP32 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
#endif /* IM2COL_REDUCED */

#ifdef COL2IM
/** This kernel performs a reshaping of the output of the convolution layer.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 *
 * @param[in]  src_ptr   Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs The attributes of the source tensor
 * @param[out] dst_ptr   Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination tensor
 * @param[in]  width     The width of output convolved dimensions
 */
SHADER_PARAMS_DECLARATION
{
    ImageAttributes    src_attrs;
    Tensor3DAttributes dst_attrs;
    uint               width;
};

#ifdef DATA_TYPE_FP32
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, restrict);
void main(void)
{
    ImageIterator    src_iter = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    uvec2 pos            = uvec2(gl_GlobalInvocationID.xy);
    uint  tmp_out_offset = TENSOR3D_OFFSET(dst_iter, pos.y % width, pos.y / width, pos.x);

    STORE(dst_ptr, tmp_out_offset, LOAD_CURRENT_ITEM(src_ptr, src_iter));
}

#elif defined(DATA_TYPE_FP16)

#else /* DATA_TYPE_FP32 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
#endif /* COL2IM */
