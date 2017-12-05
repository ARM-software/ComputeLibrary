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
#include "helpers.h"

layout(std140) uniform shader_params
{
#ifdef IM2COL_GENERIC
    TENSOR3D_PARAM_DECLARATION(src);
    IMAGE_PARAM_DECLARATION(dst);
    uint filter_depth;
    uint src_stride_w;
    uint dst_stride_w;
#endif // IM2COL_GENERIC

#ifdef IM2COL_REDUCED
    TENSOR3D_PARAM_DECLARATION(src);
    VECTOR_PARAM_DECLARATION(dst);
    uint width;
    uint height;
#endif // IM2COL_REDUCED

#ifdef COL2IM
    IMAGE_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(dst);
    uint width;
#endif // COL2IM
};

#ifdef DATA_TYPE_FP16
#if defined(IM2COL_REDUCED_8X)
BUFFER_DECLARATION(src, 1, uvec4, readonly);
BUFFER_DECLARATION(dst, 2, uvec4, restrict);
#elif defined(IM2COL_REDUCED_4X) /* IM2COL_REDUCED_8X */
BUFFER_DECLARATION(src, 1, uvec2, readonly);
BUFFER_DECLARATION(dst, 2, uvec2, restrict);
#else                            /* IM2COL_REDUCED_8X */
BUFFER_DECLARATION(src, 1, uint, readonly);
BUFFER_DECLARATION(dst, 2, uint, restrict);
#endif                           /* IM2COL_REDUCED_8X */

precision mediump float;

#ifdef IM2COL_REDUCED
#if defined(IM2COL_REDUCED_GENERIC)
/** This kernel reshapes the tensor's low three dimensions to single row for GEMM operation
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP16"
 * @note In case biases will be added in late stage, "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  width                             The width of the input tensor
 * @param[in]  height                            The height of the input tensor
 */
void main(void)
{
    uvec3    pos            = uvec3(gl_GlobalInvocationID.xyz);
    uvec3    size           = uvec3(gl_WorkGroupSize.xyz);
    Tensor3D src            = CONVERT_TO_TENSOR3D_STRUCT_FP16(src);
    Tensor3D src_nostep     = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP_FP16(src);
    Vector   dst            = CONVERT_TO_VECTOR_STRUCT_NO_STEP_FP16(dst);
    uint     image_size     = width * height;
    uint     element_count  = src_step_x / src_stride_x;
    uint     tmp_out_offset = dst.current_offset + ((pos.x * element_count + pos.y * width + pos.z * image_size) * dst.stride_x);
    uint     width_fp16     = ((width + uint(1)) >> uint(1));
    uint     tmp;

    // odd width
    if(width % uint(2) != uint(0))
    {
        // even row
        if((pos.y + pos.z * height) % uint(2) == uint(0))
        {
            LOAD1(tmp, src, src.current_offset >> uint(2));
            STORE1(dst, tmp_out_offset >> uint(2), tmp);
        }
        else
        {
            // special op
            uint tmpleft  = uint(0);
            uint tmpright = uint(0);
            LOAD1(tmpright, src, src.current_offset >> uint(2)); // right half
            if(pos.x == uint(0))
            {
                LOAD1(tmpleft, src, tensor3D_offset_fp16(src_nostep, int(width), int(pos.y) - 1, int(pos.z)) >> uint(2)); // left half
                tmpright = (tmpleft & uint(0xffff)) + (tmpright << uint(16));
            }
            else
            {
                LOAD1(tmpleft, src, tensor3D_offset_fp16(src_nostep, (int(pos.x) - 1) * int(element_count), int(pos.y), int(pos.z)) >> uint(2)); // left half
                tmpright = ((tmpleft >> uint(16)) + (tmpright << uint(16)));
            }
            STORE1(dst, tmp_out_offset >> uint(2), tmpright);
        }
    }
    else
    {
        LOAD1(tmp, src, src.current_offset >> uint(2));
        STORE1(dst, tmp_out_offset >> uint(2), tmp);
    }

#ifdef HAS_BIAS
    // If it is the last thread in the 3 dimensional workgroup
    if(pos.x == (size.x - 1) && pos.y == (size.y - 1) && pos.z == (size.z - 1))
    {
        tmp_out_offset += dst.stride_x;

        // FIXME: need odd/even detection for tmp_out_offset?
        mediump vec2 bias_vec = vec2(1.0f, 1.0f);
        uint         bias_u   = packHalf2x16(bias_vec);
        STORE1(dst, tmp_out_offset >> uint(2), bias_u);
    }
#endif // HAS_BIAS
}
#else /* IM2COL_REDUCED_GENERIC */
/** This kernel reshapes the tensor's low three dimensions to single row for GEMM operation
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP16"
 * @note In case biases will be added in late stage, "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  width                             The width of the input tensor
 * @param[in]  height                            The height of the input tensor
 */
void main(void)
{
    uvec3    pos            = uvec3(gl_GlobalInvocationID.xyz);
    Tensor3D src            = CONVERT_TO_TENSOR3D_STRUCT_FP16(src);
    Vector   dst            = CONVERT_TO_VECTOR_STRUCT_NO_STEP_FP16(dst);
#if defined(IM2COL_REDUCED_8X)
    uint     tmp_out_offset = dst.current_offset + ((pos.x * uint(8) + pos.y * width + pos.z * uint(IMAGE_SIZE)) * dst.stride_x);
    uvec4    tmp;
    LOAD1(tmp, src, src.current_offset >> uint(4));
    STORE1(dst, tmp_out_offset >> uint(4), tmp);
#elif defined(IM2COL_REDUCED_4X) /* IM2COL_REDUCED_8X */
    uint  tmp_out_offset = dst.current_offset + ((pos.x * uint(4) + pos.y * width + pos.z * uint(IMAGE_SIZE)) * dst.stride_x);
    uvec2 tmp;
    LOAD1(tmp, src, src.current_offset >> uint(3));
    STORE1(dst, tmp_out_offset >> uint(3), tmp);
#else                            /* IM2COL_REDUCED_8X */
    uint tmp_out_offset = dst.current_offset + ((pos.x * uint(2) + pos.y * width + pos.z * uint(IMAGE_SIZE)) * dst.stride_x);
    uint tmp;
    LOAD1(tmp, src, src.current_offset >> uint(2));
    STORE1(dst, tmp_out_offset >> uint(2), tmp);
#endif                           /* IM2COL_REDUCED_8X */
}
#endif                           /* IM2COL_REDUCED_GENERIC */
#endif                           // IM2COL_REDUCED

#elif defined(DATA_TYPE_FP32)
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, restrict);

#ifdef IM2COL_GENERIC
/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 * @note In case biases will be added to the convolution "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  filter_depth                      The depth of the used filter
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes).
 */
void main(void)
{
    uint xc    = gl_GlobalInvocationID.x;                // x coordinate in the convolved tensor
    uint yc    = gl_GlobalInvocationID.y;                // y coordinate in the convolved tensor
    uint ch    = gl_GlobalInvocationID.z % filter_depth; // input feature map
    uint batch = gl_GlobalInvocationID.z / filter_depth; // the batch

    // Calculate input indeces
    uint xi           = xc * uint(STRIDE_X) - uint(PAD_X);
    uint yi           = yc * uint(STRIDE_Y) - uint(PAD_Y);
    uint input_offset = (src_offset_first_element_in_bytes + (ch * src_stride_z) + (batch * src_stride_w)) >> uint(2);

    // Calculate output indeces
    uint xo            = ch * uint(KERNEL_WIDTH) * uint(KERNEL_HEIGHT);
    uint yo            = xc + yc * uint(CONVOLVED_WIDTH); // Index of the convolution
    uint output_offset = (dst_offset_first_element_in_bytes + (yo * dst_stride_y) + (batch * dst_stride_w) + xo) >> uint(2);

    // Linearize convolution elements
    for(uint y = yi, y_e = yi + uint(KERNEL_HEIGHT); y < y_e; ++y)
    {
        for(uint x = xi, x_e = xi + uint(KERNEL_WIDTH); x < x_e; ++x)
        {
#if PAD_X == 0 && PAD_Y == 0
            output_offset = input_offset + ((x * src_stride_x + y * src_stride_y) >> uint(2));
            STORE4(dst, output_offset, LOAD4(src, input_offset));
#else  // PAD_X == 0 && PAD_Y == 0
            if(x < 0 || x >= SRC_WIDTH || y < 0 || y >= SRC_HEIGHT)
            {
                STORE4(dst, output_offset, 0.0f);
            }
            else
            {
                output_offset = input_offset + (x * src_stride_x + y * src_stride_y) >> uint(2));
                STORE4(dst, output_offset, LOAD4(src, input_offset));
            }
#endif // PAD_X == 0 && PAD_Y == 0
        }
    }

#ifdef HAS_BIAS
    if(ch == (uint(KERNEL_DEPTH) - 1))
    {
        STORE4(dst, output_offset, 1.0f);
    }
#endif // HAS_BIAS
}
#endif // IM2COL_GENERIC

#ifdef IM2COL_REDUCED
/** This kernel reshapes the tensor's low three dimensions to single row for GEMM operation
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 * @note In case biases will be added in late stage, "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  width                             The width of the input tensor
 * @param[in]  height                            The height of the input tensor
 */
void main(void)
{
    uvec3    pos            = uvec3(gl_GlobalInvocationID.xyz);
    uvec3    size           = uvec3(gl_WorkGroupSize.xyz);
    Tensor3D src            = CONVERT_TO_TENSOR3D_STRUCT(src);
    Vector   dst            = CONVERT_TO_VECTOR_STRUCT_NO_STEP(dst);
    uint     image_size     = width * height;
    uint     tmp_out_offset = dst.current_offset + (((pos.x + pos.y * width + pos.z * image_size) * dst.stride_x) >> 2);

    STORE4(dst, tmp_out_offset, LOAD4(src, src.current_offset));

#ifdef HAS_BIAS
    // If it is the last thread in the 3 dimensional workgroup
    if(pos.x == (size.x - 1) && pos.y == (size.y - 1) && pos.z == (size.z - 1))
    {
        tmp_out_offset += (dst.stride_x >> uint(2));
        STORE4(dst, tmp_out_offset, 1.f);
    }
#endif // HAS_BIAS
}
#endif // IM2COL_REDUCED

#ifdef COL2IM
/** This kernel performs a reshaping of the output of the convolution layer.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
void main(void)
{
    uvec2    pos = uvec2(gl_GlobalInvocationID.xy);
    Image    src = CONVERT_TO_IMAGE_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    uint idx            = pos.x * dst.stride_z + (pos.y / width) * dst.stride_y + (pos.y % width) * dst.stride_x;
    uint tmp_out_offset = dst.current_offset + (idx >> 2);

    STORE4(dst, tmp_out_offset, LOAD4(src, src.current_offset));
}
#endif // COL2IM

#else // DATA_TYPE_FP16
#error Data type not supported
#endif // DATA_TYPE_FP16
