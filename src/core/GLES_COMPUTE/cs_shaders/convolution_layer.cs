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

layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;

#include "helpers_cs.h"

#if defined(DATA_TYPE_FP16)
precision mediump float;
#endif // DATA_TYPE_FP16

#ifdef RESHAPE_TO_COLUMNS

/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note In case biases will be added to the convolution "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr       Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs     The attributes of the source tensor
 * @param[out] dst_ptr       Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs     The attributes of the destination tensor
 * @param[in]  biases_ptr    Pointer to the biases tensor. Same as @p src_ptr
 * @param[in]  biases_attrs  The attributes of the biases tensor
 * @param[in]  width         The width of the input tensor
 * @param[in]  height        The height of the input tensor
 * @param[in]  depth         The depth of the input tensor
 * @param[in]  total_filters Total number of filters. 4th dimension of the weights matrix
 */

SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    ImageAttributes    dst_attrs;
#ifdef HAS_BIAS
    VectorAttributes biases_attrs;
#endif /* HAS_BIAS */
    uint width;
    uint height;
    uint depth;
    uint total_filters;
};

#if defined(DATA_TYPE_FP16)

TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);
#ifdef HAS_BIAS
TENSOR_DECLARATION(3, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

void main()
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    ImageIterator    dst_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);
#ifdef HAS_BIAS
    VectorIterator biases_iter = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    bool is_last_thread = (((int(gl_GlobalInvocationID.x)) == (int(gl_NumWorkGroups.x * gl_WorkGroupSize.x) - 1)) && ((int(gl_GlobalInvocationID.y)) == (int(gl_NumWorkGroups.y * gl_WorkGroupSize.y) - 1))
                           && ((int(gl_GlobalInvocationID.z)) == (int(gl_NumWorkGroups.z * gl_WorkGroupSize.z) - 1)));
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, ((uint(gl_GlobalInvocationID.x) * uint(dst_attrs.stride_y)) + (uint(gl_GlobalInvocationID.y) * uint(width) * uint(dst_attrs.stride_y)) + (uint(
                                                    gl_GlobalInvocationID.z)
                                                * uint(width) * uint(height) * uint(dst_attrs.stride_y))));
    // Linearize convolution elements
    if(is_last_thread)
    {
        for(uint i = 0u; i < uint(total_filters); i = i + 2u)
        {
            vec2 s0 = LOAD_UNPACK2_CURRENT_ITEM_HALF(src_ptr, src_iter);
            vec2 s;
            if(int(CURRENT_ITEM_OFFSET_IN_BYTES(src_iter) >> 1u) % 2 == 0)
            {
                s.x = s0.x;
            }
            else
            {
                s.x = s0.y;
            }
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, (depth * src_attrs.stride_z));

            vec2 s1 = LOAD_UNPACK2_CURRENT_ITEM_HALF(src_ptr, src_iter);
            if(int(CURRENT_ITEM_OFFSET_IN_BYTES(src_iter) >> 1u) % 2 == 0)
            {
                s.y = s1.x;
            }
            else
            {
                s.y = s1.y;
            }
            STORE_PACK2_CURRENT_ITEM_HALF(dst_ptr, dst_iter, s);
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, (depth * src_attrs.stride_z));
#ifdef HAS_BIAS
            vec2 b = LOAD_UNPACK2_CURRENT_ITEM_HALF(biases_ptr, biases_iter);
            STORE_PACK2_HALF(dst_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(dst_iter, dst_attrs.stride_y), b);
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(biases_iter, (2u * biases_attrs.stride_x));
#endif /* HAS_BIAS */
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, (2u * dst_attrs.stride_x));
        }
    }
    else
    {
        for(uint i = 0u; i < uint(total_filters); i = i + 2u)
        {
            vec2 s0 = LOAD_UNPACK2_CURRENT_ITEM_HALF(src_ptr, src_iter);
            vec2 s;
            if(int(CURRENT_ITEM_OFFSET_IN_BYTES(src_iter) >> 1u) % 2 == 0)
            {
                s.x = s0.x;
            }
            else
            {
                s.x = s0.y;
            }
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, (depth * src_attrs.stride_z));

            vec2 s1 = LOAD_UNPACK2_CURRENT_ITEM_HALF(src_ptr, src_iter);
            if(int(CURRENT_ITEM_OFFSET_IN_BYTES(src_iter) >> 1u) % 2 == 0)
            {
                s.y = s1.x;
            }
            else
            {
                s.y = s1.y;
            }
            STORE_PACK2_CURRENT_ITEM_HALF(dst_ptr, dst_iter, s);
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, (depth * src_attrs.stride_z));
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, (2u * dst_attrs.stride_x));
        }
    }
}

#endif /* DATA_TYPE_FP16 */
#endif // RESHAPE_TO_COLUMNS

#ifdef IM2COL_GENERIC

/** This kernel performs a reshaping of the input tensor to a tensor used to perform convolution using GEMM.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 * @note PAD_LEFT/PAD_RIGHT/PAD_TOP/PAD_BOTTOM must be passed for padding info, e.g. "#define PAD_LEFT xxx"
 * @note KERNEL_WIDTH/KERNEL_HEIGHT/KERNEL_DEPTH must be passed for kernel dimension, e.g. "#define KERNEL_WIDTH xxx"
 * @note STRIDE_X/STRIDE_Y must be passed for stride info, e.g. "#define STRIDE_X xxx"
 * @note CONVOLVED_WIDTH/CONVOLVED_HEIGHT must be passed for convolved dimension, e.g. "#define CONVOLVED_WIDTH xxx"
 * @note SRC_WIDTH/SRC_HEIGHT must be passed for input dimension, e.g. "#define SRC_WIDTH xxx"
 * @note In case biases will be added to the convolution "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr      Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs    The attributes of the source tensor
 * @param[out] dst_ptr      Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs    The attributes of the destination tensor
 * @param[in]  src_stride_w Stride of the source tensor in W dimension (in bytes).
 * @param[in]  dst_stride_w Stride of the destination tensor in W dimension (in bytes).
 */

SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    ImageAttributes    dst_attrs;
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
    uint ch    = gl_GlobalInvocationID.z % KERNEL_DEPTH; // input feature map
    uint batch = gl_GlobalInvocationID.z / KERNEL_DEPTH; // the batch

    // Calculate input indeces
    uint xi = xc * uint(STRIDE_X) - uint(PAD_LEFT);
    uint yi = yc * uint(STRIDE_Y) - uint(PAD_TOP);
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, (ch * src_attrs.stride_z) + (batch * src_stride_w));

    // Calculate output indeces
    uint xo = ch * uint(KERNEL_WIDTH) * uint(KERNEL_HEIGHT);
    uint yo = xc + yc * uint(CONVOLVED_WIDTH); // Index of the convolution
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, (yo * dst_attrs.stride_y) + (batch * dst_stride_w) + xo);

    uint src_pos = 0u;

    // Linearize convolution elements
    for(uint y = yi, y_e = yi + uint(KERNEL_HEIGHT); y < y_e; ++y)
    {
        for(uint x = xi, x_e = xi + uint(KERNEL_WIDTH); x < x_e; ++x, TENSOR_OFFSET_ADVANCE(dst_iter, 1u))
        {
#if PAD_LEFT == 0 && PAD_TOP == 0 && PAD_RIGHT == 0 && PAD_BOTTOM == 0
            src_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, x * src_attrs.stride_x + y * src_attrs.stride_y);
            STORE_CURRENT_ITEM(dst_ptr, dst_iter, LOAD(src_ptr, src_pos));
#else  /* PAD_LEFT == 0 && PAD_TOP == 0 && PAD_RIGHT == 0 && PAD_BOTTOM == 0 */
            if(x < 0 || x >= SRC_WIDTH || y < 0 || y >= SRC_HEIGHT)
            {
                STORE_CURRENT_ITEM(dst_ptr, dst_iter, 0.0f);
            }
            else
            {
                src_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, x * src_attrs.stride_x + y * src_attrs.stride_y);
                STORE_CURRENT_ITEM(dst_ptr, dst_iter, LOAD(src_ptr, src_pos));
            }
#endif /* PAD_LEFT == 0 && PAD_TOP == 0 && PAD_RIGHT == 0 && PAD_BOTTOM == 0 */
        }
    }

#ifdef HAS_BIAS
    if(ch == (uint(KERNEL_DEPTH) - 1))
    {
        STORE_CURRENT_ITEM(dst_ptr, dst_iter, 1.0f);
    }
#endif /* HAS_BIAS */
}

#elif defined(DATA_TYPE_FP16)

TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);

#ifdef KERNEL_1x1

void main(void)
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(src_attrs, src_shift);
    ImageIterator    dst_iter = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    uint xc    = gl_GlobalInvocationID.x;
    uint yc    = gl_GlobalInvocationID.y;
    uint zc    = gl_GlobalInvocationID.z;
    uint ch    = zc % uint(KERNEL_DEPTH); // input feature map
    uint batch = zc / uint(KERNEL_DEPTH); // the batch

    // Calculate input indeces
    uint xi = xc;
    uint yi = yc;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, batch * src_stride_w + ch * src_attrs.step_z);

    // Calculate output indeces
    uint dst_element_count = dst_attrs.step_x / dst_attrs.stride_x;
    uint xo                = ch * dst_element_count;
    uint yo                = xc + yc * uint(CONVOLVED_WIDTH);
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, batch * dst_stride_w + yo * dst_attrs.stride_y + xo);

    bool x_start_even = ((xc % 2u) == 0u);
    bool z_depth_even = ((uint(KERNEL_DEPTH) % 2u) == 0u);
    uint input_pos    = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, xi * src_attrs.stride_x + yi * src_attrs.stride_y);
    uint tmp_left     = 0u;
    uint tmp_right    = 0u;

    if(ch % 2u != 0u)
    {
        return;
    }

    if(z_depth_even || (!z_depth_even && (int(ch) < (KERNEL_DEPTH - 1))))
    {
        tmp_left  = LOAD(src_ptr, input_pos);
        input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, xi * src_attrs.stride_x + yi * src_attrs.stride_y + src_attrs.stride_z);
        tmp_right = LOAD(src_ptr, input_pos);
        if(x_start_even)
        {
            tmp_right = (tmp_left & 0xffffu) + (tmp_right << 16u);
        }
        else
        {
            tmp_right = (tmp_left >> 16u) + (tmp_right & 0xffff0000u);
        }
        STORE_CURRENT_ITEM(dst_ptr, dst_iter, tmp_right);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_attrs.step_x);

#ifdef HAS_BIAS
        if(ch == (uint(KERNEL_DEPTH) - 2u))
        {
            mediump vec2 bias_vec = vec2(1.f, 0.f);
            uint         bias_u   = packHalf2x16(bias_vec);
            STORE_CURRENT_ITEM(dst_ptr, dst_iter, bias_u);
        }
#endif /* HAS_BIAS */
    }
    else
    {
        tmp_left = LOAD(src_ptr, input_pos);
        if(x_start_even)
        {
            tmp_right = (tmp_left & 0xffffu);
        }
        else
        {
            tmp_right = (tmp_left >> 16u);
        }

#ifdef HAS_BIAS
        mediump vec2 bias_vec = vec2(0.f, 1.f);
        uint         bias_u   = packHalf2x16(bias_vec);
        tmp_right += (bias_u & 0xffff0000u);
#endif /* HAS_BIAS */

        STORE_CURRENT_ITEM(dst_ptr, dst_iter, tmp_right);
    }
}

#else /* KERNEL_1x1 */

void main(void)
{
    uint xc    = gl_GlobalInvocationID.x;
    uint yc    = gl_GlobalInvocationID.y;
    uint zc    = gl_GlobalInvocationID.z;
    uint ch    = zc % uint(KERNEL_DEPTH); // input feature map
    uint batch = zc / uint(KERNEL_DEPTH); // the batch

    Tensor3DIterator src_iter   = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(src_attrs, src_shift);
    Tensor3DIterator src_iter_b = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(src_attrs, src_shift);
    ImageIterator    dst_iter   = CONVERT_TO_IMAGE_ITERATOR_NO_STEP(dst_attrs, dst_shift);

    // Calculate input indeces
    uint src_element_count = src_attrs.step_x / src_attrs.stride_x;
    uint xi                = (xc * uint(STRIDE_X)) / src_element_count;
    uint yi                = yc * uint(STRIDE_Y);
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, batch * src_stride_w + ch * src_attrs.stride_z);

    // Calculate output indeces
    uint dst_element_count = dst_attrs.step_x / dst_attrs.stride_x;
    uint xo                = (ch * uint(KERNEL_WIDTH) * uint(KERNEL_HEIGHT)) * dst_element_count;
    uint yo                = xc + yc * uint(CONVOLVED_WIDTH);
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, batch * dst_stride_w + yo * dst_attrs.stride_y + xo);

    bool x_start_even = ((xc * uint(STRIDE_X)) % 2u == 0u);
    bool z_start_even = ((ch % 2u) == 0u);
    uint input_pos    = 0u;
    uint tmp          = 0u;
    uint tmp_left     = 0u;
    uint tmp_right    = 0u;

    // Linearize convolution elements
    for(uint y = yi, y_e = yi + uint(KERNEL_HEIGHT); y < y_e; ++y)
    {
        uint xstart = 0u;
        uint xend   = 0u;

        // even col, even row
        if(x_start_even)
        {
            if(((y - yi + ch) % 2u) == 0u)
            {
                for(uint x = xi, x_e = xi + (uint(KERNEL_WIDTH) / 2u); x < x_e; ++x, TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_attrs.step_x))
                {
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, x * src_attrs.step_x + y * src_attrs.stride_y);
                    STORE_CURRENT_ITEM(dst_ptr, dst_iter, LOAD(src_ptr, input_pos));
                }
            }
            else
            {
                // 1st pair
                if(!z_start_even && (y == yi))
                {
                    // cross 2d feature map
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter_b, (xi + (uint(KERNEL_WIDTH) / 2u)) * src_attrs.step_x + (yi + uint(KERNEL_HEIGHT) - 1u) * src_attrs.stride_y + batch * src_stride_w +
                                                               (ch - 1u) * src_attrs.stride_z);
                }
                else
                {
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter,
                                                               (xi + (uint(KERNEL_WIDTH) / 2u)) * src_attrs.step_x + (y - 1u) * src_attrs.stride_y);
                }
                tmp_right = LOAD(src_ptr, input_pos);
                input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, xi * src_attrs.step_x + y * src_attrs.stride_y);
                tmp_left  = LOAD(src_ptr, input_pos);
                tmp_right = (tmp_right & 0xffffu) + (tmp_left << 16u);
                STORE_CURRENT_ITEM(dst_ptr, dst_iter, tmp_right);
                TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_attrs.step_x);

                // remaining
                for(uint x = xi + 1u, x_e = xi + (uint(KERNEL_WIDTH) / 2u) + 1u; x < x_e; ++x, TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_attrs.step_x))
                {
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (x - 1u) * src_attrs.step_x + y * src_attrs.stride_y);
                    tmp_left  = LOAD(src_ptr, input_pos);
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, x * src_attrs.step_x + y * src_attrs.stride_y);
                    tmp_right = LOAD(src_ptr, input_pos);
                    tmp_right = (tmp_left >> 16u) + (tmp_right << 16u);
                    STORE_CURRENT_ITEM(dst_ptr, dst_iter, tmp_right);
                }
            }
        }
        else
        {
            if((((y - yi) % 2u) == 0u && !z_start_even) || (((y - yi) % 2u) != 0u && z_start_even))
            {
                // 1st pair
                if(y == yi)
                {
                    // cross 2d feature map
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter_b, (xi + (uint(KERNEL_WIDTH) / 2u)) * src_attrs.step_x + (yi + uint(KERNEL_HEIGHT) - 1u) * src_attrs.stride_y + batch * src_stride_w +
                                                               (ch - 1u) * src_attrs.stride_z);
                }
                else
                {
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter,
                                                               (xi + (uint(KERNEL_WIDTH) / 2u)) * src_attrs.step_x + (y - 1u) * src_attrs.stride_y);
                }

                tmp_right = LOAD(src_ptr, input_pos);
                input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, xi * src_attrs.step_x + y * src_attrs.stride_y);
                tmp_left  = LOAD(src_ptr, input_pos);
                tmp_right = (tmp_right >> 16u) + (tmp_left & 0xffff0000u);
                STORE_CURRENT_ITEM(dst_ptr, dst_iter, tmp_right);
                TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_attrs.step_x);

                // remaining
                for(uint x = xi + 1u, x_e = xi + (uint(KERNEL_WIDTH) / 2u) + 1u; x < x_e; ++x, TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_attrs.step_x))
                {
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, x * src_attrs.step_x + y * src_attrs.stride_y);
                    STORE_CURRENT_ITEM(dst_ptr, dst_iter, LOAD(src_ptr, input_pos));
                }
            }
            else if((((y - yi) % 2u) == 0u && z_start_even) || (((y - yi) % 2u) != 0u && !z_start_even))
            {
                // 1st pair
                input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, xi * src_attrs.step_x + y * src_attrs.stride_y);
                tmp_right = LOAD(src_ptr, input_pos);
                input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (xi + 1u) * src_attrs.step_x + y * src_attrs.stride_y);
                tmp_left  = LOAD(src_ptr, input_pos);
                tmp_right = (tmp_right >> 16u) + (tmp_left << 16u);
                STORE_CURRENT_ITEM(dst_ptr, dst_iter, tmp_right);
                TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_attrs.step_x);

                // remaining
                for(uint x = xi + 1u, x_e = xi + (uint(KERNEL_WIDTH) / 2u); x < x_e; ++x, TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_attrs.step_x))
                {
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, x * src_attrs.step_x + y * src_attrs.stride_y);
                    tmp_right = LOAD(src_ptr, input_pos);
                    input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (x + 1u) * src_attrs.step_x + y * src_attrs.stride_y);
                    tmp_left  = LOAD(src_ptr, input_pos);
                    tmp_right = (tmp_right >> 16u) + (tmp_left << 16u);
                    STORE_CURRENT_ITEM(dst_ptr, dst_iter, tmp_right);
                }
            }
        }
    }

    // NOTE: must handle last element manually instead of in loops
    // to avoid write conflict across 2d boundary
    if(ch == uint(KERNEL_DEPTH) - 1u)
    {
        uint x    = xi + (uint(KERNEL_WIDTH) / 2u);
        uint y    = yi + uint(KERNEL_HEIGHT) - 1u;
        input_pos = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, x * src_attrs.step_x + y * src_attrs.stride_y);
        tmp       = LOAD(src_ptr, input_pos);
        if(!x_start_even)
        {
            tmp = (tmp >> 16u) + (tmp << 16u);
        }

#ifdef HAS_BIAS
        mediump vec2 bias_vec = vec2(1.f, 1.f);
        uint         bias_u   = packHalf2x16(bias_vec);
        if(z_start_even)
        {
            tmp = (tmp & 0xffffu) + (bias_u & 0xffff0000u);
        }
        else
        {
            tmp = (bias_u & 0xffffu);
        }
#endif /* HAS_BIAS */

        STORE_CURRENT_ITEM(dst_ptr, dst_iter, tmp);
    }
}

#endif /* KERNEL_1x1 */
#else  /* DATA_TYPE_FP32 */
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
            uint tmp_left  = uint(0);
            uint tmp_right = uint(0);
            tmp_right      = LOAD_CURRENT_ITEM(src_ptr, src_iter); //right half
            if(pos.x == uint(0))
            {
                tmp_left  = LOAD(src_ptr, TENSOR3D_OFFSET(src_nostep_iter, int(width), int(pos.y) - 1, int(pos.z))); //left half
                tmp_right = (tmp_left & uint(0xffff)) + (tmp_right << uint(16));
            }
            else
            {
                tmp_left  = LOAD(src_ptr, TENSOR3D_OFFSET(src_nostep_iter, (int(pos.x) - 1) * int(element_count), int(pos.y), int(pos.z)));
                tmp_right = ((tmp_left >> uint(16)) + (tmp_right << uint(16)));
            }
            STORE(dst_ptr, tmp_out_offset, tmp_right);
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

#endif /* IM2COL_REDUCED_GENERIC */
#else  /* DATA_TYPE_FP32 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
#endif /* IM2COL_REDUCED */

#ifdef WIDTH_OUTPUT

/** This kernel performs a reshaping of the output of the convolution layer.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 *
 * @param[in]  src_ptr     Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs   The attributes of the source tensor
 * @param[out] dst_ptr     Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs   The attributes of the destination tensor
 * @param[in]  dst_depth   The length of the destination tensor in Z dimension
 * @param[in]  dst_strideZ The actual stride of the destination tensor in Z dimension
 */

SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
    uint               dst_depth;
    uint               dst_strideZ;
};

#ifdef DATA_TYPE_FP32

TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, restrict);

void main(void)
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    uvec3 pos = uvec3(gl_GlobalInvocationID.xyz);
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, pos.x * src_attrs.step_y + pos.y * WIDTH_OUTPUT * src_attrs.step_y + (pos.z % dst_depth) * src_attrs.stride_x + (pos.z / dst_depth) * (src_attrs.stride_z));

    STORE_CURRENT_ITEM(dst_ptr, dst_iter,
                       LOAD_CURRENT_ITEM(src_ptr, src_iter));
}

#elif defined(DATA_TYPE_FP16)

TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, uint, dst_ptr, dst_shift, 2, restrict);

void main(void)
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    uvec3 pos = uvec3(gl_GlobalInvocationID.xyz);

    if((pos.z % dst_depth) % 2u == 0u)
    {
        uint common_offset_in_bytes = pos.x * src_attrs.step_y * 2u + pos.y * uint(WIDTH_OUTPUT) * src_attrs.step_y + (pos.z % dst_depth) * src_attrs.stride_x + (pos.z / dst_depth) * dst_strideZ;
        uint tmp1_in_offset         = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, common_offset_in_bytes);
        uint tmp2_in_offset         = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, common_offset_in_bytes + src_attrs.step_y);
        vec2 tmp1                   = LOAD_UNPACK2_HALF(src_ptr, tmp1_in_offset);
        vec2 tmp2                   = LOAD_UNPACK2_HALF(src_ptr, tmp2_in_offset);
        vec2 result                 = vec2(tmp1.x, tmp2.x);
        STORE_PACK2_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
    else
    {
        uint common_offset_in_bytes = pos.x * src_attrs.step_y * 2u + pos.y * uint(WIDTH_OUTPUT) * src_attrs.step_y + (pos.z % dst_depth) * src_attrs.stride_x + (pos.z / dst_depth) * dst_strideZ - 2u;
        uint tmp1_in_offset         = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, common_offset_in_bytes);
        uint tmp2_in_offset         = TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, common_offset_in_bytes + src_attrs.step_y);
        vec2 tmp1                   = LOAD_UNPACK2_HALF(src_ptr, tmp1_in_offset);
        vec2 tmp2                   = LOAD_UNPACK2_HALF(src_ptr, tmp2_in_offset);
        vec2 result                 = vec2(tmp1.y, tmp2.y);
        STORE_PACK2_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
}

#else /* DATA_TYPE_FP32 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
#endif /* COL2IM */
