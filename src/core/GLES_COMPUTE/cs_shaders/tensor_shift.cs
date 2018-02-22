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

/** This kernel performs a shift to move "pad_x" columns to the right.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note The width must be passed at compile time using "#define WIDTH n" e.g. "#define WIDTH 1"
 *
 * @param[in,out] src_ptr   Pointer to the source tensor slice. Supported data types: F16/F32
 * @param[in]     src_attrs The attributes of the source tensor
 * @param[in]     pad_x     The padding of the source tensor in x dimension
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    uint               pad_x;
};

#if defined(DATA_TYPE_FP16)
TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, restrict);

void main()
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    int              n        = int(pad_x) % 2;

    if(n == 1)
    {
        int i = 0;
        if((WIDTH % 2) == 1)
        {
            i = WIDTH + int(pad_x) - 2;
        }
        else
        {
            vec2 s0_end = LOAD_UNPACK2_HALF(src_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (2 * (WIDTH - 2))));
            vec2 s_end  = vec2(s0_end.y, 0.f);
            STORE_PACK2_HALF(src_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (2 * (WIDTH + int(pad_x) - 1))), s_end);
            i = WIDTH + int(pad_x) - 3;
        }
        for(; i >= (int(pad_x) + 1); i = i - 2)
        {
            vec2 s0 = LOAD_UNPACK2_HALF(src_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (2 * (i - int(pad_x) - 1))));
            vec2 s1 = LOAD_UNPACK2_HALF(src_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (2 * (i - int(pad_x) + 1))));
            vec2 s  = vec2(s0.y, s1.x);
            STORE_PACK2_HALF(src_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (2 * i)), s);
        }
        for(int j = 0; j < (int(pad_x) - 1); j = j + 2)
        {
            vec2 s_origin = vec2(0.f);
            STORE_PACK2_CURRENT_ITEM_HALF(src_ptr, src_iter, s_origin);
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, 4);
        }
        vec2 s0_origin = LOAD_UNPACK2_CURRENT_ITEM_HALF(src_ptr, src_iter);
        vec2 s_origin  = vec2(0.f, s0_origin.x);
        STORE_PACK2_CURRENT_ITEM_HALF(src_ptr, src_iter, s_origin);
    }
    else
    {
        int i = 0;
        if((WIDTH % 2) == 0)
        {
            i = WIDTH + int(pad_x) - 2;
        }
        else
        {
            vec2 s0_end = LOAD_UNPACK2_HALF(src_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (2 * (WIDTH - 1))));
            vec2 s_end  = vec2(s0_end.x, 0.f);
            STORE_PACK2_HALF(src_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (2 * (WIDTH + int(pad_x) - 1))), s_end);
            i = WIDTH + int(pad_x) - 3;
        }
        for(; i >= (int(pad_x)); i = i - 2)
        {
            vec2 s = LOAD_UNPACK2_HALF(src_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (2 * (i - int(pad_x)))));
            STORE_PACK2_HALF(src_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(src_iter, (2 * i)), s);
        }
        for(int j = 0; j < int(pad_x); j = j + 2)
        {
            vec2 s = vec2(0.f);
            STORE_PACK2_CURRENT_ITEM_HALF(src_ptr, src_iter, s);
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, 4);
        }
    }
}
#elif defined(DATA_TYPE_FP32)
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, restrict);

void main()
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);

    for(int i = (WIDTH + int(pad_x) - 1); i >= int(pad_x); i--)
    {
        float sorigin = LOAD(src_ptr, TENSOR_OFFSET_ADVANCE(src_iter, (i - int(pad_x))));
        STORE(src_ptr, TENSOR_OFFSET_ADVANCE(src_iter, i), sorigin);
    }
    for(int j = 0; j < int(pad_x); j++)
    {
        STORE_CURRENT_ITEM(src_ptr, src_iter, 0.f);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, 4);
    }
}
#else /* DATA_TYPE_FP16 */
#error Data type not supported
#endif /* DATA_TYPE_FP16 */
