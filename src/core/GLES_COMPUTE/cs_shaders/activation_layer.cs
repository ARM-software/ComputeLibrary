/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#include "activation_layer_helpers_cs.h"
#include "helpers_cs.h"

/** This performs an activation function floating point inputs.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note Activation function should be given as a preprocessor argument using "#define act_name". e.g. "#define TANH"
 * @note A, B variables required by some activation functions are set using A_VAL= and B_VAL= respectively.
 *
 * @param[in]  src_ptr   Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs The attributes of the source tensor
 * @param[out] dst_ptr   Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs The attributes of the destination tensor
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
};

#ifdef DATA_TYPE_FP32
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    float data     = LOAD_CURRENT_ITEM(src_ptr, src_iter);
    float data_out = 0.f;
    // Perform activation
#ifdef LOGISTIC
    data_out = logistic_op(data);
#elif defined(TANH)     /*LOGISTIC*/
    data_out = tanh_op(data);
#elif defined(RELU)     /*RELU*/
    data_out = relu_op(data);
#elif defined(BRELU)    /*BRELU*/
    data_out = brelu_op(data);
#elif defined(LU_BRELU) /*LU_BRELU*/
    data_out = lu_brelu_op(data);
#elif defined(LRELU)    /*LRELU*/
    data_out = lrelu_op(data);
#elif defined(SRELU)    /*SRELU*/
    data_out = srelu_op(data);
#elif defined(ABS)      /*ABS*/
    data_out = abs_op(data);
#elif defined(SQUARE)   /*SQUARE*/
    data_out = square_op(data);
#elif defined(SQRT)     /*SQRT*/
    data_out = sqrt_op(data);
#elif defined(LINEAR)   /*LINEAR*/
    data_out = linear_op(data);
#elif defined(IDENTITY) /*IDENTITY*/
    data_out = identity_op(data);
#else                   /*LOGISTIC*/
#error Activation function not provided
#endif /*LOGISTIC*/

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, data_out);
}

#elif defined(DATA_TYPE_FP16)
TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    vec2 data = LOAD_UNPACK2_CURRENT_ITEM_HALF(src_ptr, src_iter);
    // Perform activation
    float a = data.x;
    float b = data.y;
    vec2  data_out;
#ifdef LOGISTIC         /*LOGISTIC*/
    data_out.x = logistic_op(a);
    data_out.y = logistic_op(b);
#elif defined(TANH)     /*TANH*/
    data_out.x = tanh_op(a);
    data_out.y = tanh_op(b);
#elif defined(RELU)     /*RELU*/
    data_out.x = relu_op(a);
    data_out.y = relu_op(b);
#elif defined(BRELU)    /*BRELU*/
    data_out.x = brelu_op(a);
    data_out.y = brelu_op(b);
#elif defined(LU_BRELU) /*LU_BRELU*/
    data_out.x = lu_brelu_op(a);
    data_out.y = lu_brelu_op(b);
#elif defined(LRELU)    /*LRELU*/
    data_out.x = lrelu_op(a);
    data_out.y = lrelu_op(b);
#elif defined(SRELU)    /*SRELU*/
    data_out.x = srelu_op(a);
    data_out.y = srelu_op(b);
#elif defined(ABS)      /*ABS*/
    data_out.x = abs_op(a);
    data_out.y = abs_op(b);
#elif defined(SQUARE)   /*SQUARE*/
    data_out.x = square_op(a);
    data_out.y = square_op(b);
#elif defined(SQRT)     /*SQRT*/
    data_out.x = sqrt_op(a);
    data_out.y = sqrt_op(b);
#elif defined(LINEAR)   /*LINEAR*/
    data_out.x = linear_op(a);
    data_out.y = linear_op(b);
#elif defined(IDENTITY)   /*IDENTITY*/
    data_out.x = identity_op(a);
    data_out.y = identity_op(b);
#else                   /*LOGISTIC*/
#error Activation function not provided
#endif /*LOGISTIC*/

    STORE_PACK2_CURRENT_ITEM_HALF(dst_ptr, dst_iter, data_out);
}
#endif /*DATA_TYPE_FP16*/
