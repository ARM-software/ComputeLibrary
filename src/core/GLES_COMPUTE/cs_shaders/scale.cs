/*
 * Copyright (c) 2016-2019 ARM Limited.
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

// We DO have to use highp for DATA_TYPE_FP16 float here to calculate the coordinates of source tensor. float is highp by default, but we still write it down here to make it more clearly, and mediump is only used for src/dst tensor in shader body.
precision highp float;

/** Performs an affine transformation on an tensor interpolating with the NEAREAST NEIGHBOUR method. Input and output are single channel FP16.
 *
 * @param[in]  src_ptr      Pointer to the source tensor. Supported data types: FP16.
 * @param[in]  src_attrs    The attributes of the source tensor
 * @param[out] dst_ptr      Pointer to the destination tensor. Supported data types: FP16. (Must be the same as the input)
 * @param[in]  dst_attrs    The attributes of the destination tensor
 * @param[in]  input_width  Input tensor width
 * @param[in]  input_height Input tensor height
 * @param[in]  scale        The scale factor along x/y dimension
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
    float              input_width;
    float              input_height;
    vec2               scale;
};

#if defined(DATA_TYPE_FP16)
#if defined(SCALE_NEAREST_GENERIC)
TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);

vec4[2] transform_nearest(vec2 coord, vec2 scale)
{
    vec4 in_x_coords = vec4(coord.x, 1.f + coord.x, 2.f + coord.x, 3.f + coord.x);

    vec4[2] t;
#if defined(SAMPLING_POLICY_CENTER) /* SAMPLING_POLICY_CENTER */
    t[0] = (in_x_coords + (vec4(0.5f))) * scale.x;
    t[1] = vec4((coord.y + 0.5f) * scale.y);
#elif defined(SAMPLING_POLICY_TOP_LEFT) /* SAMPLING_POLICY_TOP_LEFT */
    t[0] = in_x_coords * scale.x;
    t[1] = vec4(coord.y) * scale.y;
#else                                   /* Unsupported sampling policy */
#error Unsupported sampling policy
#endif /* SAMPLING_POLICY */

    return t;
}

vec4[2] clamp_to_border_with_size(vec4[2] coords, float width, float height, float border_size)
{
    vec4[2] c;
    c[0] = clamp(coords[0], 0.0f - border_size, width - 1.f + border_size);
    c[1] = clamp(coords[1], 0.0f - border_size, height - 1.f + border_size);

    return c;
}

void main()
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    vec4[2] tc = clamp_to_border_with_size(transform_nearest(vec2(gl_GlobalInvocationID.x << uint(2), gl_GlobalInvocationID.y), scale), input_width, input_height, float(BORDER_SIZE));

    mediump vec2 s = vec2(0.0f);
    mediump vec4 d = vec4(0.0f);

    for(int i = 0; i < 4; i++)
    {
        uint offset_in_bytes = tensor3D_offset_in_bytes(src_iter, int(tc[0][i]), int(tc[1][i]), int(gl_GlobalInvocationID.z));

        s = LOAD_UNPACK2_HALF(src_ptr, uint(offset_in_bytes >> src_shift));

        if(offset_in_bytes % uint(4) == uint(0))
        {
            d[i] = s.x;
        }
        else
        {
            d[i] = s.y;
        }
    }

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, d);
}
#elif defined(SCALE_NEAREST_8X) /* SCALE_NEAREST_GENERIC */
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);

void main()
{
    Tensor3DIterator src_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(src_attrs, src_shift);
    Tensor3DIterator dst_iter = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    uvec2 tc = uvec2(gl_GlobalInvocationID.x << uint(2), gl_GlobalInvocationID.y >> uint(1));

    mediump vec4 s = vec4(0.0f);
    mediump      vec4[2] d;

    s = LOAD_UNPACK4_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, int(tc[0]), int(tc[1]), int(gl_GlobalInvocationID.z)));

    d[0] = vec4(s.x, s.x, s.y, s.y);
    d[1] = vec4(s.z, s.z, s.w, s.w);

    STORE_PACK8_CURRENT_ITEM_HALF(dst_ptr, dst_iter, d);
}
#endif                          /* SCALE_NEAREST_GENERIC */

#else /* DATA_TYPE_FP16 */
#error Data type not supported
#endif /* DATA_TYPE_FP16 */
