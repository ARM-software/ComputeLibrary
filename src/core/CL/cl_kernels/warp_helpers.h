/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "helpers.h"

/** Clamps the given coordinates to the borders according to the border size.
 *
 * @param[in] coords      Vector of 2D coordinates to clamp. Even positions are X coords, odd positions are Y coords.
 * @param[in] width       Width of the image
 * @param[in] height      Height of the image
 * @param[in] border_size Border size of the image
 *
 */
inline const float8 clamp_to_border_with_size(float8 coords, const float width, const float height, const float border_size)
{
    const float4 clamped_x = clamp(coords.even, 0.0f - border_size, width - 1 + border_size);
    const float4 clamped_y = clamp(coords.odd, 0.0f - border_size, height - 1 + border_size);
    return (float8)(clamped_x.s0, clamped_y.s0, clamped_x.s1, clamped_y.s1, clamped_x.s2, clamped_y.s2, clamped_x.s3, clamped_y.s3);
}

/** Clamps the given coordinates to the borders.
 *
 * @param[in] coords Vector of 2D coordinates to clamp. Even positions are X coords, odd positions are Y coords.
 * @param[in] width  Width of the image
 * @param[in] height Height of the image
 *
 */
inline const float8 clamp_to_border(float8 coords, const float width, const float height)
{
    return clamp_to_border_with_size(coords, width, height, 1);
}

/** Reads four texels from the input image. The coords vector is used to determine which texels to be read.
 *
 * @param[in] in     Pointer to the source image.
 * @param[in] coords Vector of coordinates to be read from the image.
 */
inline const VEC_DATA_TYPE(DATA_TYPE, 4) read_texels4(const Image *in, const int8 coords)
{
    return (VEC_DATA_TYPE(DATA_TYPE, 4))(*((__global DATA_TYPE *)offset(in, coords.s0, coords.s1)),
                                         *((__global DATA_TYPE *)offset(in, coords.s2, coords.s3)),
                                         *((__global DATA_TYPE *)offset(in, coords.s4, coords.s5)),
                                         *((__global DATA_TYPE *)offset(in, coords.s6, coords.s7)));
}

/** Returns the current thread coordinates. */
inline const float2 get_current_coords()
{
    return (float2)(get_global_id(0) * 4, get_global_id(1));
}

/** Given a texel coordinates this function will return the following array of coordinates:
 * [ P, right neighbour, below neighbour, below right neighbour ]
 *
 * @note No checks to see if the coordinates are out of the image are done here.
 *
 * @param[in] coord Input coordinates
 *
 * @return vector of 8 floats with the coordinates, even positions are x and odd y.
 */
inline const float8 get_neighbour_coords(const float2 coord)
{
    return (float8)(/*tl*/ coord.s0, coord.s1, /*tr*/ coord.s0 + 1, coord.s1, /*bl*/ coord.s0, coord.s1 + 1, /*br*/ coord.s0 + 1, coord.s1 + 1);
}

/** Computes the bilinear interpolation for each set of coordinates in the vector coords and returns the values
 *
 * @param[in] in          Pointer to the source image.
 * @param[in] coords      Vector of four 2D coordinates. Even pos is x and odd y.
 * @param[in] width       Width of the image
 * @param[in] height      Height of the image
 * @param[in] border_size Border size
 */
inline const VEC_DATA_TYPE(DATA_TYPE, 4) bilinear_interpolate_with_border(const Image *in, const float8 coords, const float width, const float height, const float border_size)
{
    // If any of the 4 texels is out of the image's boundaries we use the border value (REPLICATE or CONSTANT) for any texel out of the image.

    // Sets the 4x4 coordinates for each of the four input texels
    const float8  fc = floor(coords);
    const float16 c1 = (float16)(
                           clamp_to_border_with_size(get_neighbour_coords((float2)(fc.s0, fc.s1)), width, height, border_size),
                           clamp_to_border_with_size(get_neighbour_coords((float2)(fc.s2, fc.s3)), width, height, border_size));
    const float16 c2 = (float16)(
                           clamp_to_border_with_size(get_neighbour_coords((float2)(fc.s4, fc.s5)), width, height, border_size),
                           clamp_to_border_with_size(get_neighbour_coords((float2)(fc.s6, fc.s7)), width, height, border_size));

    // Loads the values from the input image
    const float16 t = (float16)(
                          /* tl, tr, bl, br */
                          * ((__global DATA_TYPE *)offset(in, c1.s0, c1.s1)), *((__global DATA_TYPE *)offset(in, c1.s2, c1.s3)),
                          *((__global DATA_TYPE *)offset(in, c1.s4, c1.s5)), *((__global DATA_TYPE *)offset(in, c1.s6, c1.s7)),
                          *((__global DATA_TYPE *)offset(in, c1.s8, c1.s9)), *((__global DATA_TYPE *)offset(in, c1.sa, c1.sb)),
                          *((__global DATA_TYPE *)offset(in, c1.sc, c1.sd)), *((__global DATA_TYPE *)offset(in, c1.se, c1.sf)),
                          *((__global DATA_TYPE *)offset(in, c2.s0, c2.s1)), *((__global DATA_TYPE *)offset(in, c2.s2, c2.s3)),
                          *((__global DATA_TYPE *)offset(in, c2.s4, c2.s5)), *((__global DATA_TYPE *)offset(in, c2.s6, c2.s7)),
                          *((__global DATA_TYPE *)offset(in, c2.s8, c2.s9)), *((__global DATA_TYPE *)offset(in, c2.sa, c2.sb)),
                          *((__global DATA_TYPE *)offset(in, c2.sc, c2.sd)), *((__global DATA_TYPE *)offset(in, c2.se, c2.sf)));
    const float8 a  = coords - fc;
    const float8 b  = ((float8)(1.f)) - a;
    const float4 fr = (float4)(
                          ((t.s0 * b.s0 * b.s1) + (t.s1 * a.s0 * b.s1) + (t.s2 * b.s0 * a.s1) + (t.s3 * a.s0 * a.s1)),
                          ((t.s4 * b.s2 * b.s3) + (t.s5 * a.s2 * b.s3) + (t.s6 * b.s2 * a.s3) + (t.s7 * a.s2 * a.s3)),
                          ((t.s8 * b.s4 * b.s5) + (t.s9 * a.s4 * b.s5) + (t.sa * b.s4 * a.s5) + (t.sb * a.s4 * a.s5)),
                          ((t.sc * b.s6 * b.s7) + (t.sd * a.s6 * b.s7) + (t.se * b.s6 * a.s7) + (t.sf * a.s6 * a.s7)));
    return CONVERT(fr, VEC_DATA_TYPE(DATA_TYPE, 4));
}

/** Computes the bilinear interpolation for each set of coordinates in the vector coords and returns the values
 *
 * @param[in] in     Pointer to the source image.
 * @param[in] coords Vector of four 2D coordinates. Even pos is x and odd y.
 * @param[in] width  Width of the image
 * @param[in] height Height of the image
 */
inline const VEC_DATA_TYPE(DATA_TYPE, 4) bilinear_interpolate(const Image *in, const float8 coords, const float width, const float height)
{
    return bilinear_interpolate_with_border(in, coords, width, height, 1);
}
