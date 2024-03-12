/*
 * Copyright (c) 2023 Arm Limited.
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

void do_premultiply_float_6(const float       *in_ptr,
                            const unsigned int ld_row,
                            const unsigned int ld_col,
                            float             *out_ptr,
                            const unsigned int out_ld_row,
                            const unsigned int out_ld_col,
                            const unsigned int tile_rows,
                            const unsigned int tile_cols,
                            const unsigned     input_channels);

template <typename T>
void do_premultiply(const T           *in_ptr,
                    const unsigned int ld_row,
                    const unsigned int ld_col,
                    T                 *out_ptr,
                    const unsigned int out_ld_row,
                    const unsigned int out_ld_col,
                    const unsigned int tile_rows,
                    const unsigned int tile_cols,
                    const unsigned     input_channels,
                    const unsigned int channel_multiplier)
{
    if (sizeof(T) == 4 && channel_multiplier == 6)
    {
        do_premultiply_float_6((const float *)in_ptr, ld_row, ld_col, (float *)out_ptr, out_ld_row, out_ld_col,
                               tile_rows, tile_cols, input_channels);
    }
    else
    {
        for (unsigned int i = 0; i < tile_rows; i++)
        {
            const T *ip2 = in_ptr + i * ld_row;
            T       *op2 = out_ptr + i * out_ld_row;
            for (unsigned int j = 0; j < tile_cols; j++)
            {
                const T *ip = ip2;
                T       *op = op2;
                for (unsigned int c = 0; c < input_channels; c++)
                {
                    T val = *ip;
                    ip++;

                    for (unsigned int r = 0; r < channel_multiplier; r++)
                    {
                        op[r] = val;
                    }
                    op += channel_multiplier;
                }
                ip2 += ld_col;
                op2 += out_ld_col;
            }
        }
    }
}
