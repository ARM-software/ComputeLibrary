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

#include <premultiply.hpp>

#define CHANNEL_MULTIPLIER 6
#define BLOCK_SIZE 4

void do_premultiply_float_6(const float       *in_ptr,
                            const unsigned int ld_row,
                            const unsigned int ld_col,
                            float             *out_ptr,
                            const unsigned int out_ld_row,
                            const unsigned int out_ld_col,
                            const unsigned int tile_rows,
                            const unsigned int tile_cols,
                            const unsigned     input_channels)
{
    for(unsigned int i = 0; i < tile_rows; i++)
    {
        const float *ip2 = in_ptr + i * ld_row;
        float       *op2 = out_ptr + i * out_ld_row;
        for(unsigned int j = 0; j < tile_cols; j++)
        {
            const float *ip = ip2;
            float       *op = op2;

            unsigned int num_blocks = input_channels / BLOCK_SIZE;
            for(unsigned int c = 0; c < num_blocks; c++)
            {
                float vals[BLOCK_SIZE];
                for(unsigned int v = 0; v < BLOCK_SIZE; v++)
                {
                    vals[v] = ip[v];
                }
                ip += BLOCK_SIZE;

                for(unsigned int v = 0; v < BLOCK_SIZE; v++)
                {
                    for(unsigned int r = 0; r < CHANNEL_MULTIPLIER; r++)
                    {
                        op[r] = vals[v];
                    }
                    op += CHANNEL_MULTIPLIER;
                }
            }

            unsigned int rem = input_channels - num_blocks * BLOCK_SIZE;
            for(unsigned int c = 0; c < rem; c++)
            {
                float val = ip[c];
                for(unsigned int r = 0; r < CHANNEL_MULTIPLIER; r++)
                {
                    op[r] = val;
                }
                op += CHANNEL_MULTIPLIER;
            }

            ip2 += ld_col;
            op2 += out_ld_col;
        }
    }
}
