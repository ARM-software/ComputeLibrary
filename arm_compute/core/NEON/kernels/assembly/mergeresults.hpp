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
#pragma once

template<unsigned int width, unsigned int height, typename Tin, typename Tout>
void MergeResults(Tout *out, const Tin *in, int ldc, int y0, int ymax, int x0, int xmax, const Tout alpha, const Tout beta) {
    int full_y_blocks = (ymax - y0) / height;
    int y_remainder = (ymax - y0) % height;
    int y_blocks = full_y_blocks + (y_remainder ? 1 : 0);

    int full_x_blocks = (xmax - x0) / width;
    int x_remainder = (xmax - x0) % width;
    int x_blocks = full_x_blocks + (x_remainder ? 1 : 0);

    for (int y_block = 0; y_block < y_blocks; y_block++) {
        int ybase = y0 + (y_block * height);

        int fill_rows = (y_block < full_y_blocks) ? height : y_remainder;

        for (int x_block = 0; x_block < x_blocks; x_block++) {
            int xbase = x0 + (x_block * width);

            int fill_cols = (x_block < full_x_blocks) ? width : x_remainder;

            for (int row=0; row < fill_rows; row++) {
                for (int col=0; col < fill_cols; col++) {
                    Tout &p = out[(ybase + row) * ldc + xbase + col];

                    p = (p * alpha) + (beta * in[row * width + col]);
                }
            }

            in += (width * height);
        }
    }
}

#include "merges/list.hpp"
