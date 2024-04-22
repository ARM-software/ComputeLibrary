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
#include "Reorder.h"
#include "src/core/NEON/kernels/arm_gemm/utils.hpp"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{

/*
 * Generic transform.
 *
 * Assuming the untransposed case, this works by first reading <BlockBy>
 * consecutive values from the first input row.  This same number of values
 * are then read from the next <IntBy-1> rows.  Now return to the first
 * input row and repeat.
 *
 * Need to cope with the work requested in either dimension not actually
 * being a multiple of the block sizes.
 */
template <unsigned int tIntBy, unsigned int BlockBy, bool Transposed, size_t TOutSize, size_t TInSize, typename d_type, arm_gemm::VLType vlt>
struct Transform_ref
{
    template <typename TOut, typename TIn>
    static void Transform(TOut &out, const TIn in, const int stride,
                          const int y0, const int ymax, const int x0, const int xmax)
    {
        // NOTE: This code is disabled to avoid the call to get_vector_length(), so templated transforms will not be
        // correct for SVE.  This is not an issue as we have specializations for all SVE cases.
        // For SVE cases we multiply the interleave factor by the vector length.
        // const unsigned int IntBy = tIntBy * (vlt == VLType::SVE ? get_vector_length<TOut>() / BlockBy : 1);
        const unsigned int IntBy     = tIntBy;
        int                out_index = 0;

        const int n_whole_y_blocks = (ymax - y0) / IntBy;
        const int y_remainders     = (ymax - y0) % IntBy;
        const int n_y_blocks       = n_whole_y_blocks + (y_remainders ? 1 : 0);

        const int n_whole_x_blocks = (xmax - x0) / BlockBy;
        const int x_remainders     = (xmax - x0) % BlockBy;
        const int n_x_blocks       = n_whole_x_blocks + (x_remainders ? 1 : 0);

        // "Y" loop: advance down the rows of the source IntBy rows at a time.
        // Set up fill_rows to show the number rows to copy from, and blank_rows
        // for the number of blank rows to add.
        for(int y_block = 0; y_block < n_y_blocks; y_block++)
        {
            const int fill_rows  = (y_block < n_whole_y_blocks) ? IntBy : y_remainders;
            const int blank_rows = IntBy - fill_rows;

            const int y_base = y0 + (y_block * IntBy);

            // So now advance along this block of rows, BlockBy columns at a time.
            for(int x_block = 0; x_block < n_x_blocks; x_block++)
            {
                const int fill_cols  = (x_block < n_whole_x_blocks) ? BlockBy : x_remainders;
                const int blank_cols = BlockBy - fill_cols;

                const int x_base = x0 + (x_block * BlockBy);

                for(int row = 0; row < fill_rows; row++)
                {
                    for(int col = 0; col < fill_cols; col++)
                    {
                        // In-range copy.  If it's transposed, we reverse the sense of rows and columns here.
                        if(Transposed)
                        {
                            out[out_index] = in[(x_base + col) * stride + y_base + row];
                            out_index++;
                        }
                        else
                        {
                            out[out_index] = in[(y_base + row) * stride + x_base + col];
                            out_index++;
                        }
                    }
                    // "col" tail - row is in range but column is out of range.
                    for(int col = 0; col < blank_cols; col++)
                    {
                        out[out_index] = 0;
                        out_index++;
                    }
                }
                // "row" tail - row is out of range so fill with zeros always.
                const d_type zeroval = 0;
                const int    pads    = blank_rows * (fill_cols + blank_cols);

                for(int i = 0; i < pads; i++)
                {
                    out[out_index] = zeroval;
                }

                out_index += pads;
            }
        }
    }
};

template <typename T>
SimpleTensor<T> reorder_layer(const SimpleTensor<T> &src, const TensorShape &output_shape, WeightFormat output_wf)
{
    SimpleTensor<T> dst{ output_shape, src.data_type() };
    const int       cols = src.shape()[0];
    const int       rows = src.shape()[1];

    switch(output_wf)
    {
        case WeightFormat::OHWIo4:
        {
            Transform_ref<4, 1, true, sizeof(float), sizeof(float), float, arm_gemm::VLType::None>::Transform<SimpleTensor<T> &, SimpleTensor<T>>(dst, src, rows, 0, rows, 0, cols);
            break;
        }
        case WeightFormat::OHWIo8:
        {
            Transform_ref<8, 1, true, sizeof(float), sizeof(float), float, arm_gemm::VLType::None>::Transform<SimpleTensor<T> &, SimpleTensor<T>>(dst, src, rows, 0, rows, 0, cols);
            break;
        }
        default:
            break;
    }

    return dst;
}

template SimpleTensor<float> reorder_layer(const SimpleTensor<float> &src, const TensorShape &output_shape, WeightFormat output_wf);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
