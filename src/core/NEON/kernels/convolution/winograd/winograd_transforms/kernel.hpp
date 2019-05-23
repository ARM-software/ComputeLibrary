/*
 * Copyright (c) 2019 ARM Limited.
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
#include "winograd.hpp"
using namespace winograd;

#define MEMBERFN(RTYPE) template <\
  int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename TIn, typename TOut, WinogradRoots Roots\
> RTYPE WeightTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, TIn, TOut, Roots>

MEMBERFN()::WeightTransform(
  const int n_output_channels,
  const int n_input_channels
) : _n_output_channels(n_output_channels), _n_input_channels(n_input_channels),
    _matrices(nullptr), _matrix_stride(0), _matrix_row_stride(0), _weights(nullptr)
{

}

MEMBERFN(void)::set_weight_tensor(const void * const weights)
{
  _weights = static_cast<const TIn *>(weights);
}

MEMBERFN(void)::set_output_matrices(void * const mptr, const int ldmatrix, const int ldrow)
{
  _matrices = static_cast<TOut *>(mptr);
  _matrix_stride = ldmatrix;
  _matrix_row_stride = ldrow;
}

MEMBERFN(size_t)::get_working_space_size(unsigned int) const
{
  return 0;
}

MEMBERFN(void)::set_working_space(void *)
{
}

MEMBERFN(unsigned int)::get_window(void) const
{
  // TODO When the weights transform supports multithreading, return the number
  // of output channels. For now we return 1 to indicate that the weights must
  // be transformed as a single block.
  // return n_output_channels;
  return 1;
}

MEMBERFN(void)::run(const unsigned int, const unsigned int, unsigned int)
{
  execute(
    _n_output_channels, _n_input_channels, _weights,
    _matrices, _matrix_stride, _matrix_row_stride
  );
}
