/*
 * Copyright (c) 2021 Arm Limited.
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

#include "arm_gemm.hpp"
#include "src/core/NEON/kernels/arm_gemm/utils.hpp"
#include "src/core/NEON/kernels/assembly/depthwise.hpp"
#include <cstdint>
#include <cstring>

using namespace arm_gemm;

size_t generic_get_packed_size(
  const VLType vec_type,
  const unsigned int acc_depth,
  const unsigned int kernel_rows,
  const unsigned int kernel_cols,
  const unsigned int n_input_channels
);

void generic_pack(
  const VLType vec_type,
  const unsigned int acc_depth,
  const unsigned int kernel_rows,
  const unsigned int kernel_cols,
  const unsigned int n_channels,
  void *_outptr,
  const void *_weights,
  size_t ld_weight_col,
  size_t ld_weight_row
);

#define ADD_IMPLEMENTATION(ARCH, TYPENAME, TYPE, VEC_TYPE, ACC_DEPTH, KERN_ROWS, KERN_COLS) \
struct interleave_  ## ARCH ## _ ## TYPENAME ## _ ## KERN_ROWS ## x ## KERN_COLS ## _mla \
{ \
  static size_t get_packed_size(const DepthwiseArgs &args); \
  static void pack_parameters( \
    unsigned int n_channels, void *outptr, \
    const TYPE *weights, size_t ld_weight_col, size_t ld_weight_row \
  ); \
}; \
\
size_t interleave_  ## ARCH ## _ ## TYPENAME ## _ ## KERN_ROWS ## x ## KERN_COLS ## _mla::get_packed_size(const DepthwiseArgs &args) \
{ \
  return generic_get_packed_size(VLType::VEC_TYPE, ACC_DEPTH, KERN_ROWS, KERN_COLS, args.input_channels); \
} \
\
void interleave_  ## ARCH ## _ ## TYPENAME ## _ ## KERN_ROWS ## x ## KERN_COLS ## _mla::pack_parameters(unsigned int n_channels, void *outptr, \
                            const TYPE *weights, size_t ld_weight_col, size_t ld_weight_row) \
{ \
  generic_pack(VLType::VEC_TYPE, ACC_DEPTH, KERN_ROWS, KERN_COLS, n_channels, outptr, weights, ld_weight_col, ld_weight_row); \
}
