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
)
{
  const auto per_iter = acc_depth * arm_gemm::utils::get_vector_length<int32_t>(vec_type);
  return arm_gemm::roundup((long unsigned int) n_input_channels, per_iter) * kernel_rows * kernel_cols * sizeof(int8_t);
}

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
)
{
  int8_t *outptr = reinterpret_cast<int8_t *>(_outptr);
  const int8_t *weights = reinterpret_cast<const int8_t *>(_weights);

  // Get the strides
  ld_weight_col = (ld_weight_col == 0) ? n_channels * sizeof(int8_t) : ld_weight_col;
  ld_weight_row = (ld_weight_row == 0) ? kernel_cols * ld_weight_col : ld_weight_row;

  // Pack into per-iter chunks.
  const auto per_iter = acc_depth * arm_gemm::utils::get_vector_length<int32_t>(vec_type);
  for (unsigned int c = 0; c < n_channels; c += per_iter)
  {
    auto weight_row = weights + c;
    const auto to_copy = std::min<unsigned int>(per_iter, n_channels - c);

    for (unsigned int i = 0; i < kernel_rows; i++)
    {
      auto weight_col = weight_row;

      for (unsigned int j = 0; j < kernel_cols; j++)
      {
        memcpy(outptr, weight_col, to_copy);
        outptr += per_iter;
        weight_col += ld_weight_col;
      }

      weight_row += ld_weight_row;
    }
  }
}


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


namespace arm_conv {
namespace depthwise {

#if defined(__ARM_FEATURE_SVE)

ADD_IMPLEMENTATION(sve, s8q, int8_t, SVE, 2, 3, 3)
ADD_IMPLEMENTATION(sve, s8q, int8_t, SVE, 2, 5, 5)
ADD_IMPLEMENTATION(sve, u8q, uint8_t, SVE, 2, 3, 3)
ADD_IMPLEMENTATION(sve, u8q, uint8_t, SVE, 2, 5, 5)

#endif  // defined(__ARM_FEATURE_SVE)

ADD_IMPLEMENTATION(a64, s8q, int8_t, None, 2, 3, 3)
ADD_IMPLEMENTATION(a64, s8q, int8_t, None, 2, 5, 5)
ADD_IMPLEMENTATION(a64, u8q, uint8_t, None, 2, 3, 3)
ADD_IMPLEMENTATION(a64, u8q, uint8_t, None, 2, 5, 5)

}  // namespace depthwise
}  // namespace arm_conv
