/*
 * Copyright (c) 2022 Arm Limited.
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

#include "src/core/NEON/kernels/arm_gemm/utils.hpp"
#include "interleaves/generic.hpp"
#include "depthfirst_driver.hpp"

namespace arm_conv {
namespace depthwise {

class DepthfirstStrategyUntyped : public IDepthfirstStrategy
{
  public:
  virtual arm_gemm::VLType get_vl_type() const = 0;

  virtual unsigned int get_kernel_rows() const = 0;
  virtual unsigned int get_kernel_cols() const = 0;

  virtual unsigned int get_stride_rows() const = 0;
  virtual unsigned int get_stride_cols() const = 0;

  virtual unsigned int get_input_rows() const override;
  virtual unsigned int get_input_cols() const override;

  virtual unsigned int get_n_input_points() const;
  virtual unsigned int get_n_output_points() const;
  virtual unsigned int get_n_kernel_points() const;

  // Get the number of VLs used in the accumulator, this defaults to 1.
  virtual unsigned int get_accumulator_depth_vl() const;

  // Get the order in which to pack the weights, this defaults to a row-major
  // sweep over the weight tensor.
  virtual bool get_kernel_packing_point(const unsigned int index, unsigned int &x, unsigned int &y) const;
};

template <typename TInput, typename TWeight, typename TOutput, typename TAccum, typename OutputStage>
class DepthfirstStrategy : public DepthfirstStrategyUntyped
{
  public:
  virtual size_t get_storage_size(const DepthwiseArgs &args) const
  {
    interleaves::PackingArguments packing_args(
      this->get_kernel_rows(), this->get_kernel_cols(), sizeof(TWeight),
      true, sizeof(TAccum),
      this->get_vl_type(), sizeof(TAccum), this->get_accumulator_depth_vl(),
      [this] (unsigned int idx, unsigned int &x, unsigned int &y) -> bool
      { return this->get_kernel_packing_point(idx, x, y); }
    );
    return interleaves::get_storage_size_generic(packing_args, args);
  }

  virtual void pack_parameters(
    const DepthwiseArgs &args, void *buffer,
    const void *biases, const OutputStage &,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) const
  {
    interleaves::PackingArguments packing_args(
      this->get_kernel_rows(), this->get_kernel_cols(), sizeof(TWeight),
      true, sizeof(TAccum),
      this->get_vl_type(), sizeof(TAccum), this->get_accumulator_depth_vl(),
      [this] (unsigned int idx, unsigned int &x, unsigned int &y) -> bool
      { return this->get_kernel_packing_point(idx, x, y); }
    );
    interleaves::pack_parameters_generic(
      packing_args, args, buffer, biases, weights, ld_weight_col, ld_weight_row);
  }
};

}  // namespace depthwise
}  // namespace arm_conv
