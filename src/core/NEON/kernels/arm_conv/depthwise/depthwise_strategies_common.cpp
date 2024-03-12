/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#include "depthwise_strategies_common.hpp"

namespace arm_conv {
namespace depthwise {

unsigned int DepthfirstStrategyUntyped::get_input_rows() const
{
  return this->get_kernel_rows() + (this->get_output_rows() - 1) * this->get_stride_rows();
}

unsigned int DepthfirstStrategyUntyped::get_input_cols() const
{
  return this->get_kernel_cols() + (this->get_output_cols() - 1) * this->get_stride_cols();
}

unsigned int DepthfirstStrategyUntyped::get_n_input_points() const { return this->get_input_rows() * this->get_input_cols(); }
unsigned int DepthfirstStrategyUntyped::get_n_output_points() const { return this->get_output_rows() * this->get_output_cols(); }
unsigned int DepthfirstStrategyUntyped::get_n_kernel_points() const { return this->get_kernel_rows() * this->get_kernel_cols(); }

bool DepthfirstStrategyUntyped::uses_premultiply() const { return true; }

unsigned int DepthfirstStrategyUntyped::get_accumulator_depth_vl() const { return 1; }

bool DepthfirstStrategyUntyped::get_kernel_packing_point(const unsigned int index, unsigned int &x, unsigned int &y) const
{
  // Get the kernel point to pack at the given index; return false to
  // indicate that this index, and all greater indices, is out of range.
  if (index < (this->get_kernel_cols() * this->get_kernel_rows()))
  {
    y = index % this->get_kernel_cols();
    x = index / this->get_kernel_cols();
    return true;
  }
  return false;
}

}  // namespace depthwise
}  // namespace arm_conv
