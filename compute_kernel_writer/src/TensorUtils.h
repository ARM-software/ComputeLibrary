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

#ifndef COMPUTE_KERNEL_WRITER_SRC_TENSORUTILS_H
#define COMPUTE_KERNEL_WRITER_SRC_TENSORUTILS_H

#include <cstdint>

/** Tensor specific utility functions */
namespace ckw
{
// Forward declarations
enum class TensorDataLayout;
enum class TensorDataLayoutComponent;
enum class TensorComponent : uint32_t;

/** Get tensor dimension from a given data layout and data layout component
 *
 * @param[in] layout    Layout of the tensor
 * @param[in] component Data layout component
 *
 * @return the @ref TensorComponent
 */
TensorComponent get_tensor_dimension(TensorDataLayout layout, TensorDataLayoutComponent component);

/** Get tensor stride from a given data layout and data layout component
 *
 * @param[in] layout    Layout of the tensor
 * @param[in] component Data layout component
 *
 * @return the @ref TensorComponent
 */
TensorComponent get_tensor_stride(TensorDataLayout layout, TensorDataLayoutComponent component);
} // namespace ckw
#endif /* COMPUTE_KERNEL_WRITER_SRC_TENSORUTILS_H */
