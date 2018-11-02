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
#ifndef __ARM_COMPUTE_TEST_DECONVOLUTION_LAYER_H__
#define __ARM_COMPUTE_TEST_DECONVOLUTION_LAYER_H__

#include "tests/SimpleTensor.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
/** Deconvolution reference implementation.
 *
 * src              Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Data types supported: F32.
 * weights          The 4d weights with dimensions [width, height, OFM, IFM]. Data type supported: Same as @p input.
 * bias             Optional, ignored if NULL. The biases have one dimension. Data type supported: Same as @p input.
 * output_shape     Output tensor shape. The output has the same number of dimensions as the @p input.
 * info             Contains padding and policies to be used in the deconvolution, this is decribed in @ref PadStrideInfo.
 * a                The number of zeros added to right edge of the input.
 *
 */
template <typename T>
SimpleTensor<T> deconvolution_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<T> &bias, const TensorShape &output_shape, const PadStrideInfo &info,
                                    const std::pair<unsigned int, unsigned int> &a);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_DECONVOLUTION_LAYER_H__ */
