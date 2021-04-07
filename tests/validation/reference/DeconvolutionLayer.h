/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_DECONVOLUTION_LAYER_H
#define ARM_COMPUTE_TEST_DECONVOLUTION_LAYER_H

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
 * src              Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs.
 *                  Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
 * weights          The 4d weights with dimensions [width, height, OFM, IFM]. Data type supported: Same as @p input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
 * bias             Optional, ignored if NULL. The biases have one dimension.
 *                  Data type supported: Same as @p input, except for input of QASYMM8/QASYMM8_SIGNED types where biases should be of S32 type
 * output_shape     Output tensor shape. The output has the same number of dimensions as the @p input.
 * info             Contains padding and policies to be used in the deconvolution, this is decribed in @ref PadStrideInfo.
 * a                The number of zeros added to right and top edges of the input.
 *
 */
template <typename T, typename TW, typename TB>
SimpleTensor<T> deconvolution_layer(const SimpleTensor<T> &src, const SimpleTensor<TW> &weights, const SimpleTensor<TB> &bias, const TensorShape &output_shape, const PadStrideInfo &info,
                                    QuantizationInfo out_qinfo = QuantizationInfo());
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DECONVOLUTION_LAYER_H */
