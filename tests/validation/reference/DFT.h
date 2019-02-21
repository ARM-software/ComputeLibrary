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
#ifndef __ARM_COMPUTE_TEST_FFT_H__
#define __ARM_COMPUTE_TEST_FFT_H__

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
enum class FFTDirection
{
    Forward,
    Inverse
};

/** Performs an one dimensional DFT on a real input.
 *
 * @param[in] src Source tensor.
 *
 * @return Complex output of length n/2 + 1 due to symmetry.
 */
template <typename T>
SimpleTensor<T> rdft_1d(const SimpleTensor<T> &src);

/** Performs an one dimensional inverse DFT on a real input.
 *
 * @param[in] src    Source tensor.
 * @param[in] is_odd (Optional) Specifies if the output has odd dimensions.
 *                   Is used by the inverse variant to reconstruct odd sequences.
 *
 * @return Complex output of length n/2 + 1 due to symmetry.
 */
template <typename T>
SimpleTensor<T> ridft_1d(const SimpleTensor<T> &src, bool is_odd = false);

/**  Performs an one dimensional DFT on a complex input.
 *
 * @param[in] src       Source tensor.
 * @param[in] direction Direction of the DFT.
 *
 * @return Complex output of same length as input.
 */
template <typename T>
SimpleTensor<T> dft_1d(const SimpleTensor<T> &src, FFTDirection direction);

/** Performs a two dimensional DFT on a real input.
 *
 * @param[in] src Source tensor.
 *
 * @return Complex output of length n/2 + 1 across width due to symmetry and height of same size as the input.
 */
template <typename T>
SimpleTensor<T> rdft_2d(const SimpleTensor<T> &src);

/** Performs a two dimensional inverse DFT on a real input.
 *
 * @param[in] src    Source tensor.
 * @param[in] is_odd (Optional) Specifies if the output has odd dimensions across width.
 *                   Is used by the inverse variant to reconstruct odd sequences.
 *
 * @return Complex output of length n/2 + 1 across width due to symmetry and height of same size as the input.
 */
template <typename T>
SimpleTensor<T> ridft_2d(const SimpleTensor<T> &src, bool is_odd = false);

/**  Performs a two dimensional DFT on a complex input.
 *
 * @param[in] src       Source tensor.
 * @param[in] direction Direction of the DFT.
 *
 * @return Complex output of same length as input.
 */
template <typename T>
SimpleTensor<T> dft_2d(const SimpleTensor<T> &src, FFTDirection direction);

/** Performs and DFT based convolution on a real input.
 *
 * @param[in] src       Source tensor.
 * @param[in] w         Weights tensor.
 * @param[in] conv_info Convolution related metadata.
 *
 * @return The output tensor.
 */
template <typename T>
SimpleTensor<T> conv2d_dft(const SimpleTensor<T> &src, const SimpleTensor<T> &w, const PadStrideInfo &conv_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_FFT_H__ */
