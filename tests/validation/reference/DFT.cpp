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
#include "DFT.h"

#include "PadLayer.h"
#include "Permute.h"
#include "Reverse.h"
#include "SliceOperations.h"

#include <cmath>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
/** Performs an one dimensional DFT on a given real sequence.
 *
 * @param[in]  src_ptr Pointer to the real input sequence.
 * @param[in]  N       Size of input sequence.
 * @param[out] dst_ptr Pointer to the complex output sequence.
 * @param[out] K       Size of the output sequence
 */
template <typename T>
void rdft_1d_step(const T *src_ptr, size_t N, T *dst_ptr, size_t K)
{
    for(unsigned int k = 0; k < K; ++k)
    {
        float Xr = 0;
        float Xi = 0;
        for(unsigned int n = 0; n < N; ++n)
        {
            const float alpha = (2 * M_PI * k * n) / N;
            const float val_r = src_ptr[n];
            // Assuming DFT from the R domain thus skipping imaginary calculations
            Xr += val_r * cos(alpha);
            Xi -= val_r * sin(alpha);
        }

        dst_ptr[k * 2]     = Xr;
        dst_ptr[k * 2 + 1] = Xi;
    }
}

/** Performs an one dimensional DFT on a given complex sequence.
 *
 * @param[in]  src_ptr Pointer to the complex input sequence.
 * @param[out] dst_ptr Pointer to the complex output sequence.
 * @param[in]  N       Size of the sequences
 */
template <typename T>
void dft_1d_step(const T *src_ptr, T *dst_ptr, size_t N)
{
    for(unsigned int k = 0; k < N; ++k)
    {
        float Xr = 0;
        float Xi = 0;
        for(unsigned int n = 0; n < N; ++n)
        {
            const float alpha     = (2 * M_PI * k * n) / N;
            const float val_r     = src_ptr[2 * n];
            const float val_i     = src_ptr[2 * n + 1];
            const float cos_alpha = cos(alpha);
            const float sin_alpha = sin(alpha);

            Xr += val_r * cos_alpha + val_i * sin_alpha;
            Xi += val_i * cos_alpha - val_r * sin_alpha;
        }

        dst_ptr[k * 2]     = Xr;
        dst_ptr[k * 2 + 1] = Xi;
    }
}

/** Performs an one dimensional inverse DFT on a given real sequence.
 *
 * @param[in]  src_ptr Pointer to the real input sequence.
 * @param[in]  K       Size of input sequence.
 * @param[out] dst_ptr Pointer to the complex output sequence.
 * @param[out] N       Size of the output sequence
 */
template <typename T>
void irdft_1d_step(const T *src_ptr, size_t K, T *dst_ptr, size_t N)
{
    const bool         is_odd     = N % 2;
    const unsigned int Nleft      = N - K;
    const int          tail_start = is_odd ? K - 1 : K - 2;

    for(unsigned int n = 0; n < N; ++n)
    {
        float xr = 0;
        for(unsigned int k = 0; k < K; ++k)
        {
            const float alpha = (2 * M_PI * k * n) / N;
            xr += src_ptr[2 * k] * cos(alpha) - src_ptr[2 * k + 1] * sin(alpha);
        }

        unsigned int j = tail_start;
        for(unsigned int k = 0; k < Nleft; ++k)
        {
            const float alpha = (2 * M_PI * (k + K) * n) / N;
            xr += src_ptr[2 * j] * cos(alpha) + src_ptr[2 * j + 1] * sin(alpha);
            --j;
        }

        dst_ptr[n] = xr;
    }
}

/** Performs an one dimensional inverse DFT on a given complex sequence.
 *
 * @param[in]  src_ptr Pointer to the complex input sequence.
 * @param[out] dst_ptr Pointer to the complex output sequence.
 * @param[in]  N       Size of the sequences
 */
template <typename T>
void idft_1d_step(const T *src_ptr, T *dst_ptr, size_t N)
{
    for(unsigned int n = 0; n < N; ++n)
    {
        float xr = 0;
        float xi = 0;
        for(unsigned int k = 0; k < N; ++k)
        {
            const float alpha     = (2 * M_PI * k * n) / N;
            const float cos_alpha = cos(alpha);
            const float sin_alpha = sin(alpha);
            const float val_r     = src_ptr[2 * k];
            const float val_i     = src_ptr[2 * k + 1];

            xr += val_r * cos_alpha - val_i * sin_alpha;
            xi += val_i * cos_alpha + val_r * sin_alpha;
        }

        dst_ptr[2 * n]     = xr;
        dst_ptr[2 * n + 1] = xi;
    }
}

template <typename T>
SimpleTensor<T> rdft_1d_core(const SimpleTensor<T> &src, FFTDirection direction, bool is_odd)
{
    // Performs only rdft
    ARM_COMPUTE_ERROR_ON(direction == FFTDirection::Forward && src.num_channels() != 1);
    ARM_COMPUTE_ERROR_ON(direction == FFTDirection::Inverse && src.num_channels() != 2);

    const unsigned int inverse_tail = is_odd ? 1 : 0;
    const unsigned int N            = src.shape()[0];
    const unsigned int K            = direction == FFTDirection::Forward ? N / 2 + 1 : (N - 1) * 2 + inverse_tail;
    const unsigned int num_channels = direction == FFTDirection::Forward ? 2 : 1;

    TensorShape dst_shape = src.shape();
    dst_shape.set(0, K);

    SimpleTensor<T> dst(dst_shape, src.data_type(), num_channels);

    const unsigned int upper_dims = src.shape().total_size_upper(1);
    for(unsigned int du = 0; du < upper_dims; ++du)
    {
        const T *src_row_ptr = src.data() + du * N * src.num_channels();
        T       *dst_row_ptr = dst.data() + du * K * dst.num_channels();
        direction == FFTDirection::Forward ? rdft_1d_step(src_row_ptr, N, dst_row_ptr, K) : irdft_1d_step(src_row_ptr, N, dst_row_ptr, K);
    }

    return dst;
}

template <typename T>
SimpleTensor<T> dft_1d_core(const SimpleTensor<T> &src, FFTDirection direction)
{
    ARM_COMPUTE_ERROR_ON(src.num_channels() != 2);

    const unsigned int N = src.shape()[0];

    SimpleTensor<T> dst(src.shape(), src.data_type(), src.num_channels());

    const unsigned int upper_dims = src.shape().total_size_upper(1);
    for(unsigned int du = 0; du < upper_dims; ++du)
    {
        const T *src_row_ptr = src.data() + du * N * src.num_channels();
        T       *dst_row_ptr = dst.data() + du * N * dst.num_channels();
        direction == FFTDirection::Forward ? dft_1d_step(src_row_ptr, dst_row_ptr, N) : idft_1d_step(src_row_ptr, dst_row_ptr, N);
    }

    return dst;
}

/** Scale a tensor by a given scaling factor.
 *
 * @param[in,out] tensor         Tensor to scale.
 * @param[in]     scaling_factor Scaling to scale the tensor data with.
 */
template <typename T>
void scale(SimpleTensor<T> &tensor, T scaling_factor)
{
    const int total_elements = tensor.num_elements() * tensor.num_channels();
    T        *data_ptr       = tensor.data();
    for(int i = 0; i < total_elements; ++i)
    {
        data_ptr[i] /= scaling_factor;
    }
}

/** Performs a complex element-wise multiplication with reduction across the channels axis.
 *
 * @param[in] input   Input tensor.
 * @param[in] weights Weights tensor.
 *
 * @return Output tensor.
 */
template <typename T>
SimpleTensor<T> complex_mul_and_reduce(const SimpleTensor<T> &input, const SimpleTensor<T> &weights)
{
    const int W  = input.shape().x();
    const int H  = input.shape().y();
    const int Ci = input.shape().z();
    const int Co = weights.shape()[3];
    const int N  = input.shape().total_size() / (W * H * Ci);

    TensorShape output_shape = input.shape();
    output_shape.set(2, Co);
    SimpleTensor<T> dst(output_shape, input.data_type(), input.num_channels());

    // MemSet dst memory to zero
    std::memset(dst.data(), 0, dst.size());

    for(int b = 0; b < N; ++b)
    {
        for(int co = 0; co < Co; ++co)
        {
            for(int ci = 0; ci < Ci; ++ci)
            {
                for(int h = 0; h < H; ++h)
                {
                    for(int w = 0; w < W; ++w)
                    {
                        size_t            i_index  = w + h * W + ci * H * W + b * H * W * Ci;
                        size_t            w_index  = w + h * W + ci * H * W + co * H * W * Ci;
                        size_t            o_index  = w + h * W + co * H * W + b * H * W * Co;
                        const Coordinates i_coords = index2coords(input.shape(), i_index);
                        const Coordinates w_coords = index2coords(weights.shape(), w_index);
                        const Coordinates o_coords = index2coords(dst.shape(), o_index);

                        auto i_ptr = static_cast<const T *>(input(i_coords));
                        auto w_ptr = static_cast<const T *>(weights(w_coords));
                        auto o_ptr = static_cast<T *>(dst(o_coords));

                        const T Rin = i_ptr[0];
                        const T Iin = i_ptr[1];
                        const T Rw  = w_ptr[0];
                        const T Iw  = w_ptr[1];

                        o_ptr[0] += Rin * Rw - Iin * Iw;
                        o_ptr[1] += Rin * Iw + Rw * Iin;
                    }
                }
            }
        }
    }
    return dst;
}
} // namespace

template <typename T>
SimpleTensor<T> rdft_1d(const SimpleTensor<T> &src)
{
    return rdft_1d_core(src, FFTDirection::Forward, false);
}

template <typename T>
SimpleTensor<T> ridft_1d(const SimpleTensor<T> &src, bool is_odd)
{
    auto dst = rdft_1d_core(src, FFTDirection::Inverse, is_odd);

    const T scaling_factor = dst.shape()[0];
    scale(dst, scaling_factor);

    return dst;
}

template <typename T>
SimpleTensor<T> dft_1d(const SimpleTensor<T> &src, FFTDirection direction)
{
    auto dst = dft_1d_core(src, direction);
    if(direction == FFTDirection::Inverse)
    {
        const T scaling_factor = dst.shape()[0];
        scale(dst, scaling_factor);
    }
    return dst;
}

template <typename T>
SimpleTensor<T> rdft_2d(const SimpleTensor<T> &src)
{
    ARM_COMPUTE_ERROR_ON(src.num_channels() != 1);
    constexpr FFTDirection direction = FFTDirection::Forward;

    auto first_pass  = rdft_1d_core(src, direction, false);
    auto transposed  = permute(first_pass, PermutationVector(1U, 0U));
    auto second_pass = dft_1d_core(transposed, direction);
    return permute(second_pass, PermutationVector(1U, 0U));
}

template <typename T>
SimpleTensor<T> ridft_2d(const SimpleTensor<T> &src, bool is_odd)
{
    ARM_COMPUTE_ERROR_ON(src.num_channels() != 2);
    constexpr FFTDirection direction = FFTDirection::Inverse;

    auto transposed   = permute(src, PermutationVector(1U, 0U));
    auto first_pass   = dft_1d_core(transposed, direction);
    auto transposed_2 = permute(first_pass, PermutationVector(1U, 0U));
    auto dst          = rdft_1d_core(transposed_2, direction, is_odd);

    const T scaling_factor = dst.shape()[0] * dst.shape()[1];
    scale(dst, scaling_factor);
    return dst;
}

template <typename T>
SimpleTensor<T> dft_2d(const SimpleTensor<T> &src, FFTDirection direction)
{
    ARM_COMPUTE_ERROR_ON(src.num_channels() != 2);

    if(direction == FFTDirection::Forward)
    {
        auto first_pass  = dft_1d_core(src, direction);
        auto transposed  = permute(first_pass, PermutationVector(1U, 0U));
        auto second_pass = dft_1d_core(transposed, direction);
        return permute(second_pass, PermutationVector(1U, 0U));
    }
    else
    {
        auto transposed   = permute(src, PermutationVector(1U, 0U));
        auto first_pass   = dft_1d_core(transposed, direction);
        auto transposed_2 = permute(first_pass, PermutationVector(1U, 0U));
        auto dst          = dft_1d_core(transposed_2, direction);

        const T scaling_factor = dst.shape()[0] * dst.shape()[1];
        scale(dst, scaling_factor);

        return dst;
    }
}

template <typename T>
SimpleTensor<T> conv2d_dft(const SimpleTensor<T> &src, const SimpleTensor<T> &w, const PadStrideInfo &conv_info)
{
    // Pad input to full padding
    const PaddingList padding_in = { { 0, w.shape()[0] - 1 }, { 0, w.shape()[1] - 1 } };
    auto              padded_src = pad_layer(src, padding_in);

    // Flip weights
    std::vector<uint32_t>  axis_v = { 0, 1 };
    SimpleTensor<uint32_t> axis{ TensorShape(2U), DataType::U32 };
    std::copy(axis_v.begin(), axis_v.begin() + axis.shape().x(), axis.data());
    auto flipped_w = reverse(w, axis);

    // Pad weights to have the same size as input
    const PaddingList paddings_w = { { 0, src.shape()[0] - 1 }, { 0, src.shape()[1] - 1 } };
    auto              padded_w   = pad_layer(flipped_w, paddings_w);

    // Transform input and weights to frequency domain
    auto Fsrc = rdft_2d(padded_src);
    auto Fw   = rdft_2d(padded_w);

    // Perform dot product
    auto Fdst = complex_mul_and_reduce(Fsrc, Fw);

    // Transform output back to frequency domain
    auto conv_res = ridft_2d(Fdst);

    // Slice output
    const int start_left = w.shape().x() - conv_info.pad_left() - 1;
    const int start_top  = w.shape().y() - conv_info.pad_top() - 1;
    const int end_right  = conv_res.shape().x() - (w.shape().x() - conv_info.pad_right() - 1);
    const int end_botton = conv_res.shape().y() - (w.shape().y() - conv_info.pad_bottom() - 1);
    return slice(conv_res, Coordinates(start_left, start_top), Coordinates(end_right, end_botton));
}

template SimpleTensor<float> rdft_1d(const SimpleTensor<float> &src);
template SimpleTensor<float> ridft_1d(const SimpleTensor<float> &src, bool is_odd);
template SimpleTensor<float> dft_1d(const SimpleTensor<float> &src, FFTDirection direction);

template SimpleTensor<float> rdft_2d(const SimpleTensor<float> &src);
template SimpleTensor<float> ridft_2d(const SimpleTensor<float> &src, bool is_odd);
template SimpleTensor<float> dft_2d(const SimpleTensor<float> &src, FFTDirection direction);

template SimpleTensor<float> conv2d_dft(const SimpleTensor<float> &src, const SimpleTensor<float> &w, const PadStrideInfo &conv_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
