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
#include "GEMM.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> gemm(const SimpleTensor<T> &a, const SimpleTensor<T> &b, const SimpleTensor<T> &c, float alpha, float beta)
{
    // Create reference
    SimpleTensor<T> dst{ c.shape(), c.data_type(), 1 };

    // Compute reference
    const int M = a.shape().y();
    const int N = b.shape().x();
    const int K = a.shape().x();
    const int D = a.shape().z(); // Number of matrices in a batch
    const int W = a.shape()[3];  // Number of batched-gemm (Winograd case)

    const int a_stride_z = K * M;
    const int a_stride_w = K * M * D;

    const int b_stride_z = b.shape().num_dimensions() > 2 ? N * K : 0;     // Do not slide the matrix B along the 3th dimension in case matrix B has less than 3 dimensions
    int       b_stride_w = b.shape().num_dimensions() > 3 ? K * N * D : 0; // Do not slide the matrix B along the 4th dimension in case matrix B has less than 4 dimensions

    // Note: There are 3 gemm types: batched-gemm, multi-gemm, and batched of multi-gemms. The third dimension of tensor b is overloaded when tensor b has exactly 3 dimensions:
    // it can be either number of batches or multis. Batched-GEMM computation is detected only when the third dimension of "a" and "c" tensors is 1 and the number of dimensions is 4
    const bool is_batched_gemm = b.shape().num_dimensions() == 3 && a.shape().num_dimensions() == 4 && c.shape().num_dimensions() == 4 && a.shape()[2] == 1 && c.shape()[2] == 1;

    // Batched-GEMM
    if(is_batched_gemm)
    {
        b_stride_w = b_stride_z;
    }

    const int c_stride_z = N * M;
    const int c_stride_w = N * M * D;

#if defined(_OPENMP) && !(defined(__arm__) && defined(__ANDROID__))
    #pragma omp parallel for collapse(2)
#endif /* _OPENMP */
    for(int w = 0; w < W; ++w)
    {
        for(int depth = 0; depth < D; ++depth)
        {
            const int base_addr_a = depth * a_stride_z + w * a_stride_w;
            const int base_addr_b = depth * b_stride_z + w * b_stride_w;
            const int base_addr_c = depth * c_stride_z + w * c_stride_w;

            for(int row = 0; row < M; ++row)
            {
                for(int col = 0; col < N; ++col)
                {
                    T acc(0);

                    for(int k = 0; k < K; ++k)
                    {
                        acc += a[base_addr_a + k + row * K] * b[base_addr_b + col + k * N];
                    }

                    // Finalize the result: alpha * A * B + beta * C
                    dst[base_addr_c + col + row * N] = alpha * acc + beta * c[base_addr_c + col + row * N];
                }
            }
        }
    }

    return dst;
}

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> gemm_mixed_precision(const SimpleTensor<T> &a, const SimpleTensor<T> &b, const SimpleTensor<T> &c, float alpha, float beta)
{
    // GEMM mixed-precision combines F32 accumulators with F16 multiplications
    // Create reference
    SimpleTensor<T> dst{ c.shape(), c.data_type(), 1 };

    // Compute reference
    const int M = a.shape().y();
    const int N = b.shape().x();
    const int K = a.shape().x();
    const int D = a.shape().z(); // Number of matrices in a batch
    const int W = a.shape()[3];  // Number of batched-gemm (Winograd case)

    const int a_stride_z = K * M;
    const int a_stride_w = K * M * D;

    const int b_stride_z = b.shape().num_dimensions() > 2 ? N * K : 0;     // Do not slide the matrix B along the 3th dimension in case matrix B has less than 3 dimensions
    int       b_stride_w = b.shape().num_dimensions() > 3 ? K * N * D : 0; // Do not slide the matrix B along the 4th dimension in case matrix B has less than 4 dimensions

    // Note: There are 3 gemm types: batched-gemm, multi-gemm, and batched of multi-gemms. The third dimension of tensor b is overloaded when tensor b has exactly 3 dimensions:
    // it can be either number of batches or multis. Batched-GEMM computation is detected only when the third dimension of "a" and "c" tensors is 1 and the number of dimensions is 4
    const bool is_batched_gemm = b.shape().num_dimensions() == 3 && a.shape().num_dimensions() == 4 && c.shape().num_dimensions() == 4 && a.shape()[2] == 1 && c.shape()[2] == 1;

    // Batched-GEMM
    if(is_batched_gemm)
    {
        b_stride_w = b_stride_z;
    }

    const int c_stride_z = N * M;
    const int c_stride_w = N * M * D;

#if defined(_OPENMP) && !(defined(__arm__) && defined(__ANDROID__))
    #pragma omp parallel for collapse(2)
#endif /* _OPENMP */
    for(int w = 0; w < W; ++w)
    {
        for(int depth = 0; depth < D; ++depth)
        {
            const int base_addr_a = depth * a_stride_z + w * a_stride_w;
            const int base_addr_b = depth * b_stride_z + w * b_stride_w;
            const int base_addr_c = depth * c_stride_z + w * c_stride_w;

            for(int row = 0; row < M; ++row)
            {
                for(int col = 0; col < N; ++col)
                {
                    float acc(0);

                    for(int k = 0; k < K; ++k)
                    {
                        acc += static_cast<float>(a[base_addr_a + k + row * K] * b[base_addr_b + col + k * N]);
                    }

                    // Finalize the result: alpha * A * B + beta * C
                    dst[base_addr_c + col + row * N] = static_cast<T>(alpha * acc + beta * c[base_addr_c + col + row * N]);
                }
            }
        }
    }

    return dst;
}

template SimpleTensor<float> gemm(const SimpleTensor<float> &a, const SimpleTensor<float> &b, const SimpleTensor<float> &c, float alpha, float beta);
template SimpleTensor<half> gemm(const SimpleTensor<half> &a, const SimpleTensor<half> &b, const SimpleTensor<half> &c, float alpha, float beta);
template SimpleTensor<half> gemm_mixed_precision(const SimpleTensor<half> &a, const SimpleTensor<half> &b, const SimpleTensor<half> &c, float alpha, float beta);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
