/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#include "arm_compute/core/NEON/kernels/assembly/Helpers.h"

#include "arm_compute/core/NEON/kernels/assembly/arm_gemm.hpp"

namespace arm_compute
{
arm_gemm::KernelDescription get_gemm_info(DataType                            input_type,
                                          const CPUInfo                      &ci,
                                          const unsigned int                  num_threads,
                                          const INEGEMMWrapperKernel::Params &p,
                                          float                               alpha,
                                          float                               beta,
                                          bool                                pretranspose_hint)
{
    switch(input_type)
    {
#ifdef __aarch64__
        case DataType::QASYMM8:
        case DataType::U8:
        {
            arm_gemm::GemmArgs<uint32_t> args(&ci, p.M, p.N, p.K, p.batches, p.multis, false, false, alpha, beta, num_threads, pretranspose_hint);
            return arm_gemm::get_gemm_method<uint8_t, uint32_t>(args);
        }
        case DataType::S8:
        {
            arm_gemm::GemmArgs<int32_t> args(&ci, p.M, p.N, p.K, p.batches, p.multis, false, false, alpha, beta, num_threads, pretranspose_hint);
            return arm_gemm::get_gemm_method<int8_t, int32_t>(args);
        }
#endif // __aarch64__
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            arm_gemm::GemmArgs<__fp16> args(&ci, p.M, p.N, p.K, p.batches, p.multis, false, false, alpha, beta, num_threads, pretranspose_hint);
            return arm_gemm::get_gemm_method<__fp16, __fp16>(args);
        }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
        {
            arm_gemm::GemmArgs<float> args(&ci, p.M, p.N, p.K, p.batches, p.multis, false, false, alpha, beta, num_threads, pretranspose_hint);
            return arm_gemm::get_gemm_method<float, float>(args);
        }
        default:
            return arm_gemm::KernelDescription();
    }
}
} // namespace arm_compute
