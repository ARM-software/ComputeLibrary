/*
 * Copyright (c) 2018 ARM Limited.
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

#include "NEGEMMInterleavedStrategies.h"

namespace arm_compute
{
namespace
{
template <typename InputType, bool use_dot = false>
BlockSizes calculate_block_sizes_template(const CPUInfo &ci, unsigned int M, unsigned int N, unsigned int K)
{
    using strategy = typename Kernel<InputType, use_dot>::strategy;
    return calculate_block_sizes<strategy>(ci, M, N, K);
}
} // namespace

const char *get_strategy_name(DataType input_type, bool use_dot)
{
    switch(input_type)
    {
        case DataType::F32:
            return Kernel<float>::name;
#ifdef __aarch64__
        case DataType::U8:
        case DataType::QASYMM8:
            if(use_dot)
            {
                return Kernel<uint8_t, true>::name;
            }
            else
            {
                return Kernel<uint8_t, false>::name;
            }
        case DataType::S8:
            if(use_dot)
            {
                return Kernel<int8_t, true>::name;
            }
            else
            {
                return Kernel<int8_t, false>::name;
            }
#endif /* __aarch64__ */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            return Kernel<__fp16>::name;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("DataType not supported");
            break;
    }
}

BlockSizes calculate_block_sizes_from_data_type(const CPUInfo &ci, unsigned int M, unsigned int N, unsigned int K, DataType input_type, bool use_dot)
{
    switch(input_type)
    {
        case DataType::F32:
            return calculate_block_sizes_template<float>(ci, M, N, K);
#ifdef __aarch64__
        case DataType::U8:
        case DataType::QASYMM8:
            if(use_dot)
            {
                return calculate_block_sizes_template<uint8_t, true>(ci, M, N, K);
            }
            else
            {
                return calculate_block_sizes_template<uint8_t, false>(ci, M, N, K);
            }
        case DataType::S8:
            if(use_dot)
            {
                return calculate_block_sizes_template<int8_t, true>(ci, M, N, K);
            }
            else
            {
                return calculate_block_sizes_template<int8_t, false>(ci, M, N, K);
            }
#endif /* __aarch64__ */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            return calculate_block_sizes_template<__fp16>(ci, M, N, K);
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("DataType not supported");
            break;
    }
}
} // namespace arm_compute
