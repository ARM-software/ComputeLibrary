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
#ifndef __ARM_COMPUTE_ASSEMBLY_HELPERS_H__
#define __ARM_COMPUTE_ASSEMBLY_HELPERS_H__

#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/Utils.h"

#include "arm_compute/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"
#include "arm_compute/core/NEON/kernels/assembly/arm_gemm.hpp"

namespace arm_compute
{
/** Block sizes to use to break the M, N, K dimension */
struct BlockSizes
{
    unsigned int k_block{ 0 };             /**< Block size alon the K dimension */
    unsigned int x_block{ 0 };             /**< Block size along the N (x) dimension */
    unsigned int m_round{ 0 };             /**< Block size along the M dimension (Must be a multiple of strategy_out_height) */
    unsigned int strategy_out_height{ 0 }; /**< Number of rows (M) processed by the selected strategy */
};

/** Extracts the kernel description of the selected kernel by the GEMM backend heuristics
 *
 * @param[in] input_type        Data type of the input tensor.
 * @param[in] ci                CPU information.
 * @param[in] num_threads       Maximum number of threads that might be used for the calculations.
 * @param[in] p                 M, N, K sizes.
 * @param[in] alpha             Alpha value.
 * @param[in] beta              Beta value.
 * @param[in] pretranspose_hint Is B also pretransposed ?
 *
 * @return Kernel description that the assembly heuristics picked for the given configuration
 */
arm_gemm::KernelDescription get_gemm_info(DataType                            input_type,
                                          const CPUInfo                      &ci,
                                          const unsigned int                  num_threads,
                                          const INEGEMMWrapperKernel::Params &p,
                                          float                               alpha,
                                          float                               beta,
                                          bool                                pretranspose_hint);

/** Calculate the recommended block sizes to use based on the CPU cache sizes and the strategy which will be used
 *
 * @param[in] ci CPU information.
 * @param[in] M  M dimension.
 * @param[in] N  N dimension.
 * @param[in] K  K dimension.
 *
 * @return Recommeded block sizes to use for the given M, N, K dimensions.
 */
template <typename strategy>
BlockSizes calculate_block_sizes(const CPUInfo &ci, unsigned int M, unsigned int N, unsigned int K)
{
    BlockSizes bs;

    using Toi = typename strategy::operand_type;

    const unsigned int L1_size = ci.get_L1_cache_size();
    const unsigned int L2_size = ci.get_L2_cache_size();

    // Work out blocking parameters

    // k_block: Find out how much of the larger array can be loaded into half the cache.
    // This should account for associative caches.
    bs.k_block = (L1_size / 2) / (sizeof(Toi) * (std::max(strategy::out_width(), strategy::out_height())));

    // Needs to be (at least a single) multiple of the K unroll level.
    bs.k_block /= strategy::k_unroll();
    bs.k_block = std::max(bs.k_block, 1U) * strategy::k_unroll();

    // Now tune to presented problem size; this is how many blocks we need.
    int num_k_blocks = DIV_CEIL(K, bs.k_block);

    // So divide the space equally into that many blocks.
    bs.k_block = DIV_CEIL(K, num_k_blocks);

    // And round UP to the K unroll level required.
    bs.k_block = ceil_to_multiple(bs.k_block, strategy::k_unroll());

    // x_block: Work out how many rows (of length k_block) will fit in the L2
    // Don't allocate more than 90% of the L2 to allow for overheads, and subtract off the L1 contents.
    bs.x_block = (((L2_size * 9) / 10) - (bs.k_block * sizeof(Toi) * (strategy::out_width() + strategy::out_height()))) / (sizeof(Toi) * bs.k_block);

    // Needs to be (at least a single) multiple of the kernel output width.
    bs.x_block /= strategy::out_width();
    bs.x_block = std::max(bs.x_block, 1U) * strategy::out_width();

    // And tune to the presented problem size.
    int num_x_blocks = DIV_CEIL(N, bs.x_block);
    bs.x_block       = DIV_CEIL(N, num_x_blocks);

    bs.x_block = ceil_to_multiple(bs.x_block, strategy::out_width());

    // Work out the rounded size of M - needed for some buffers.
    bs.m_round             = ceil_to_multiple(M, strategy::out_height());
    bs.strategy_out_height = strategy::out_height();

    return bs;
}

} // namespace arm_compute
#endif /* __ARM_COMPUTE_ASSEMBLY_HELPERS_H__ */
