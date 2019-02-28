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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVEDSTRATEGIES_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVEDSTRATEGIES_H__

#include "../arm_gemm/utils.hpp"
#include "arm_gemm.hpp"

#include "../arm_gemm/mergeresults.hpp"
#include "../arm_gemm/transform.hpp"

#include "../arm_gemm/kernels/a32_sgemm_8x6.hpp"
#include "../arm_gemm/kernels/a64_gemm_s8_12x8.hpp"
#include "../arm_gemm/kernels/a64_gemm_s8_4x4.hpp"
#include "../arm_gemm/kernels/a64_gemm_u8_12x8.hpp"
#include "../arm_gemm/kernels/a64_gemm_u8_4x4.hpp"
#include "../arm_gemm/kernels/a64_hgemm_24x8.hpp"
#include "../arm_gemm/kernels/a64_sgemm_12x8.hpp"
#include "../arm_gemm/kernels/sve_interleaved_fp16_mla_3VLx8.hpp"
#include "../arm_gemm/kernels/sve_interleaved_fp32_mla_3VLx8.hpp"
#include "../arm_gemm/kernels/sve_interleaved_s8s32_dot_3VLx8.hpp"
#include "../arm_gemm/kernels/sve_interleaved_u8u32_dot_3VLx8.hpp"

namespace arm_compute
{
namespace detail
{
/** GEMM Interleaved Strategy interface */
class IInterleavedStrategy
{
public:
    /** Virtual Destructor */
    virtual ~IInterleavedStrategy() = default;
    /** Return output height of the interleaved strategy
     *
     * @return Output height of strategy
     */
    virtual unsigned int out_height() const = 0;
    /** Instantiate and configure a prepareB Kernel
     *
     * @param[in] b             Input tensor B.
     * @param[in] transformed_b Reshaped tensor B.
     * @param[in] params        GM, N, K sizes.
     * @param[in] ci            CPUInfo to be used for kernel configuration.
     *
     * @return A wrapped specialized prepareB kernel
     */
    virtual std::unique_ptr<NEGEMMInterleavedPrepareBWrapperKernel> instantiate_prepareB(const ITensor                      *b,
                                                                                         ITensor                            *transformed_b,
                                                                                         const INEGEMMWrapperKernel::Params &params,
                                                                                         const CPUInfo                      &ci) = 0;
    /** Instantiate and configure a transformA Kernel
     *
     * @param[in] a             Input tensor A.
     * @param[in] transformed_a Reshaped tensor A.
     * @param[in] block_walker  Window representing the layout of the matrix's blocks.
     * @param[in] params        M, N, K sizes.
     *
     * @return A wrapped specialized transformA kernel
     */
    virtual std::unique_ptr<NEGEMMInterleavedTransformAWrapper> instantiate_transformA(const ITensor                      *a,
                                                                                       ITensor                            *transformed_a,
                                                                                       const Window                       &block_walker,
                                                                                       const INEGEMMWrapperKernel::Params &params) = 0;
    /** Instantiate and configure a prepareB Kernel
     *
     * @param transformed_a  Already reshaped tensor A.
     * @param transformed_b  Already reshaped tensor B.
     * @param tmp_c          Temporary buffer to be used to store intermediate results.
     * @param c              Result tensor C.
     * @param block_walker   Window containing iteration information for the M and batch dimensions.
     * @param block_sizes    Block sizes to use for the matrix multiplication (A & B must have been reshaped using these same block sizes).
     * @param params         M, N, K sizes.
     * @param alpha          Alpha value
     * @param beta           Beta value
     * @param pretranspose_b Is B also pretransposed ?
     * @param num_threads    Maximum number of threads that might be used for the calculations.
     *
     * @return A wrapped specialized MatrixMultiply kernel
     */
    virtual std::unique_ptr<NEGEMMInterleavedMatrixMultiplyWrapper> instantiate_matrix_multiply(const ITensor *transformed_a, const ITensor *transformed_b, ITensor *tmp_c, ITensor *c,
                                                                                                const Window &block_walker, const BlockSizes &block_sizes,
                                                                                                const INEGEMMWrapperKernel::Params &params, float alpha, float beta, bool pretranspose_b,
                                                                                                unsigned int num_threads) = 0;
    /** Calculates the block sizes of a given strategy
     *
     * @param[in] ci     CPUInfo to be used for kernel configuration.
     * @param[in] params M, N, K sizes.
     *
     * @return BlockSizes for a given strategy
     */
    virtual BlockSizes calculate_block_sizes_for_strategy(const CPUInfo &ci, const INEGEMMWrapperKernel::Params &params) = 0;
};

/** Interleaved Strategy class */
template <typename StrategyType>
class InterleavedStrategy : public IInterleavedStrategy
{
public:
    using strategy = StrategyType;

public:
    // Inherited methods overridden
    unsigned int out_height() const override
    {
        return strategy::out_height();
    }
    std::unique_ptr<NEGEMMInterleavedPrepareBWrapperKernel> instantiate_prepareB(const ITensor                      *b,
                                                                                 ITensor                            *transformed_b,
                                                                                 const INEGEMMWrapperKernel::Params &params,
                                                                                 const CPUInfo                      &ci) override
    {
        auto prepare_b = support::cpp14::make_unique<NEGEMMInterleavedPrepareBWrapperKernelTemplate<strategy>>();
        prepare_b->configure(b, transformed_b, false, ci, params);
        return std::move(prepare_b);
    }
    std::unique_ptr<NEGEMMInterleavedTransformAWrapper> instantiate_transformA(const ITensor                      *a,
                                                                               ITensor                            *transformed_a,
                                                                               const Window                       &block_walker,
                                                                               const INEGEMMWrapperKernel::Params &params) override
    {
        auto transform_a = support::cpp14::make_unique<NEGEMMInterleavedTransformAWrapperTemplate<strategy>>();
        transform_a->configure(a, transformed_a, false, block_walker, params);
        return std::move(transform_a);
    }
    std::unique_ptr<NEGEMMInterleavedMatrixMultiplyWrapper> instantiate_matrix_multiply(const ITensor *transformed_a, const ITensor *transformed_b, ITensor *tmp_c, ITensor *c,
                                                                                        const Window &block_walker, const BlockSizes &block_sizes,
                                                                                        const INEGEMMWrapperKernel::Params &params, float alpha, float beta, bool pretranspose_b,
                                                                                        unsigned int num_threads) override
    {
        auto matrix_multiply = support::cpp14::make_unique<NEGEMMInterleavedMatrixMultiplyWrapperTemplate<strategy>>();
        matrix_multiply->configure(transformed_a, transformed_b, tmp_c, c, block_walker, block_sizes, params, pretranspose_b, alpha, beta, num_threads);
        return std::move(matrix_multiply);
    }

    BlockSizes calculate_block_sizes_for_strategy(const CPUInfo &ci, const INEGEMMWrapperKernel::Params &params) override
    {
        return calculate_block_sizes<strategy>(ci, params.M, params.N, params.K);
    }
};

/** Create the backend GEMM strategy to use given the provided kernel info
 *
 * @param[in] kernel_name Kernel name of the backend strategy to instantiate
 *
 * @return The requested kernel strategy if exists else nullptr
 */
std::unique_ptr<IInterleavedStrategy> create_strategy(const std::string &kernel_name)
{
#if defined(__arm__)
    if(kernel_name.find("sgemm_8x6") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::sgemm_8x6>>();
    }
#endif // defined(__arm__)
#if defined(__aarch64__)
    if(kernel_name.find("gemm_s8_4x4") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::gemm_s8_4x4>>();
    }
    if(kernel_name.find("gemm_s8_12x8") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::gemm_s8_12x8>>();
    }
    if(kernel_name.find("gemm_u8_4x4") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::gemm_u8_4x4>>();
    }
    if(kernel_name.find("gemm_u8_12x8") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::gemm_u8_12x8>>();
    }
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    if(kernel_name.find("hgemm_24x8") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::hgemm_24x8>>();
    }
#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    if(kernel_name.find("sgemm_12x8") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::sgemm_12x8>>();
    }
#if defined(__ARM_FEATURE_SVE)
    if(kernel_name.find("interleaved_fp16_mla_3VLx8") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::interleaved_fp16_mla_3VLx8>>();
    }
    if(kernel_name.find("interleaved_fp32_mla_3VLx8") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::interleaved_fp32_mla_3VLx8>>();
    }
    if(kernel_name.find("interleaved_s8s32_dot_3VLx8") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::interleaved_s8s32_dot_3VLx8>>();
    }
    if(kernel_name.find("interleaved_u8u32_dot_3VLx8") != std::string::npos)
    {
        return support::cpp14::make_unique<InterleavedStrategy<arm_gemm::interleaved_u8u32_dot_3VLx8>>();
    }
#endif // defined(__ARM_FEATURE_SVE)
#endif // defined(__aarch64__)_
    return nullptr;
}
} // namespace detail
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMINTERLEAVEDSTRATEGIES_H__ */
