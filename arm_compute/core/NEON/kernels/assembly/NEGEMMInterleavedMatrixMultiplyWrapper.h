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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVEDMATRIXMULTIPLYWRAPPER_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVEDMATRIXMULTIPLYWRAPPER_H__

#include "arm_compute/core/NEON/kernels/assembly/Helpers.h"

#include "arm_compute/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
class ITensor;

/** Unit of work for @ref NEGEMMInterleavedMatrixMultiplyWrapper to process */
struct MatrixMultiplyWorkload
{
    /** Constructor
     *
     * @param[in] offset_transformed_b Offset from the start of transformed_b's allocation.
     * @param[in] x0                   First value to process along the X dimension (N).
     * @param[in] xmax                 Last value to process along the X dimension (N).
     * @param[in] k0                   First value to process along the K dimension.
     * @param[in] kmax                 Last value to process along the K dimension.
     * @param[in] multi                Multi index.
     * @param[in] kern_k               Number of elements along K actually processed by the kernel.
     * @param[in] bblocks              Number of x_block processed by the kernel.
     */
    MatrixMultiplyWorkload(unsigned int offset_transformed_b, unsigned int x0, unsigned int xmax, unsigned int k0, unsigned int kmax, unsigned int multi, int kern_k, int bblocks)
        : _offset_transformed_b(offset_transformed_b), _x0(x0), _xmax(xmax), _k0(k0), _kmax(kmax), _multi(multi), _kern_k(kern_k), _bblocks(bblocks)
    {
    }
    unsigned int _offset_transformed_b; /**< Offset from the start of transformed_b's allocation.*/
    unsigned int _x0;                   /**< First value to process along the X dimension (N). */
    unsigned int _xmax;                 /**< Last value to process along the X dimension (N). */
    unsigned int _k0;                   /**< First value to process along the K dimension. */
    unsigned int _kmax;                 /**< Last value to process along the K dimension. */
    unsigned int _multi;                /**< Multi index. */
    int          _kern_k;               /**< Number of elements along K actually processed by the kernel. */
    int          _bblocks;              /**< Number of x_block processed by the kernel. */
};

/** Common interface for the templated wrappers around the matrix multiply NEON assembly implementations */
class NEGEMMInterleavedMatrixMultiplyWrapper
{
public:
    /** Transform the block at the given coordinates
     *
     * @param[in] wl           Workload to process.
     * @param[in] info         Information about the current thread.
     * @param[in] batch_window Window containing iteration information for the M and batch dimensions.
     * @param[in] start_offset Offset relative to the beginning of batch_window to start the processing from.
     * @param[in] end_offset   Offset relative to the beginning of batch_window to stop the processing.
     */
    virtual void transform(const MatrixMultiplyWorkload &wl, const ThreadInfo &info, const Window &batch_window, const Coordinates &start_offset, const Coordinates &end_offset) = 0;
    /** Generate an array of workloads
     *
     * @param[out] workloads Container to store the generated workloads.
     */
    virtual void create_workloads(std::vector<MatrixMultiplyWorkload> &workloads) = 0;
    /** Default destructor */
    virtual ~NEGEMMInterleavedMatrixMultiplyWrapper() = default;
};

/** Equivalent to arm_gemm::GemmInterleaved's strategy::kernel() but using Compute Library types. */
template <typename To, typename Tr, bool use_dot = false>
class NEGEMMInterleavedMatrixMultiplyWrapperTemplate : public NEGEMMInterleavedMatrixMultiplyWrapper
{
public:
    /** Configure the matrix multiplication: C = alpha * A * B + beta * C
     *
     * @param[in]     prepared_a       Already reshaped matrix A.
     * @param[in]     transformed_b    Already reshaped matrix B.
     * @param[out]    tmp_c            Temporary buffer to be used to store intermediate results.
     * @param[in,out] c                Result matrix C.
     * @param[in]     batch_window     Window containing iteration information for the M and batch dimensions.
     * @param[in]     block_sizes      Block sizes to use for the matrix multiplication (A & B must have been reshaped using these same block sizes).
     * @param[in]     params           M, N, K sizes.
     * @param[in]     is_pretransposed Is B also pretransposed ?
     * @param[in]     alpha            Alpha value
     * @param[in]     beta             Beta value
     * @param[in]     max_num_threads  Maximum number of threads that might be used for the calculations.
     */
    void configure(const ITensor *prepared_a, const ITensor *transformed_b, ITensor *tmp_c, ITensor *c, const Window &batch_window, const BlockSizes &block_sizes,
                   const INEGEMMWrapperKernel::Params &params, bool b_is_pretransposed, float alpha, float beta, unsigned int max_num_threads);

    // Inherited methods overridden:
    void transform(const MatrixMultiplyWorkload &wl, const ThreadInfo &info, const Window &batch_window, const Coordinates &start_offset, const Coordinates &end_offset) override;
    void create_workloads(std::vector<MatrixMultiplyWorkload> &workloads) override;

private:
    const ITensor *_prepared_a
    {
        nullptr
    };
    const ITensor               *_transformed_b{ nullptr };
    ITensor                     *_tmp_c{ nullptr };
    ITensor                     *_c{ nullptr };
    unsigned int                 _Nsize{ 0 };
    unsigned int                 _Ksize{ 0 };
    bool                         _transpose_b{ false };
    BlockSizes                   _block_sizes{};
    INEGEMMWrapperKernel::Params _params{};
    Window                       _block_walker{};
    bool                         _b_is_pretransposed{ false };
    Tr                           _alpha{};
    Tr                           _beta{};
};

} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMINTERLEAVEDMATRIXMULTIPLYWRAPPER_H__ */
