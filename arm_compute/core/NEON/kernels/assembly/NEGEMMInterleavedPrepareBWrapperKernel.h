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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVEDPREPAREBWRAPPERKERNEL_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVEDPREPAREBWRAPPERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/kernels/assembly/Helpers.h"
#include "arm_compute/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"

namespace arm_compute
{
class ITensor;

/** Unit of work for @ref NEGEMMInterleavedPrepareBWrapperKernel to process */
struct PrepareBWorkload
{
    /** Constructor
     *
     * @param[in] offset_b             Offset from the start of b's allocation
     * @param[in] offset_transformed_b Offset from the start of transformed_b's allocation.
     * @param[in] x0                   First value to process along the X dimension (N).
     * @param[in] xmax                 Last value to process along the X dimension (N).
     * @param[in] k0                   First value to process along the K dimension.
     * @param[in] kmax                 Last value to process along the K dimension.
     */
    PrepareBWorkload(unsigned int offset_b, unsigned int offset_transformed_b, unsigned int x0, unsigned int xmax, unsigned int k0, unsigned int kmax)
        : _offset_b(offset_b), _offset_transformed_b(offset_transformed_b), _x0(x0), _xmax(xmax), _k0(k0), _kmax(kmax)
    {
    }
    unsigned int _offset_b;             /**< Offset from the start of b's allocation.*/
    unsigned int _offset_transformed_b; /**< Offset from the start of transformed_b's allocation.*/
    unsigned int _x0;                   /**< First value to process along the X dimension (N). */
    unsigned int _xmax;                 /**< Last value to process along the X dimension (N). */
    unsigned int _k0;                   /**< First value to process along the K dimension. */
    unsigned int _kmax;                 /**< Last value to process along the K dimension. */
};

/** Common interface for the templated wrappers around the B reshape NEON assembly implementations */
class NEGEMMInterleavedPrepareBWrapperKernel : public INEKernel
{
public:
    /** Transform the block at the given coordinates
     *
     * @param[in] wl   Workload to process.
     * @param[in] info Information about the current thread.
     */
    virtual void transform(const PrepareBWorkload &wl, const ThreadInfo &info) = 0;
    /** Generate an array of workloads
     *
     * @param[out] workloads Container to store the generated workloads.
     */
    virtual void create_workloads(std::vector<PrepareBWorkload> &workloads) = 0;
    /** Return the block_sizes used to resape B
     *
     * The same block sizes must be used to reshape A and for the matrix multiplication
     *
     * @return The block sizes used to reshape B.
     */
    virtual BlockSizes block_sizes() const = 0;

    // Inherited methods overridden:
    const char *name() const override
    {
        return "NEGEMMInterleavedPrepareBWrapperKernel";
    }

    bool is_parallelisable() const override
    {
        return false; // Can't run on arbitrary windows but can be parallelised using an array of workloads
    }
};

/** Equivalent to arm_gemm::GemmInterleaved's strategy::transforms::PrepareB() but using Compute Library types.
 */
template <typename To, bool use_dot = false>
class NEGEMMInterleavedPrepareBWrapperKernelTemplate : public NEGEMMInterleavedPrepareBWrapperKernel
{
public:
    /** Configure the reshape B routine.
     *
     * @param[in]  b             Input matrix B.
     * @param[out] transformed_b Reshaped matrix B.
     * @param[in]  transpose_b   Also transpose B ?
     * @param[in]  ci            CPU information
     * @param[in]  params        M, N, K sizes.
     */
    void configure(const ITensor *b, ITensor *transformed_b, bool transpose_b, const CPUInfo &ci, const INEGEMMWrapperKernel::Params &params);

    // Inherited methods overridden:
    void transform(const PrepareBWorkload &wl, const ThreadInfo &info) override;
    void create_workloads(std::vector<PrepareBWorkload> &workloads) override;
    void run(const Window &window, const ThreadInfo &info) override;
    BlockSizes block_sizes() const override;

private:
    const ITensor *_b
    {
        nullptr
    };
    ITensor     *_transformed_b{ nullptr };
    unsigned int _Nsize{ 0 };
    unsigned int _Ksize{ 0 };
    bool         _transpose_b{ false };
    BlockSizes   _block_sizes{};
};

} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMINTERLEAVEDPREPAREBWRAPPERKERNEL_H__ */
