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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVEDTRANSFORMAWRAPPER_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVEDTRANSFORMAWRAPPER_H__

#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
class ITensor;

/** Unit of work for @ref NEGEMMInterleavedTransformAWrapper to process */
struct TransformAWorkload
{
    /** Constructor
     *
     * @param[in] k0    First value to process along the K dimension.
     * @param[in] kmax  Last value to process along the K dimension.
     * @param[in] multi Multi index.
     */
    TransformAWorkload(unsigned int k0, unsigned int kmax, unsigned int multi)
        : _k0(k0), _kmax(kmax), _multi(multi)
    {
    }
    unsigned int _k0;    /**< First value to process along the K dimension. */
    unsigned int _kmax;  /**< Last value to process along the K dimension. */
    unsigned int _multi; /**< Multi index. */
};

/** Equivalent to arm_gemm::GemmInterleaved's Transform<strategy::A_interleave, strategy::A_block but using Compute Library types.
 *
 * Note: Each workload converts a different slice of a and writes it to transformed_a (Which can store only one slice at the time), therefore the workloads' execution should be interleaved with other workloads that make use of their result.
 */
class NEGEMMInterleavedTransformAWrapper
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
    virtual void transform(const TransformAWorkload &wl, const ThreadInfo &info, const Window &batch_window, const Coordinates &start_offset, const Coordinates &end_offset) = 0;
    /** Generate an array of workloads
     *
     * @param[out] workloads Container to store the generated workloads.
     */
    virtual void create_workloads(std::vector<TransformAWorkload> &workloads) = 0;
    /** Default destructor */
    virtual ~NEGEMMInterleavedTransformAWrapper() = default;
};

/** Type specialisations of @ref NEGEMMInterleavedTransformAWrapper */
template <typename To, bool use_dot = false>
class NEGEMMInterleavedTransformAWrapperTemplate : public NEGEMMInterleavedTransformAWrapper
{
public:
    /** Configure the reshape A routine.
     *
     * @param[in]  a             Input matrix A.
     * @param[out] transformed_a Reshaped matrix A.
     * @param[in]  transpose_a   Also transpose A ?
     * @param[in]  block_walker  Window representing the layout of the matrix's blocks
     * @param[in]  params        M, N, K sizes.
     */
    void configure(const ITensor *a, ITensor *transformed_a, bool transpose_a, const Window &block_walker, const INEGEMMWrapperKernel::Params &params);

    // Inherited methods overridden:
    void transform(const TransformAWorkload &wl, const ThreadInfo &info, const Window &batch_window, const Coordinates &start_offset, const Coordinates &end_offset) override;
    void create_workloads(std::vector<TransformAWorkload> &workloads) override;

private:
    const ITensor *_a
    {
        nullptr
    };
    ITensor     *_transformed_a{ nullptr };
    unsigned int _Msize{ 0 };
    unsigned int _Ksize{ 0 };
    bool         _transpose_a{ false };
    Window       _k_multi_window{};
};

} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMINTERLEAVEDTRANSFORMAWRAPPER_H__ */
