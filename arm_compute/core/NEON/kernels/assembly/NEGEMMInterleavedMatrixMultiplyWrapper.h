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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVEDMATRIXMULTIPLYWRAPPER_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVEDMATRIXMULTIPLYWRAPPER_H__

#include "arm_compute/core/NEON/kernels/assembly/Helpers.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/WindowIterator.h"

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
template <typename strategy>
class NEGEMMInterleavedMatrixMultiplyWrapperTemplate : public NEGEMMInterleavedMatrixMultiplyWrapper
{
public:
    /** Configure the matrix multiplication: C = alpha * A * B + beta * C
     *
     * @param[in]     prepared_a       Already reshaped matrix A.
     * @param[in]     transformed_b    Already reshaped matrix B.
     * @param[out]    tmp_c            Temporary buffer to be used to store intermediate results.
     * @param[in,out] c                Result matrix C.
     * @param[in]     block_walker     Window containing iteration information for the M and batch dimensions.
     * @param[in]     block_sizes      Block sizes to use for the matrix multiplication (A & B must have been reshaped using these same block sizes).
     * @param[in]     params           M, N, K sizes.
     * @param[in]     is_pretransposed Is B also pretransposed ?
     * @param[in]     alpha            Alpha value
     * @param[in]     beta             Beta value
     * @param[in]     max_num_threads  Maximum number of threads that might be used for the calculations.
     */
    void configure(const ITensor *prepared_a, const ITensor *transformed_b, ITensor *tmp_c, ITensor *c, const Window &block_walker, const BlockSizes &block_sizes,
                   const INEGEMMWrapperKernel::Params &params, bool b_is_pretransposed, float alpha, float beta, unsigned int max_num_threads)
    {
        _prepared_a         = prepared_a;
        _transformed_b      = transformed_b;
        _tmp_c              = tmp_c;
        _c                  = c;
        _block_walker       = block_walker;
        _block_sizes        = block_sizes;
        _params             = params;
        _b_is_pretransposed = b_is_pretransposed;
        _alpha              = alpha;
        _beta               = beta;

        auto_init_if_empty(*_tmp_c->info(), c->info()->clone()->set_tensor_shape(TensorShape{ _block_sizes.x_block * strategy::out_height(), max_num_threads }));
    }

    // Inherited methods overridden:
    void transform(const MatrixMultiplyWorkload &wl, const ThreadInfo &info, const Window &batch_window, const Coordinates &start_offset, const Coordinates &end_offset) override
    {
        strategy                                        strat(info.cpu_info);
        TensorAccessor<typename strategy::operand_type> prepared_a(*_prepared_a);
        TensorAccessor<typename strategy::operand_type> transformed_b(*_transformed_b);
        TensorAccessor<typename strategy::result_type>  c(*_c);
        TensorAccessor<typename strategy::result_type>  tmp_c(*_tmp_c);

        int                              prev_batch = -1;
        typename strategy::operand_type *a_ptr      = nullptr;
        auto window_iterator                        = arm_compute::create_window_iterator(batch_window, start_offset, end_offset, [&](const Coordinates & id)
        {
            const unsigned int y     = id.x();
            const unsigned int batch = id.y();
            const unsigned int ymax  = std::min(_params.M, y + strategy::out_height());

            // If it's the first block of a new batch then reset the pointer to A.
            if(prev_batch != static_cast<int>(batch))
            {
                const unsigned int first_m = id.x();
                a_ptr                      = prepared_a(0, first_m, batch);
                prev_batch                 = batch;
            }

            // Call matrix multiply assembly routine to process the block:
            strat.kernel(a_ptr, transformed_b(wl._offset_transformed_b), tmp_c(0, info.thread_id), 1, wl._bblocks, wl._kern_k);
            a_ptr += strategy::out_height() * wl._kern_k;

            // Merge the result with the other blocks' results:
            strat.transforms.Merge(c(0, 0, batch, wl._multi), tmp_c(0, info.thread_id), c.stride(1), y, ymax, wl._x0, wl._xmax, _alpha, (wl._k0 == 0 ? _beta : static_cast<typename strategy::result_type>(1)));
        });
        auto on_new_row_size = [&](unsigned int start, unsigned int end)
        {
            //Nothing to do
        };
        window_iterator.iterate_2D(on_new_row_size);
    }
    void create_workloads(std::vector<MatrixMultiplyWorkload> &workloads) override
    {
        unsigned int offset_transformed_b = 0;
        unsigned int wl_index             = 0;
        unsigned int num_buffers = 0, reshaped_block_size = 0;

        if(!_b_is_pretransposed)
        {
            num_buffers         = _transformed_b->info()->tensor_shape()[1];
            reshaped_block_size = _transformed_b->info()->tensor_shape()[0];
        }
        execute_window_loop(_block_walker, [&](const Coordinates & id)
        {
            const unsigned int x0    = id.x();
            const unsigned int k0    = id.y();
            const unsigned int multi = id.z();

            const unsigned int xmax = std::min(x0 + _block_walker.x().step(), _params.N);
            const unsigned int kmax = std::min(k0 + _block_walker.y().step(), _params.K);

            // Figure out how many "K" the kernel will actually process.
            const int kern_k  = ceil_to_multiple(kmax - k0, strategy::k_unroll());
            const int bblocks = DIV_CEIL(xmax - x0, strategy::out_width());

            workloads.push_back(MatrixMultiplyWorkload(offset_transformed_b, x0, xmax, k0, kmax, multi, kern_k, bblocks));

            if(_b_is_pretransposed)
            {
                offset_transformed_b += bblocks * strategy::out_width() * kern_k;
            }
            else
            {
                // Rotate through the BufferManager's buffers:
                wl_index++;
                offset_transformed_b = (wl_index % num_buffers) * reshaped_block_size;
            }
        });
    }

private:
    const ITensor *_prepared_a
    {
        nullptr
    };
    const ITensor                 *_transformed_b{ nullptr };
    ITensor                       *_tmp_c{ nullptr };
    ITensor                       *_c{ nullptr };
    unsigned int                   _Nsize{ 0 };
    unsigned int                   _Ksize{ 0 };
    bool                           _transpose_b{ false };
    BlockSizes                     _block_sizes{};
    INEGEMMWrapperKernel::Params   _params{};
    Window                         _block_walker{};
    bool                           _b_is_pretransposed{ false };
    typename strategy::result_type _alpha{};
    typename strategy::result_type _beta{};
};

} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMINTERLEAVEDMATRIXMULTIPLYWRAPPER_H__ */
