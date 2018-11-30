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

#include "arm_compute/core/NEON/kernels/assembly/NEGEMMInterleavedMatrixMultiplyWrapper.h"

#include "NEGEMMInterleavedStrategies.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/WindowIterator.h"

namespace arm_compute
{
template <typename To, typename Tr, bool use_dot>
void NEGEMMInterleavedMatrixMultiplyWrapperTemplate<To, Tr, use_dot>::configure(const ITensor *prepared_a, const ITensor *transformed_b, ITensor *tmp_c, ITensor *c, const Window &block_walker,
                                                                                const BlockSizes &block_sizes, const INEGEMMWrapperKernel::Params &params, bool b_is_pretransposed, float alpha, float beta, unsigned int max_num_threads)
{
    using strategy = typename Kernel<To, use_dot>::strategy;

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

template <typename To, typename Tr, bool use_dot>
void NEGEMMInterleavedMatrixMultiplyWrapperTemplate<To, Tr, use_dot>::transform(const MatrixMultiplyWorkload &wl, const ThreadInfo &info, const Window &batch_window, const Coordinates &start_offset,
                                                                                const Coordinates &end_offset)
{
    using strategy = typename Kernel<To, use_dot>::strategy;

    strategy           strat(info.cpu_info);
    TensorAccessor<To> prepared_a(*_prepared_a);
    TensorAccessor<To> transformed_b(*_transformed_b);
    TensorAccessor<Tr> c(*_c);
    TensorAccessor<Tr> tmp_c(*_tmp_c);

    int  prev_batch      = -1;
    To *a_ptr           = nullptr;
    auto window_iterator = arm_compute::create_window_iterator(batch_window, start_offset, end_offset, [&](const Coordinates & id)
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
        strat.transforms.Merge(c(0, 0, batch, wl._multi), tmp_c(0, info.thread_id), c.stride(1), y, ymax, wl._x0, wl._xmax, _alpha, (wl._k0 == 0 ? _beta : static_cast<Tr>(1)));
    });
    auto on_new_row_size = [&](unsigned int start, unsigned int end)
    {
        //Nothing to do
    };
    window_iterator.iterate_2D(on_new_row_size);
}

template <typename To, typename Tr, bool use_dot>
void NEGEMMInterleavedMatrixMultiplyWrapperTemplate<To, Tr, use_dot>::create_workloads(std::vector<MatrixMultiplyWorkload> &workloads)
{
    using strategy = typename Kernel<To, use_dot>::strategy;

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

//TODO: regroup somewhere ?
template class NEGEMMInterleavedMatrixMultiplyWrapperTemplate<float, float>;
#ifdef __aarch64__
template class NEGEMMInterleavedMatrixMultiplyWrapperTemplate<uint8_t, uint32_t>;
template class NEGEMMInterleavedMatrixMultiplyWrapperTemplate<int8_t, int32_t>;
template class NEGEMMInterleavedMatrixMultiplyWrapperTemplate<uint8_t, uint32_t, true>;
template class NEGEMMInterleavedMatrixMultiplyWrapperTemplate<int8_t, int32_t, true>;
#endif /* __aarch64__ */

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template class NEGEMMInterleavedMatrixMultiplyWrapperTemplate<float16_t, float16_t>;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
} // namespace arm_compute
