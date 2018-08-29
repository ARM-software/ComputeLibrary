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

#include "arm_compute/core/NEON/kernels/assembly/NEGEMMInterleavedTransformAWrapper.h"

#include "NEGEMMInterleavedStrategies.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/WindowIterator.h"

#include "utils/TypePrinter.h"

namespace arm_compute
{
template <typename To, bool use_dot>
void NEGEMMInterleavedTransformAWrapperTemplate<To, use_dot>::configure(const ITensor *a, ITensor *transformed_a, bool transpose_a, const Window &block_walker,
                                                                        const INEGEMMWrapperKernel::Params &params)
{
    _a              = a;
    _transformed_a  = transformed_a;
    _transpose_a    = transpose_a;
    _Ksize          = params.K;
    _Msize          = params.M;
    _k_multi_window = block_walker.shift_dimensions(1); // block_walker contains (M,K,Multi) --> shift by 1 to get rid of the "M" dimension
}

template <typename To, bool use_dot>
void NEGEMMInterleavedTransformAWrapperTemplate<To, use_dot>::transform(const TransformAWorkload &wl, const ThreadInfo &info, const Window &batch_window, const Coordinates &start_offset,
                                                                        const Coordinates &end_offset)
{
    using strategy = typename Kernel<To, use_dot>::strategy;

    strategy           strat(info.cpu_info);
    TensorAccessor<To> a(*_a);
    TensorAccessor<To> transformed_a(*_transformed_a);

    if(_a->info()->data_layout() == DataLayout::NHWC)
    {
        // In the case of NHWC we want to interpret the output shape as 3D. Thus, the batch stride for A is
        // the relevant multiple of the row stride.
        const size_t nhwc_batch_stride = _a->info()->strides_in_bytes().y() * _Msize;
        a.set_stride(2, nhwc_batch_stride);
    }

    unsigned int last_m = 0;
    int  last_y          = -1;
    auto window_iterator = arm_compute::create_window_iterator(batch_window, start_offset, end_offset, [&](const Coordinates & id)
    {
        if(id.y() != last_y)
        {
            last_y               = id.y();
            unsigned int batch   = id.y();
            unsigned int first_m = id.x();

            if(first_m >= last_m)
                return;

            strat.transforms.PrepareA(transformed_a(0, first_m, batch),
                                      a(0, 0, batch, wl._multi),
                                      a.stride(1), first_m, last_m, wl._k0, wl._kmax, _transpose_a);
        }
    });
    auto on_new_row_size = [&](unsigned int start, unsigned int end)
    {
        last_m = std::min(end, _Msize);
    };
    window_iterator.iterate_2D(on_new_row_size);
}

template <typename To, bool use_dot>
void NEGEMMInterleavedTransformAWrapperTemplate<To, use_dot>::create_workloads(std::vector<TransformAWorkload> &workloads)
{
    execute_window_loop(_k_multi_window, [&](const Coordinates & id)
    {
        const unsigned int k0    = id.x();
        const unsigned int multi = id.y();
        const unsigned int kmax  = std::min(k0 + _k_multi_window.x().step(), _Ksize);

        workloads.push_back(TransformAWorkload(k0, kmax, multi));
    });
}

template class NEGEMMInterleavedTransformAWrapperTemplate<float>;
#ifdef __aarch64__
template class NEGEMMInterleavedTransformAWrapperTemplate<uint8_t>;
template class NEGEMMInterleavedTransformAWrapperTemplate<int8_t>;
template class NEGEMMInterleavedTransformAWrapperTemplate<uint8_t, true>;
template class NEGEMMInterleavedTransformAWrapperTemplate<int8_t, true>;
#endif /* __aarch64__ */

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template class NEGEMMInterleavedTransformAWrapperTemplate<float16_t>;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
} // namespace arm_compute
