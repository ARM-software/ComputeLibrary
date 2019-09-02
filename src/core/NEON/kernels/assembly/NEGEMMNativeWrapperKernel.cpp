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

#include "arm_compute/core/NEON/kernels/assembly/NEGEMMNativeWrapperKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/WindowIterator.h"

#include "../arm_gemm/utils.hpp"
#include "arm_gemm.hpp"

#include "../arm_gemm/mergeresults.hpp"
#include "../arm_gemm/transform.hpp"

#include "../arm_gemm/kernels/a64_sgemm_native_16x4.hpp"

namespace arm_compute
{
namespace
{
template <typename To, typename Tr>
struct Kernel
{
};

#ifdef __aarch64__
template <>
struct Kernel<float, float>
{
    using strategy = arm_gemm::sgemm_native_16x4;
};
#endif /* __aarch64__ */

} // namespace

template <typename To, typename Tr>
Window NEGEMMNativeWrapperKernel<To, Tr>::configure_internal(float alpha, float beta)
{
    ARM_COMPUTE_UNUSED(alpha);
    using strategy = typename Kernel<To, Tr>::strategy;

    _beta = beta;

    //Note: The window is shifted down by 1 dimension compare to the tensors
    Window window;
    window.set(Window::DimX, Window::Dimension(0, ceil_to_multiple(_params.M, strategy::out_height()), strategy::out_height()));
    window.set(Window::DimY, Window::Dimension(0, _params.batches));
    window.set(Window::DimZ, Window::Dimension(0, _params.multis));

    return window;
}

template <typename To, typename Tr>
void NEGEMMNativeWrapperKernel<To, Tr>::run_internal(const Window &window, const Coordinates &start_offset, const Coordinates &end_offset, const ThreadInfo &info)
{
    using strategy = typename Kernel<To, Tr>::strategy;

    TensorAccessor<To> a(*_a);
    TensorAccessor<To> b(*_b);
    TensorAccessor<Tr> c(*_c);

    // Handle 3d input re-interpretation
    if(_gemm_info.reinterpret_input_as_3d())
    {
        Strides a_strides_as_3d = _a->info()->strides_in_bytes();
        a_strides_as_3d.remove(Window::DimZ);
        a.set_strides(a_strides_as_3d);
    }

    // Handle 3d output re-interpretation
    if(_gemm_info.depth_output_gemm3d() != 0)
    {
        Strides c_strides_as_3d = _c->info()->strides_in_bytes();
        c_strides_as_3d.remove(Window::DimZ);
        c.set_strides(c_strides_as_3d);
    }

    unsigned int m_end = 0;

    strategy strat(info.cpu_info);
    auto window_iterator = arm_compute::create_window_iterator(window, start_offset, end_offset, [&](const Coordinates & id)
    {
        const unsigned int y0    = id.x();
        const unsigned int batch = id.y();
        const unsigned int multi = id.z();
        const unsigned int ymax  = std::min(y0 + strategy::out_height(), m_end);

        // TODO(COMPMID-1424) : Agree on gemm IO layouts
        strat.kernel(a(0, y0, batch, multi), a.stride(Window::DimY),
                     b(0, 0, multi), b.stride(Window::DimY),
                     c(0, y0, batch, multi), c.stride(Window::DimY),
                     _beta, (ymax - y0), _params.N, _params.K);
    });

    auto on_new_row_size = [&](unsigned int start, unsigned int end)
    {
        ARM_COMPUTE_UNUSED(start);
        m_end = std::min(end, _params.M);
    };

    window_iterator.iterate_3D(on_new_row_size);
}

#ifdef __aarch64__
template class NEGEMMNativeWrapperKernel<float, float>;
#endif /* __aarch64__ */

} // namespace arm_compute
