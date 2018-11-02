/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEWinogradLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "support/ToolchainSupport.h"

#include "src/core/NEON/kernels/winograd/winograd_shim_nchw.hpp"

using T = winograd_shim_nchw::Winograd2x2_3x3GEMM<float, float>;

namespace arm_compute
{
class Winograd3x3F32::Private
{
public:
    Private(const KernelShape &kernel_shape, const Tensor4DShape input_shape, const PaddingType padding_type, void *kernel_storage)
        : convolver(kernel_shape, input_shape, padding_type, kernel_storage)
    {
    }

    T convolver;
};

Winograd3x3F32::~Winograd3x3F32()
{
}

void Winograd3x3F32::nchw2nhwc(const Tensor4DShape &input_shape, const PaddingType padding_type, void *working_space, const void *const input)
{
    _pimpl->convolver.nchw2nhwc(input_shape, padding_type, working_space, reinterpret_cast<const float *>(input));
}

void Winograd3x3F32::nhwc2nchw(const Tensor4DShape &input_shape, const PaddingType padding_type, void *working_space, void *const output)
{
    _pimpl->convolver.nhwc2nchw(input_shape, padding_type, working_space, reinterpret_cast<float *const>(output));
}

void Winograd3x3F32::transform_weights(const void *const kernel, void *transform_working_space)
{
    _pimpl->convolver.transform_weights(reinterpret_cast<const float *>(kernel), transform_working_space);
}

void Winograd3x3F32::reshape_input(const Tensor4DShape &input_shape, const PaddingType padding_type, const void *const input, void *working_space)
{
    _pimpl->convolver.reshape_input(input_shape, padding_type, reinterpret_cast<const float *>(input), working_space);
}

void Winograd3x3F32::reshape_output(const Tensor4DShape &input_shape, const PaddingType padding_type, void *const output)
{
#if defined(__aarch64__)
    _pimpl->convolver.reshape_output(input_shape, padding_type, reinterpret_cast<float *const>(output));
#else  /* __aarch64__ */
    ARM_COMPUTE_UNUSED(input_shape);
    ARM_COMPUTE_UNUSED(padding_type);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_ERROR("Not implemented");
#endif /* __aarch64__ */
}

std::pair<void *, void *> Winograd3x3F32::get_nhwc_ptrs(const Tensor4DShape &input_shape, const PaddingType padding_type, void *working_space)
{
    return _pimpl->convolver.get_nhwc_ptrs(input_shape, padding_type, working_space);
}

Winograd3x3F32::Winograd3x3F32(const KernelShape &kernel_shape, const Tensor4DShape input_shape, const PaddingType padding_type, void *kernel_storage)
    : _pimpl(support::cpp14::make_unique<Private>(kernel_shape, input_shape, padding_type, kernel_storage))
{
}

size_t NEWinogradLayerKernel::get_kernel_storage_size(const KernelShape &shape)
{
    return T::get_kernel_storage_size(shape);
}

size_t NEWinogradLayerKernel::get_working_space_size(const Tensor4DShape &input_shape, const KernelShape &k_shape, const PaddingType padding)
{
    return T::get_working_space_size(input_shape, k_shape, padding);
}

size_t NEWinogradLayerKernel::get_kernel_transform_working_size(const KernelShape &shape)
{
    return T::get_kernel_transform_working_size(shape);
}

NEWinogradLayerKernel::NEWinogradLayerKernel()
    : _convolver(nullptr), _output(nullptr)
{
}

void NEWinogradLayerKernel::configure(ITensor *output, Winograd3x3F32 *convolver)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    _convolver = convolver;
    Window win = calculate_max_window(*output->info());
    INEKernel::configure(win);
}

void NEWinogradLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(info.num_threads < 1);
    const size_t tid                  = info.thread_id;
    const size_t num_threads          = std::min(info.num_threads, 16);
    const size_t num_gemms_per_thread = 16 / num_threads;
    const size_t first_gemm           = tid * num_gemms_per_thread;
    const size_t last_gemm            = (tid == (num_threads - 1)) ? 15 : first_gemm + num_gemms_per_thread - 1;
    _convolver->_pimpl->convolver.execute(first_gemm, last_gemm);
}
} // namespace arm_compute
