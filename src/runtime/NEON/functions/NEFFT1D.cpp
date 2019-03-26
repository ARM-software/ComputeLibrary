/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEFFT1D.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/helpers/fft.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
NEFFT1D::NEFFT1D(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _digit_reversed_input(), _digit_reverse_indices(), _digit_reverse_kernel(), _fft_kernels(), _n_ffts(0)
{
}

void NEFFT1D::configure(const ITensor *input, ITensor *output, const FFT1DInfo &config)
{
    // Decompose size to radix factors
    const auto         supported_radix   = NEFFTRadixStageKernel::supported_radix();
    const unsigned int N                 = input->info()->tensor_shape()[config.axis];
    const auto         decomposed_vector = arm_compute::helpers::fft::decompose_stages(N, supported_radix);
    ARM_COMPUTE_ERROR_ON(decomposed_vector.empty());

    // Configure digit reverse
    TensorInfo digit_reverse_indices_info(TensorShape(input->info()->tensor_shape()[config.axis]), 1, DataType::U32);
    _digit_reverse_indices.allocator()->init(digit_reverse_indices_info);
    _memory_group.manage(&_digit_reversed_input);
    _digit_reverse_kernel.configure(input, &_digit_reversed_input, &_digit_reverse_indices, config.axis);

    // Create and configure FFT kernels
    unsigned int Nx = 1;
    _n_ffts         = decomposed_vector.size();
    _fft_kernels.resize(_n_ffts);
    for(unsigned int i = 0; i < _n_ffts; ++i)
    {
        const unsigned int radix_for_stage = decomposed_vector.at(i);

        FFTRadixStageKernelInfo fft_kernel_desc;
        fft_kernel_desc.axis           = config.axis;
        fft_kernel_desc.radix          = radix_for_stage;
        fft_kernel_desc.Nx             = Nx;
        fft_kernel_desc.is_first_stage = (i == 0);
        _fft_kernels[i].configure(&_digit_reversed_input, i == (_n_ffts - 1) ? output : nullptr, fft_kernel_desc);

        Nx *= radix_for_stage;
    }

    // Allocate tensors
    _digit_reversed_input.allocator()->allocate();
    _digit_reverse_indices.allocator()->allocate();

    // Init digit reverse indices
    const auto digit_reverse_cpu = arm_compute::helpers::fft::digit_reverse_indices(N, decomposed_vector);
    std::copy_n(digit_reverse_cpu.data(), N, reinterpret_cast<unsigned int *>(_digit_reverse_indices.buffer()));
}

Status NEFFT1D::validate(const ITensorInfo *input, const ITensorInfo *output, const FFT1DInfo &config)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 2, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(config.axis != 0);

    // Check if FFT is decomposable
    const auto         supported_radix   = NEFFTRadixStageKernel::supported_radix();
    const unsigned int N                 = input->tensor_shape()[config.axis];
    const auto         decomposed_vector = arm_compute::helpers::fft::decompose_stages(N, supported_radix);
    ARM_COMPUTE_RETURN_ERROR_ON(decomposed_vector.empty());

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

void NEFFT1D::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    NEScheduler::get().schedule(&_digit_reverse_kernel, Window::DimY);

    for(unsigned int i = 0; i < _n_ffts; ++i)
    {
        NEScheduler::get().schedule(&_fft_kernels[i], Window::DimY);
    }
}
} // namespace arm_compute
