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
#include "arm_compute/runtime/CL/functions/CLFFT1D.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/helpers/fft.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
CLFFT1D::CLFFT1D(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _digit_reverse_kernel(), _fft_kernels(), _scale_kernel(), _digit_reversed_input(), _digit_reverse_indices(), _num_ffts(0), _run_scale(false)
{
}

void CLFFT1D::configure(const ICLTensor *input, ICLTensor *output, const FFT1DInfo &config)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLFFT1D::validate(input->info(), output->info(), config));

    // Decompose size to radix factors
    const auto         supported_radix   = CLFFTRadixStageKernel::supported_radix();
    const unsigned int N                 = input->info()->tensor_shape()[config.axis];
    const auto         decomposed_vector = arm_compute::helpers::fft::decompose_stages(N, supported_radix);
    ARM_COMPUTE_ERROR_ON(decomposed_vector.empty());

    // Flags
    _run_scale        = config.direction == FFTDirection::Inverse;
    const bool is_c2r = input->info()->num_channels() == 2 && output->info()->num_channels() == 1;

    // Configure digit reverse
    FFTDigitReverseKernelInfo digit_reverse_config;
    digit_reverse_config.axis      = config.axis;
    digit_reverse_config.conjugate = config.direction == FFTDirection::Inverse;
    TensorInfo digit_reverse_indices_info(TensorShape(input->info()->tensor_shape()[config.axis]), 1, DataType::U32);
    _digit_reverse_indices.allocator()->init(digit_reverse_indices_info);
    _memory_group.manage(&_digit_reversed_input);
    _digit_reverse_kernel.configure(input, &_digit_reversed_input, &_digit_reverse_indices, digit_reverse_config);

    // Create and configure FFT kernels
    unsigned int Nx = 1;
    _num_ffts       = decomposed_vector.size();
    _fft_kernels.resize(_num_ffts);
    for(unsigned int i = 0; i < _num_ffts; ++i)
    {
        const unsigned int radix_for_stage = decomposed_vector.at(i);

        FFTRadixStageKernelInfo fft_kernel_info;
        fft_kernel_info.axis           = config.axis;
        fft_kernel_info.radix          = radix_for_stage;
        fft_kernel_info.Nx             = Nx;
        fft_kernel_info.is_first_stage = (i == 0);
        _fft_kernels[i].configure(&_digit_reversed_input, ((i == (_num_ffts - 1)) && !is_c2r) ? output : nullptr, fft_kernel_info);

        Nx *= radix_for_stage;
    }

    // Configure scale kernel
    if(_run_scale)
    {
        FFTScaleKernelInfo scale_config;
        scale_config.scale     = static_cast<float>(N);
        scale_config.conjugate = config.direction == FFTDirection::Inverse;
        is_c2r ? _scale_kernel.configure(&_digit_reversed_input, output, scale_config) : _scale_kernel.configure(output, nullptr, scale_config);
    }

    // Allocate tensors
    _digit_reversed_input.allocator()->allocate();
    _digit_reverse_indices.allocator()->allocate();

    // Init digit reverse indices
    const auto digit_reverse_cpu = arm_compute::helpers::fft::digit_reverse_indices(N, decomposed_vector);
    _digit_reverse_indices.map(CLScheduler::get().queue(), true);
    std::copy_n(digit_reverse_cpu.data(), N, reinterpret_cast<unsigned int *>(_digit_reverse_indices.buffer()));
    _digit_reverse_indices.unmap(CLScheduler::get().queue());
}

Status CLFFT1D::validate(const ITensorInfo *input, const ITensorInfo *output, const FFT1DInfo &config)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() != DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_channels() != 1 && input->num_channels() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(std::set<unsigned int>({ 0, 1 }).count(config.axis) == 0);

    // Check if FFT is decomposable
    const auto         supported_radix   = CLFFTRadixStageKernel::supported_radix();
    const unsigned int N                 = input->tensor_shape()[config.axis];
    const auto         decomposed_vector = arm_compute::helpers::fft::decompose_stages(N, supported_radix);
    ARM_COMPUTE_RETURN_ERROR_ON(decomposed_vector.empty());

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_channels() == 1 && input->num_channels() == 1);
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_channels() != 1 && output->num_channels() != 2);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

void CLFFT1D::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    // Run digit reverse
    CLScheduler::get().enqueue(_digit_reverse_kernel, false);

    // Run radix kernels
    for(unsigned int i = 0; i < _num_ffts; ++i)
    {
        CLScheduler::get().enqueue(_fft_kernels[i], i == (_num_ffts - 1) && !_run_scale);
    }

    // Run output scaling
    if(_run_scale)
    {
        CLScheduler::get().enqueue(_scale_kernel, true);
    }
}
} // namespace arm_compute
