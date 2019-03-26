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
#include "arm_compute/core/NEON/kernels/NEFFTDigitReverseKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *idx, const FFTDigitReverseKernelInfo &config)
{
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() != DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_channels() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(idx, 1, DataType::U32);
    ARM_COMPUTE_RETURN_ERROR_ON(std::set<unsigned int>({ 0, 1 }).count(config.axis) == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[config.axis] != idx->tensor_shape().x());

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_channels() != 2);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *idx, const FFTDigitReverseKernelInfo &config)
{
    ARM_COMPUTE_UNUSED(idx, config);

    auto_init_if_empty(*output, input->clone()->set_num_channels(2));

    Window win = calculate_max_window(*input, Steps());
    input->set_valid_region(ValidRegion(Coordinates(), input->tensor_shape()));

    return std::make_pair(Status{}, win);
}
} // namespace

NEFFTDigitReverseKernel::NEFFTDigitReverseKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _idx(nullptr)
{
}

void NEFFTDigitReverseKernel::configure(const ITensor *input, ITensor *output, const ITensor *idx, const FFTDigitReverseKernelInfo &config)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, idx);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), idx->info(), config));

    _input  = input;
    _output = output;
    _idx    = idx;

    const size_t axis             = config.axis;
    const bool   is_conj          = config.conjugate;
    const bool   is_input_complex = (input->info()->num_channels() == 2);

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), idx->info(), config);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);

    if(axis == 0)
    {
        if(is_input_complex)
        {
            if(is_conj)
            {
                _func = &NEFFTDigitReverseKernel::digit_reverse_kernel_axis_0<true, true>;
            }
            else
            {
                _func = &NEFFTDigitReverseKernel::digit_reverse_kernel_axis_0<true, false>;
            }
        }
        else
        {
            _func = &NEFFTDigitReverseKernel::digit_reverse_kernel_axis_0<false, false>;
        }
    }
    else if(axis == 1)
    {
        if(is_input_complex)
        {
            if(is_conj)
            {
                _func = &NEFFTDigitReverseKernel::digit_reverse_kernel_axis_1<true, true>;
            }
            else
            {
                _func = &NEFFTDigitReverseKernel::digit_reverse_kernel_axis_1<true, false>;
            }
        }
        else
        {
            _func = &NEFFTDigitReverseKernel::digit_reverse_kernel_axis_1<false, false>;
        }
    }
    else
    {
        ARM_COMPUTE_ERROR("Not supported");
    }
}

Status NEFFTDigitReverseKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *idx, const FFTDigitReverseKernelInfo &config)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, idx, config));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), idx->clone().get(), config).first);
    return Status{};
}

template <bool is_input_complex, bool is_conj>
void NEFFTDigitReverseKernel::digit_reverse_kernel_axis_0(const Window &window)
{
    const size_t N = _input->info()->dimension(0);

    // Copy the look-up buffer to a local array
    std::vector<unsigned int> buffer_idx(N);
    std::copy_n(reinterpret_cast<unsigned int *>(_idx->buffer()), N, buffer_idx.data());

    // Input/output iterators
    Window slice = window;
    slice.set(0, Window::DimX);
    Iterator in(_input, slice);
    Iterator out(_output, slice);

    // Row buffers
    std::vector<float> buffer_row_out(2 * N);
    std::vector<float> buffer_row_in(2 * N);

    execute_window_loop(slice, [&](const Coordinates &)
    {
        if(is_input_complex)
        {
            // Load
            memcpy(buffer_row_in.data(), reinterpret_cast<float *>(in.ptr()), 2 * N * sizeof(float));

            // Shuffle
            for(size_t x = 0; x < 2 * N; x += 2)
            {
                size_t idx            = buffer_idx[x / 2];
                buffer_row_out[x]     = buffer_row_in[2 * idx];
                buffer_row_out[x + 1] = (is_conj ? -buffer_row_in[2 * idx + 1] : buffer_row_in[2 * idx + 1]);
            }
        }
        else
        {
            // Load
            memcpy(buffer_row_in.data(), reinterpret_cast<float *>(in.ptr()), N * sizeof(float));

            // Shuffle
            for(size_t x = 0; x < N; ++x)
            {
                size_t idx            = buffer_idx[x];
                buffer_row_out[2 * x] = buffer_row_in[idx];
            }
        }

        // Copy back
        memcpy(reinterpret_cast<float *>(out.ptr()), buffer_row_out.data(), 2 * N * sizeof(float));
    },
    in, out);
}

template <bool is_input_complex, bool is_conj>
void NEFFTDigitReverseKernel::digit_reverse_kernel_axis_1(const Window &window)
{
    const size_t Nx = _input->info()->dimension(0);
    const size_t Ny = _input->info()->dimension(1);

    // Copy the look-up buffer to a local array
    std::vector<unsigned int> buffer_idx(Ny);
    std::copy_n(reinterpret_cast<unsigned int *>(_idx->buffer()), Ny, buffer_idx.data());

    // Output iterator
    Window slice = window;
    slice.set(0, Window::DimX);
    Iterator out(_output, slice);

    // Row buffer
    std::vector<float> buffer_row(Nx);

    // Strides
    const size_t stride_z = _input->info()->strides_in_bytes()[2];
    const size_t stride_w = _input->info()->strides_in_bytes()[3];

    execute_window_loop(slice, [&](const Coordinates & id)
    {
        auto        *out_ptr    = reinterpret_cast<float *>(out.ptr());
        auto        *in_ptr     = reinterpret_cast<float *>(_input->buffer() + id.z() * stride_z + id[3] * stride_w);
        const size_t y_shuffled = buffer_idx[id.y()];

        if(is_input_complex)
        {
            // Shuffle the entire row into the output
            memcpy(out_ptr, in_ptr + 2 * Nx * y_shuffled, 2 * Nx * sizeof(float));

            // Conjugate if necessary
            if(is_conj)
            {
                for(size_t x = 0; x < 2 * Nx; x += 2)
                {
                    out_ptr[x + 1] = -out_ptr[x + 1];
                }
            }
        }
        else
        {
            // Shuffle the entire row into the buffer
            memcpy(buffer_row.data(), in_ptr + Nx * y_shuffled, Nx * sizeof(float));

            // Copy the buffer to the output, with a zero imaginary part
            for(size_t x = 0; x < 2 * Nx; x += 2)
            {
                out_ptr[x] = buffer_row[x / 2];
            }
        }
    },
    out);
}

void NEFFTDigitReverseKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_UNUSED(info);
    (this->*_func)(window);
}

} // namespace arm_compute
