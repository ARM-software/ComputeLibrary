/*
 * Copyright (c) 2023-2024 Arm Limited.
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
#if defined(__aarch64__)

#include "src/core/NEON/kernels/NEReorderKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

#include "src/common/utils/Log.h"
#include "src/core/NEON/kernels/arm_gemm/transform.hpp"

namespace arm_compute
{

void NEReorderKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    switch (_input->info()->data_type())
    {
        case DataType::F32:
        {
            const int ksize_rows_elements = _xmax * _ksize;
            const int jump_rows           = ksize_rows_elements * window.x().start();
            const int k_start             = window.x().start() * _ksize;
            const int k_end               = std::min(window.x().end() * _ksize, _kmax);
            const int stride              = _kmax;
            if (k_start < k_end)
            {
                switch (_output_wf)
                {
                    case WeightFormat::OHWIo4:
                    {
                        switch (_output->info()->data_type())
                        {
                            case DataType::F32:
                                arm_gemm::Transform<4, 1, true, arm_gemm::VLType::None>(
                                    reinterpret_cast<float *>(_output->buffer()) + jump_rows,
                                    reinterpret_cast<float *>(_input->buffer()), stride, k_start, k_end, 0, _xmax);
                                break;
                            case DataType::BFLOAT16:
                                arm_gemm::Transform<4, 4, true, arm_gemm::VLType::None>(
                                    reinterpret_cast<bfloat16 *>(_output->buffer()) + jump_rows,
                                    reinterpret_cast<float *>(_input->buffer()), stride, k_start, k_end, 0, _xmax);
                                break;
                            default:
                                ARM_COMPUTE_ERROR("Unsupported data type!");
                        }
                        break;
                    }
#if defined(ARM_COMPUTE_ENABLE_SVE)
                    case WeightFormat::OHWIo8:
                    {
                        switch (_output->info()->data_type())
                        {
                            case DataType::F32:
                                arm_gemm::Transform<1, 1, true, arm_gemm::VLType::SVE>(
                                    reinterpret_cast<float *>(_output->buffer()) + jump_rows,
                                    reinterpret_cast<float *>(_input->buffer()), stride, k_start, k_end, 0, _xmax);
                                break;
                            case DataType::BFLOAT16:
                                arm_gemm::Transform<2, 4, true, arm_gemm::VLType::SVE>(
                                    reinterpret_cast<bfloat16 *>(_output->buffer()) + jump_rows,
                                    reinterpret_cast<float *>(_input->buffer()), stride, k_start, k_end, 0, _xmax);
                                break;
                            default:
                                ARM_COMPUTE_ERROR("Unsupported data type!");
                        }
                        break;
                    }
#endif /* ARM_COMPUTE_ENABLE_SVE */
                    default:
                    {
                        ARM_COMPUTE_ERROR("Unsupported data type!");
                        break;
                    }
                }
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported data type!");
    }
}

NEReorderKernel::NEReorderKernel()
    : _input(nullptr),
      _output(nullptr),
      _ksize(0),
      _kmax(0),
      _xmax(0),
      _input_wf(WeightFormat::ANY),
      _output_wf(WeightFormat::ANY)
{
}

void NEReorderKernel::configure(const ITensor            *input,
                                ITensor                  *output,
                                arm_compute::WeightFormat input_wf,
                                arm_compute::WeightFormat output_wf)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, input_wf, output_wf);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), input_wf, output_wf));

    // Set variables
    _input     = input;
    _output    = output;
    _input_wf  = input_wf;
    _output_wf = output_wf;

    // Setting parameters for transform
    auto dims = input->info()->num_dimensions();
    switch (dims)
    {
        case 2:
        {
            _xmax = input->info()->dimension(0); // Number of columns in input matrix
            _kmax = input->info()->dimension(1); // Number of rows in input matrix
            break;
        }
        case 4:
        {
            _xmax = input->info()->dimension(2); // Number of columns in input matrix
            _kmax = input->info()->dimension(3); // Number of rows in input matrix
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Only 2 or 4 dimensions supported.");
        }
    }

    // Configure kernel window
    // Window size is set by rows / _ksize
    Window win;
    int    window_size = 0;
    switch (_output_wf)
    {
#if defined(ARM_COMPUTE_ENABLE_SVE)
        case WeightFormat::OHWIo8:
        {
            _ksize      = 8;
            window_size = _kmax / _ksize;
            break;
        }
#endif /* ARM_COMPUTE_ENABLE_SVE */
        case WeightFormat::OHWIo4:
        {
            _ksize      = 4;
            window_size = _kmax / _ksize;
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported weight format.");
            break;
        }
    }
    if (_kmax % _ksize != 0)
    {
        window_size += 1;
    }

    win.set(Window::DimX, Window::Dimension(0, window_size, 1));

    INEKernel::configure(win);
}

Status NEReorderKernel::validate(const ITensorInfo        *input,
                                 const ITensorInfo        *output,
                                 arm_compute::WeightFormat input_wf,
                                 arm_compute::WeightFormat output_wf)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    if (output->tensor_shape().total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() != DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON(output->data_type() != DataType::F32 && output->data_type() != DataType::BFLOAT16);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
        // Only input WeightFormat OHWI supported
        ARM_COMPUTE_RETURN_ERROR_ON(input_wf != arm_compute::WeightFormat::OHWI);
        int  input_x_dim;
        int  input_k_dim;
        int  output_x_dim;
        int  output_k_dim;
        auto dims = output->num_dimensions();
        switch (dims)
        {
            case 2:
            {
                input_x_dim  = input->dimension(0);  // Number of columns in input matrix
                input_k_dim  = input->dimension(1);  // Number of rows in input matrix
                output_x_dim = output->dimension(0); // Number of columns in output matrix
                output_k_dim = output->dimension(1); // Number of rows in output matrix
                break;
            }
            case 4:
            {
                input_x_dim  = input->dimension(2);  // Number of columns in input matrix
                input_k_dim  = input->dimension(3);  // Number of rows in input matrix
                output_x_dim = output->dimension(2); // Number of columns in output matrix
                output_k_dim = output->dimension(3); // Number of rows in output matrix
                break;
            }
            default:
            {
                ARM_COMPUTE_RETURN_ERROR_MSG("Only 2 or 4 dimensions supported.");
            }
        }

        int ksize;
        switch (output_wf)
        {
#if defined(ARM_COMPUTE_ENABLE_SVE)
            case WeightFormat::OHWIo8:
            {
                ksize = 8;
                break;
            }
#endif /* ARM_COMPUTE_ENABLE_SVE */
            case WeightFormat::OHWIo4:
            {
                ksize = 4;
                break;
            }
            default:
            {
                ARM_COMPUTE_RETURN_ERROR_MSG("Unsupported weight format.");
                break;
            }
        }

        // output k_dim needs to be same as input but multiple of ksize
        int32_t rnd_up_input_kdim = arm_compute::ceil_to_multiple<int32_t, int32_t>(input_k_dim, ksize);
        ARM_COMPUTE_RETURN_ERROR_ON(rnd_up_input_kdim != output_k_dim);
        // output x_dim needs to be same as input
        ARM_COMPUTE_RETURN_ERROR_ON(input_x_dim != output_x_dim);
    }
    return Status{};
}

} // namespace arm_compute

#endif // defined(__aarch64__)
