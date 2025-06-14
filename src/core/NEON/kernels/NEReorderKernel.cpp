/*
 * Copyright (c) 2023-2025 Arm Limited.
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
#include "arm_compute/runtime/Scheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/NEON/kernels/arm_gemm/transform.hpp"

#include <map>

namespace arm_compute
{

namespace
{
struct TransformParams
{
    int              interleave_by;
    int              block_by;
    bool             transpose;
    arm_gemm::VLType vltype;

    bool operator<(const TransformParams &b) const
    {
        if (interleave_by != b.interleave_by)
            return interleave_by < b.interleave_by;
        if (block_by != b.block_by)
            return block_by < b.block_by;
        if (static_cast<int>(vltype) != static_cast<int>(b.vltype))
            return static_cast<int>(vltype) < static_cast<int>(b.vltype);
        return transpose < b.transpose;
    }
};

std::map<TransformParams, void (*)(float *, const float *, int, int, int, int, int)> supported_float_transforms = {
    {{4, 1, true, arm_gemm::VLType::None}, &arm_gemm::Transform<4, 1, true, arm_gemm::VLType::None, float, float>},
    {{4, 1, false, arm_gemm::VLType::None}, &arm_gemm::Transform<4, 1, false, arm_gemm::VLType::None, float, float>},
    {{8, 1, false, arm_gemm::VLType::None}, &arm_gemm::Transform<8, 1, false, arm_gemm::VLType::None, float, float>},
    {{8, 1, true, arm_gemm::VLType::None}, &arm_gemm::Transform<8, 1, true, arm_gemm::VLType::None, float, float>},
#ifdef ARM_COMPUTE_ENABLE_SVE
    // When there is an asm kernel, use formula in transform.cpp to get the interleave_by_ number
    {{1, 1, true, arm_gemm::VLType::SVE}, &arm_gemm::Transform<1, 1, true, arm_gemm::VLType::SVE, float, float>},
#endif // ARM_COMPUTE_ENABLE_SVE
};

std::map<TransformParams, void (*)(bfloat16 *, const float *, int, int, int, int, int)> supported_bf16_transforms = {
#ifdef ARM_COMPUTE_ENABLE_BF16
    {{4, 4, true, arm_gemm::VLType::None}, &arm_gemm::Transform<4, 4, true, arm_gemm::VLType::None, bfloat16, float>},
    {{4, 4, false, arm_gemm::VLType::None}, &arm_gemm::Transform<4, 4, false, arm_gemm::VLType::None, bfloat16, float>},
    {{8, 4, false, arm_gemm::VLType::None}, &arm_gemm::Transform<8, 4, false, arm_gemm::VLType::None, bfloat16, float>},
    {{8, 4, true, arm_gemm::VLType::None}, &arm_gemm::Transform<8, 4, true, arm_gemm::VLType::None, bfloat16, float>},
#ifdef ARM_COMPUTE_ENABLE_SVE
    {{2, 4, true, arm_gemm::VLType::SVE}, &arm_gemm::Transform<2, 4, true, arm_gemm::VLType::SVE, bfloat16, float>},
#endif // ARM_COMPUTE_ENABLE_SVE
#endif // ARM_COMPUTE_ENABLE_BF16
};

#ifdef ARM_COMPUTE_ENABLE_SVE

// Calculate the interleave_by parameter needed for SVE kernels
// using the formula listed in transform.cpp
template <typename TOut>
inline int get_sve_interleave_by(int interleave_by, int block_by)
{
    return interleave_by / (get_vector_length<TOut>() / block_by);
}
#endif // ARM_COMPUTE_ENABLE_SVE

} // namespace

void NEReorderKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON_MSG(_input->info()->data_type() != DataType::F32, "Unsupported input data type");
    const int ksize_rows_elements = _xmax * _ksize;
    const int jump_rows           = ksize_rows_elements * window.x().start();
    const int k_start             = window.x().start() * _ksize;
    const int k_end               = std::min(window.x().end() * _ksize, _kmax);
    const int stride              = _transpose ? _kmax : _xmax;
    const int block_by            = arm_compute::block_by(_output_wf);
    const int interleave_by       = arm_compute::interleave_by(_output_wf);
    ARM_COMPUTE_ERROR_ON(interleave_by != 4 && interleave_by != 8);

    if (k_start >= k_end)
        return;

    switch (_output->info()->data_type())
    {
        case DataType::F32:
        {
            void (*transform_func)(float *, const float *, int, int, int, int, int) = nullptr;
#ifdef ARM_COMPUTE_ENABLE_SVE
            if (CPUInfo::get().has_sve())
            {
                TransformParams tparams = {get_sve_interleave_by<float>(interleave_by, block_by), block_by, _transpose,
                                           arm_gemm::VLType::SVE};
                if (supported_float_transforms.count(tparams))
                {
                    transform_func = supported_float_transforms[tparams];
                }
            }
#endif // ARM_COMPUTE_ENABLE_SVE
            if (transform_func == nullptr)
            {
                transform_func =
                    supported_float_transforms[{interleave_by, block_by, _transpose, arm_gemm::VLType::None}];
            }
            transform_func(reinterpret_cast<float *>(_output->buffer()) + jump_rows,
                           reinterpret_cast<float *>(_input->buffer()), stride, k_start, k_end, 0, _xmax);
            break;
        }
        case DataType::BFLOAT16:
        {
            if (CPUInfo::get().has_bf16())
            {
                void (*transform_func)(bfloat16 *, const float *, int, int, int, int, int) = nullptr;
#ifdef ARM_COMPUTE_ENABLE_SVE
                if (CPUInfo::get().has_sve())
                {
                    TransformParams tparams = {get_sve_interleave_by<bfloat16>(interleave_by, block_by), block_by,
                                               _transpose, arm_gemm::VLType::SVE};
                    if (supported_bf16_transforms.count(tparams))
                        transform_func = supported_bf16_transforms[tparams];
                }
#endif // ARM_COMPUTE_ENABLE_SVE
                if (transform_func == nullptr)
                {
                    transform_func =
                        supported_bf16_transforms[{interleave_by, block_by, _transpose, arm_gemm::VLType::None}];
                }
                transform_func(reinterpret_cast<bfloat16 *>(_output->buffer()) + jump_rows,
                               reinterpret_cast<float *>(_input->buffer()), stride, k_start, k_end, 0, _xmax);
                break;
            }
            ARM_COMPUTE_ERROR("Trying to run BF16 on unsupported machine\n");
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported data type!");
            break;
        }
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
                                arm_compute::WeightFormat output_wf,
                                bool                      transpose)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, input_wf, output_wf);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), input_wf, output_wf, transpose));

    // Set variables
    _input     = input;
    _output    = output;
    _input_wf  = input_wf;
    _output_wf = output_wf;
    _transpose = transpose;

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
    ARM_COMPUTE_ERROR_ON(arm_compute::interleave_by(_output_wf) != 4 && arm_compute::interleave_by(_output_wf) != 8);
    _ksize      = arm_compute::interleave_by(_output_wf);
    window_size = _kmax / _ksize;
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
                                 arm_compute::WeightFormat output_wf,
                                 bool                      transpose)
{
    ARM_COMPUTE_UNUSED(input_wf);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() != DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(output->data_type() != DataType::F32 && output->data_type() != DataType::BFLOAT16);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);

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

    int ksize         = 0;
    int interleave_by = arm_compute::interleave_by(output_wf);
    int block_by      = arm_compute::block_by(output_wf);
    ARM_COMPUTE_RETURN_ERROR_ON(interleave_by != 4 && interleave_by != 8);
    ksize = interleave_by;

    // output x_dim needs to be same as input but multiple of block_by
    int32_t rnd_up_input_xdim = arm_compute::ceil_to_multiple<int32_t, int32_t>(input_x_dim, block_by);
    ARM_COMPUTE_RETURN_ERROR_ON(rnd_up_input_xdim != output_x_dim);
    // output k_dim needs to be same as input but multiple of ksize
    int32_t rnd_up_input_kdim = arm_compute::ceil_to_multiple<int32_t, int32_t>(input_k_dim, ksize);
    ARM_COMPUTE_RETURN_ERROR_ON(rnd_up_input_kdim != output_k_dim);
    // output x_dim needs to be same as input
    ARM_COMPUTE_RETURN_ERROR_ON(input_x_dim != output_x_dim);

    switch (output->data_type())
    {
        case DataType::F32:
        {
#ifdef ARM_COMPUTE_ENABLE_SVE
            if (CPUInfo::get().has_sve() &&
                supported_float_transforms.count({get_sve_interleave_by<float>(interleave_by, block_by), block_by,
                                                  transpose, arm_gemm::VLType::SVE}))
                break;
#endif // ARM_COMPUTE_ENABLE_SVE
            ARM_COMPUTE_RETURN_ERROR_ON(
                !supported_float_transforms.count({interleave_by, block_by, transpose, arm_gemm::VLType::None}));
            break;
        }
        case DataType::BFLOAT16:
        {
            ARM_COMPUTE_RETURN_ERROR_ON(!CPUInfo::get().has_bf16());
#ifdef ARM_COMPUTE_ENABLE_SVE
            if (CPUInfo::get().has_sve() &&
                supported_bf16_transforms.count({get_sve_interleave_by<bfloat16>(interleave_by, block_by), block_by,
                                                 transpose, arm_gemm::VLType::SVE}))
                break;
#endif // ARM_COMPUTE_ENABLE_SVE
            ARM_COMPUTE_RETURN_ERROR_ON(
                !supported_bf16_transforms.count({interleave_by, block_by, transpose, arm_gemm::VLType::None}));
            break;
        }
        default:
        {
            ARM_COMPUTE_RETURN_ERROR_MSG("Unsupported output data type");
            break;
        }
    }
    return Status{};
}

} // namespace arm_compute

#endif // defined(__aarch64__)
