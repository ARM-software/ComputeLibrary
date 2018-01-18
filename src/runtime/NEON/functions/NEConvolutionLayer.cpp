/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"

#include "arm_compute/core/NEON/kernels/arm32/NEGEMMAArch32Kernel.h"
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMAArch64Kernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
#include "arm_compute/core/NEON/kernels/assembly/gemm_interleaved.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a32_sgemm_8x6.hpp"
#include "arm_compute/core/NEON/kernels/assembly/kernels/a64_sgemm_12x8.hpp"
} // namespace arm_compute

#include <cmath>
#include <tuple>

namespace arm_compute
{
namespace
{
TensorShape get_reshaped_weights_shape(const ITensorInfo *weights, bool append_bias)
{
    const unsigned int mat_weights_cols = weights->dimension(3);
    const unsigned int mat_weights_rows = weights->dimension(0) * weights->dimension(1) * weights->dimension(2) + (append_bias ? 1 : 0);
    return TensorShape(mat_weights_cols, mat_weights_rows);
}
} // namespace

NEConvolutionLayerReshapeWeights::NEConvolutionLayerReshapeWeights(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _weights_reshape_kernel(), _weights_transposed_kernel(), _weights_reshaped(), _transpose1xW(false)
{
}

void NEConvolutionLayerReshapeWeights::configure(const ITensor *weights, const ITensor *biases, ITensor *output, bool transpose1xW)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEConvolutionLayerReshapeWeights::validate(weights->info(),
                                                                          (biases != nullptr) ? biases->info() : nullptr,
                                                                          output->info(),
                                                                          transpose1xW));

    // Check if bias are present, if yes they will be embedded to the weights matrix
    const bool append_biases = (biases != nullptr) && !is_data_type_quantized_asymmetric(weights->info()->data_type());
    //const unsigned bias_element  = (append_biases) ? 1 : 0;
    const ITensor *biases_to_use = (append_biases) ? biases : nullptr;

    _transpose1xW = transpose1xW;

    if(transpose1xW)
    {
        // Create tensor to store the reshaped weights
        TensorInfo info_wr = weights->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(get_reshaped_weights_shape(weights->info(), append_biases));

        _weights_reshaped.allocator()->init(info_wr);
        _memory_group.manage(&_weights_reshaped);

        _weights_reshape_kernel.configure(weights, biases, &_weights_reshaped);
        _weights_transposed_kernel.configure(&_weights_reshaped, output);

        _weights_reshaped.allocator()->allocate();
    }
    else
    {
        _weights_reshape_kernel.configure(weights, biases_to_use, output);
    }

    output->info()->set_quantization_info(weights->info()->quantization_info());
}

Status NEConvolutionLayerReshapeWeights::validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, bool transpose1xW)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    if(!is_data_type_quantized_asymmetric(weights->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(weights, output);
    }
    // Check if bias are present, if yes they will be embedded to the weights matrix
    const bool append_bias = (biases != nullptr);

    if(append_bias)
    {
        ARM_COMPUTE_ERROR_ON(is_data_type_quantized_asymmetric(weights->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(weights, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(3));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    // Checks performed when biases are present
    if(append_bias)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(3));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if(transpose1xW)
    {
        TensorInfo weights_reshaped = weights->clone()->set_tensor_shape(get_reshaped_weights_shape(weights, append_bias));
        ARM_COMPUTE_RETURN_ON_ERROR(NEWeightsReshapeKernel::validate(weights, biases, &weights_reshaped));
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMTranspose1xWKernel::validate(&weights_reshaped, output));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEWeightsReshapeKernel::validate(weights, biases, output));
    }

    return Status{};
}

void NEConvolutionLayerReshapeWeights::run()
{
    _memory_group.acquire();

    NEScheduler::get().schedule(&_weights_reshape_kernel, 3);

    if(_transpose1xW)
    {
        NEScheduler::get().schedule(&_weights_transposed_kernel, Window::DimY);
    }

    _memory_group.release();
}

namespace
{
TensorShape get_reshaped_weights_shape_conv(const ITensorInfo *weights, bool append_bias, bool is_fully_connected_convolution)
{
    unsigned int mat_weights_cols = weights->dimension(3);
    unsigned int mat_weights_rows = weights->dimension(0) * weights->dimension(1) * weights->dimension(2) + (append_bias ? 1 : 0);

    if(is_fully_connected_convolution)
    {
        // Create tensor to store the reshaped weights
        return TensorShape(mat_weights_cols, mat_weights_rows);
    }
    else
    {
        // Create tensor to store transposed weights
        const float transpose_width = 16.0f / weights->element_size();
        return TensorShape(mat_weights_rows * static_cast<unsigned int>(transpose_width), static_cast<unsigned int>(std::ceil(mat_weights_cols / transpose_width)));
    }
}

Status validate_and_initialize_values(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const PadStrideInfo &conv_info, const WeightsInfo &weights_info, DataType &dt,
                                      bool &append_bias,
                                      bool &are_weights_reshaped, unsigned int &kernel_width, unsigned int &kernel_height,
                                      bool &is_fully_connected_convolution, bool &is_interleaved_transposed, bool &is_quantized,
                                      unsigned int &mat_weights_cols, unsigned int &mat_weights_rows,
                                      unsigned int &conv_w, unsigned int &conv_h)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON(!weights_info.are_reshaped() && weights->dimension(2) != input->dimension(2));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON(weights_info.are_reshaped() && is_data_type_quantized_asymmetric(input->data_type()));

    dt           = input->data_type();
    is_quantized = is_data_type_quantized_asymmetric(dt);

    if(biases != nullptr)
    {
        if(is_quantized)
        {
            ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(!weights_info.are_reshaped() && biases->dimension(0) != weights->dimension(3));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    append_bias          = (biases != nullptr) && (!is_quantized);
    are_weights_reshaped = weights_info.are_reshaped();
    kernel_width         = (are_weights_reshaped) ? weights_info.kernel_size().first : weights->dimension(0);
    kernel_height        = (are_weights_reshaped) ? weights_info.kernel_size().second : weights->dimension(1);
    mat_weights_cols     = weights->dimension(3);
    mat_weights_rows     = weights->dimension(0) * weights->dimension(1) * weights->dimension(2) + (append_bias ? 1 : 0);

    std::tie(conv_w, conv_h) = scaled_dimensions(input->dimension(0), input->dimension(1), kernel_width, kernel_height,
                                                 conv_info);

    // Check if its a "fully connected" convolution
    is_fully_connected_convolution = ((conv_w == 1) && (conv_h == 1));
    is_interleaved_transposed      = (!is_fully_connected_convolution && !is_quantized);

    return Status{};
}
} // namespace

NEConvolutionLayer::NEConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _input_im2col_kernel(), _input_interleave_kernel(), _reshape_weights(), _mm_kernel(), _mm_optimised_kernel(nullptr), _mm_gemmlowp(memory_manager),
      _gemmlowp_output_stage(), _output_col2im_kernel(), _input_im2col_reshaped(), _input_interleaved_reshaped(), _weights_reshaped(), _gemm_output(), _tmp_output(), _workspace(), _append_bias(false),
      _is_fully_connected_convolution(false), _are_weights_reshaped(false), _is_quantized(false), _is_interleaved_transposed(false)
{
}

void NEConvolutionLayer::configure_mm(const ITensor *input, const ITensor *weights, ITensor *output)
{
    if(_is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = input->info()->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->info()->quantization_info();

        input->info()->set_quantization_info(QuantizationInfo(input_quantization_info.scale, -input_quantization_info.offset));
        weights->info()->set_quantization_info(QuantizationInfo(weights_quantization_info.scale, -weights_quantization_info.offset));

        _mm_gemmlowp.configure(input, weights, output, GEMMInfo(false, false, true /* Reshape weights only for the first run*/));

        // Revert back QuantizatioInfo as input and weights could be used in other convolution layers
        input->info()->set_quantization_info(input_quantization_info);
        weights->info()->set_quantization_info(weights_quantization_info);
    }
    else
    {
        _mm_kernel.configure(input, weights, output, 1.f);
    }
}

void NEConvolutionLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    DataType     dt{};
    unsigned int kernel_width     = 0;
    unsigned int kernel_height    = 0;
    unsigned int mat_weights_cols = 0;
    unsigned int mat_weights_rows = 0;
    unsigned int conv_w           = 0;
    unsigned int conv_h           = 0;

    Status status = validate_and_initialize_values(input->info(), weights->info(), (biases == nullptr) ? nullptr : biases->info(), conv_info, weights_info, dt, _append_bias, _are_weights_reshaped,
                                                   kernel_width, kernel_height,
                                                   _is_fully_connected_convolution, _is_interleaved_transposed, _is_quantized,
                                                   mat_weights_cols, mat_weights_rows, conv_w, conv_h);

    ARM_COMPUTE_ERROR_THROW_ON(status);

    const unsigned int fixed_point_position = input->info()->fixed_point_position();
    const ITensor     *biases_to_use        = (_append_bias) ? biases : nullptr;

#if defined(__arm__)
    if(NEScheduler::get().cpu_info().CPU == CPUTarget::ARMV7 && dt == DataType::F32)
    {
        _mm_optimised_kernel = support::cpp14::make_unique<NEGEMMAArch32Kernel>();
    }
#elif defined(__aarch64__)
    if(NEScheduler::get().cpu_info().CPU >= CPUTarget::ARMV8 && dt == DataType::F32)
    {
        _mm_optimised_kernel = support::cpp14::make_unique<NEGEMMAArch64Kernel>();
    }
#endif /* defined(__arm__) || defined(__aarch64__) */

    // Reshape weights if needed
    if(_mm_optimised_kernel != nullptr)
    {
        if(_are_weights_reshaped)
        {
            mat_weights_cols = weights_info.num_kernels();
            mat_weights_rows = weights->info()->dimension(1);
        }
        else
        {
            TensorShape reshaped_weights_shape{ mat_weights_cols, mat_weights_rows };

            // Create tensor to store the reshaped weights
            _weights_reshaped.allocator()->init(TensorInfo(reshaped_weights_shape, 1, dt, fixed_point_position));
            _reshape_weights.configure(weights, biases, &_weights_reshaped, false /* 1xW transpose */);
            weights = &_weights_reshaped;
        }
    }
    else
    {
        if(_are_weights_reshaped)
        {
            if(_is_fully_connected_convolution || _is_quantized)
            {
                mat_weights_cols = weights_info.num_kernels();
                mat_weights_rows = weights->info()->dimension(1);
            }
            else
            {
                const unsigned int transpose_width = 16 / input->info()->element_size();
                mat_weights_cols                   = weights_info.num_kernels();
                mat_weights_rows                   = weights->info()->dimension(0) / transpose_width + (_append_bias ? 1 : 0);
            }
        }
        else
        {
            TensorShape reshaped_weights_shape;

            if(_is_fully_connected_convolution || _is_quantized)
            {
                reshaped_weights_shape = TensorShape{ mat_weights_cols, mat_weights_rows };
            }
            else
            {
                // Create tensor to store transposed weights
                const float transpose_width = 16.0f / input->info()->element_size();
                reshaped_weights_shape      = TensorShape{ mat_weights_rows *static_cast<unsigned int>(transpose_width),
                                                           static_cast<unsigned int>(std::ceil(mat_weights_cols / transpose_width)) };
            }

            // Create tensor to store the reshaped weights
            _weights_reshaped.allocator()->init(TensorInfo(reshaped_weights_shape, 1, dt, fixed_point_position));
            _reshape_weights.configure(weights, biases_to_use, &_weights_reshaped, _is_interleaved_transposed /* 1xW transpose */);
            weights = &_weights_reshaped;
        }
    }

    // Create tensor to store im2col reshaped inputs
    const unsigned int mat_input_cols = mat_weights_rows;
    const unsigned int mat_input_rows = conv_w * conv_h;

    TensorShape shape_im2col(input->info()->tensor_shape());
    shape_im2col.set(0, mat_input_cols);
    shape_im2col.set(1, mat_input_rows);
    shape_im2col.set(2, 1);
    _input_im2col_reshaped.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_im2col));
    _memory_group.manage(&_input_im2col_reshaped);

    // Create tensor (interleave) to prepare input tensor for GEMM
    if(!_is_fully_connected_convolution && _mm_optimised_kernel == nullptr)
    {
        TensorShape shape_interleaved(shape_im2col);
        shape_interleaved.set(0, shape_interleaved.x() * 4);
        shape_interleaved.set(1, std::ceil(shape_interleaved.y() / 4.f));
        _input_interleaved_reshaped.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_interleaved));
        _memory_group.manage(&_input_interleaved_reshaped);
    }

    // Create GEMM output tensor
    TensorShape shape_gemm(_input_im2col_reshaped.info()->tensor_shape());
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
    const DataType gemm_data_type = _is_quantized ? DataType::S32 : dt;
    // GEMM output should be S32 for acquiring raw integer accumulator without quantized postprocessing for quantized asymmetric input.
    TensorInfo info_gemm(shape_gemm, 1, gemm_data_type, input->info()->fixed_point_position());
    info_gemm.set_quantization_info(output->info()->quantization_info());
    _gemm_output.allocator()->init(info_gemm);
    _memory_group.manage(&_gemm_output);

    // Configure kernels
    // Configure im2col
    _input_im2col_kernel.configure(input, &_input_im2col_reshaped, Size2D(kernel_width, kernel_height), conv_info, _append_bias);

    // Configure matrix multiply
#if defined(__arm__) || defined(__aarch64__)
    if(_mm_optimised_kernel != nullptr)
    {
        struct CPUInfo ci = NEScheduler::get().cpu_info();

        const int M = _gemm_output.info()->tensor_shape().y();
        const int N = _gemm_output.info()->tensor_shape().x();
        const int K = _input_im2col_reshaped.info()->tensor_shape().x();

#if defined(__arm__)
        GemmInterleaved<sgemm_8x6, float, float> gemm(&ci, M, N, K, false, false);
#elif defined(__aarch64__)
        GemmInterleaved<sgemm_12x8, float, float> gemm(&ci, M, N, K, false, false);
#endif /* defined(__arm__) || defined(__aarch64__) */

        constexpr size_t alignment = 4096;
        _workspace.allocator()->init(TensorInfo(TensorShape{ (gemm.get_working_size() + alignment - 1) * NEScheduler::get().num_threads() }, 1, DataType::U8));
        _memory_group.manage(&_workspace);

        // Configure matrix multiplication kernel
        _mm_optimised_kernel->configure(&_input_im2col_reshaped, weights, &_gemm_output, &_workspace);

        _workspace.allocator()->allocate();
    }
    else
#endif /* defined(__arm__) || defined(__aarch64__) */
    {
        if(_is_interleaved_transposed)
        {
            // Configure GEMMInterleave4x4. _input_interleaved_reshaped will be auto configured in the kernel
            _memory_group.manage(&_input_interleaved_reshaped);
            _input_interleave_kernel.configure(&_input_im2col_reshaped, &_input_interleaved_reshaped);

            // Configure GEMM
            configure_mm(&_input_interleaved_reshaped, weights, &_gemm_output);
            _input_interleaved_reshaped.allocator()->allocate();
        }
        else
        {
            configure_mm(&_input_im2col_reshaped, weights, &_gemm_output);
        }
    }

    _input_im2col_reshaped.allocator()->allocate();

    // Configure output stage for quantized case
    if(_is_quantized)
    {
        float multiplier = input->info()->quantization_info().scale * weights->info()->quantization_info().scale / output->info()->quantization_info().scale;
        int   output_multiplier, output_shift;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        _memory_group.manage(&_tmp_output);
        _gemmlowp_output_stage.configure(&_gemm_output, biases, &_tmp_output, output_multiplier, output_shift, output->info()->quantization_info().offset);
    }

    // Configure Col2Im
    _output_col2im_kernel.configure(_is_quantized ? &_tmp_output : &_gemm_output, output, Size2D(conv_w, conv_h));
    if(_is_quantized)
    {
        _tmp_output.allocator()->allocate();
    }
    _gemm_output.allocator()->allocate();

    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(0) != conv_w) || (output->info()->dimension(1) != conv_h), "Output shape does not match the expected one");

    // Allocate intermediate tensor
    if(!_are_weights_reshaped)
    {
        _weights_reshaped.allocator()->allocate();
    }
}

Status NEConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                    const WeightsInfo &weights_info)
{
    DataType     dt{};
    bool         append_bias{};
    bool         are_weights_reshaped{};
    bool         is_fully_connected_convolution{};
    bool         is_interleaved_transposed{};
    bool         is_quantized{};
    unsigned int kernel_width     = 0;
    unsigned int kernel_height    = 0;
    unsigned int mat_weights_cols = 0;
    unsigned int mat_weights_rows = 0;
    unsigned int conv_w           = 0;
    unsigned int conv_h           = 0;

    Status status = validate_and_initialize_values(input, weights, biases, conv_info, weights_info, dt, append_bias, are_weights_reshaped, kernel_width, kernel_height,
                                                   is_fully_connected_convolution, is_interleaved_transposed, is_quantized, mat_weights_cols, mat_weights_rows,
                                                   conv_w, conv_h);

    ARM_COMPUTE_RETURN_ON_ERROR(status);

    std::unique_ptr<ITensorInfo> reshaped_weights = weights->clone();
    bool                         optimised_kernel = false;

#if defined(__arm__)
    if(NEScheduler::get().cpu_info().CPU == CPUTarget::ARMV7 && dt == DataType::F32)
    {
        optimised_kernel = true;
    }
#elif defined(__aarch64__)
    if(NEScheduler::get().cpu_info().CPU >= CPUTarget::ARMV8 && dt == DataType::F32)
    {
        optimised_kernel = true;
    }
#endif /* defined(__arm__) || defined(__aarch64__) */

    // Reshape weights if needed
    if(optimised_kernel)
    {
        if(are_weights_reshaped)
        {
            mat_weights_cols = weights_info.num_kernels();
            mat_weights_rows = weights->dimension(1);
        }
        else
        {
            TensorShape reshaped_weights_shape{ mat_weights_cols, mat_weights_rows };

            // Create tensor to store the reshaped weights
            reshaped_weights->set_tensor_shape(get_reshaped_weights_shape_conv(weights, append_bias, is_fully_connected_convolution));
            ARM_COMPUTE_RETURN_ON_ERROR(NEConvolutionLayerReshapeWeights::validate(weights, biases, reshaped_weights.get(), !is_fully_connected_convolution /* 1xW transpose */));
            weights = reshaped_weights.get();
        }
    }
    else
    {
        if(are_weights_reshaped)
        {
            const unsigned int transpose_width = 16 / input->element_size();
            mat_weights_cols                   = weights_info.num_kernels();
            mat_weights_rows                   = weights->dimension(0) / transpose_width + (append_bias ? 1 : 0);
        }
        else
        {
            TensorShape reshaped_weights_shape;

            if(is_fully_connected_convolution || is_quantized)
            {
                reshaped_weights_shape = TensorShape{ mat_weights_cols, mat_weights_rows };
            }
            else
            {
                // Create tensor to store transposed weights
                const float transpose_width = 16.0f / input->element_size();
                reshaped_weights_shape      = TensorShape{ mat_weights_rows *static_cast<unsigned int>(transpose_width),
                                                           static_cast<unsigned int>(std::ceil(mat_weights_cols / transpose_width)) };
            }

            // Create tensor to store the reshaped weights
            reshaped_weights->set_tensor_shape(get_reshaped_weights_shape_conv(weights, append_bias, is_fully_connected_convolution));
            ARM_COMPUTE_RETURN_ON_ERROR(NEConvolutionLayerReshapeWeights::validate(weights, biases, reshaped_weights.get(), !is_fully_connected_convolution /* 1xW transpose */));
            weights = reshaped_weights.get();
        }
    }

    // Validate im2col
    const unsigned int mat_input_cols = mat_weights_rows;
    const unsigned int mat_input_rows = conv_w * conv_h;
    TensorShape        shape_im2col   = input->tensor_shape();
    shape_im2col.set(0, mat_input_cols);
    shape_im2col.set(1, mat_input_rows);
    shape_im2col.set(2, 1);
    TensorInfo im2_col_info = input->clone()->set_tensor_shape(shape_im2col);
    ARM_COMPUTE_RETURN_ON_ERROR(NEIm2ColKernel::validate(input, &im2_col_info, Size2D(weights->dimension(0), weights->dimension(1)), conv_info, append_bias));

    // Create GEMM output tensor
    TensorShape shape_gemm(im2_col_info.tensor_shape());
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
    TensorInfo gemm_output_info = input->clone()->set_tensor_shape(shape_gemm);

    // Validate GEMM interleave and multiply
    if(is_interleaved_transposed)
    {
        TensorShape shape_interleaved = shape_im2col;
        shape_interleaved.set(0, shape_interleaved.x() * 4);
        shape_interleaved.set(1, std::ceil(shape_interleaved.y() / 4.f));
        TensorInfo input_interleaved_info = input->clone()->set_tensor_shape(shape_interleaved);
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMInterleave4x4Kernel::validate(&im2_col_info, &input_interleaved_info));
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixMultiplyKernel::validate(&input_interleaved_info, weights, &gemm_output_info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixMultiplyKernel::validate(&im2_col_info, weights, &gemm_output_info));
    }

    ARM_COMPUTE_RETURN_ON_ERROR(NECol2ImKernel::validate(&gemm_output_info, output, Size2D(conv_w, conv_h)));

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((output->dimension(0) != conv_w) || (output->dimension(1) != conv_h), "Output shape does not match the expected one");

    return Status{};
}

void NEConvolutionLayer::run()
{
    // Run weights reshaping (Runs once for every configure)
    if(!_are_weights_reshaped)
    {
        _are_weights_reshaped = true;
        _reshape_weights.run();
    }

    _memory_group.acquire();

    // Run input reshaping
    NEScheduler::get().schedule(&_input_im2col_kernel, Window::DimY);

    // Runs matrix multiply on reshaped matrices
    if(_mm_optimised_kernel != nullptr)
    {
        NEScheduler::get().schedule(_mm_optimised_kernel.get(), Window::DimY);
    }
    else
    {
        if(_is_interleaved_transposed)
        {
            // Run interleave
            NEScheduler::get().schedule(&_input_interleave_kernel, Window::DimY);
        }

        // Runs matrix multiply on reshaped matrices
        if(_is_quantized)
        {
            _mm_gemmlowp.run();
        }
        else
        {
            NEScheduler::get().schedule(&_mm_kernel, Window::DimY);
        }
    }

    // Run output stage for quantized case
    if(_is_quantized)
    {
        _gemmlowp_output_stage.run();
    }

    // Reshape output matrix
    NEScheduler::get().schedule(&_output_col2im_kernel, Window::DimY);

    _memory_group.release();
}
} // namespace arm_compute
