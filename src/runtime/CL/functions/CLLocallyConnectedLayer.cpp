/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLLocallyConnectedLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLCol2ImKernel.h"
#include "src/core/CL/kernels/CLIm2ColKernel.h"
#include "src/core/CL/kernels/CLLocallyConnectedMatrixMultiplyKernel.h"
#include "src/core/CL/kernels/CLWeightsReshapeKernel.h"
#include "support/MemorySupport.h"

#include <cmath>
#include <tuple>

using namespace arm_compute;

namespace
{
void calculate_shapes(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                      TensorShape &shape_wr, TensorShape &shape_im2col, TensorShape &shape_gemm)
{
    ARM_COMPUTE_UNUSED(output);

    const unsigned int kernel_width  = weights->dimension(0);
    const unsigned int kernel_height = weights->dimension(1);

    bool has_bias = (biases != nullptr);

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(input->dimension(0),
                                                 input->dimension(1),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv_info);

    const size_t mat_weights_cols = weights->dimension(3);
    const size_t mat_weights_rows = weights->dimension(0) * weights->dimension(1) * weights->dimension(2) + ((has_bias) ? 1 : 0);
    const size_t mat_weights_num  = weights->dimension(4);

    shape_wr = TensorShape(mat_weights_cols, mat_weights_rows, mat_weights_num);

    const size_t mat_input_cols = mat_weights_rows;
    const size_t mat_input_rows = conv_w * conv_h;

    shape_im2col = input->tensor_shape();
    if(shape_im2col.num_dimensions() >= 3)
    {
        shape_im2col.remove_dimension(2);
    }
    shape_im2col.set(0, mat_input_cols);
    shape_im2col.set(1, mat_input_rows);

    shape_gemm = shape_im2col;
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
}
} // namespace

CLLocallyConnectedLayer::CLLocallyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _input_im2col_kernel(support::cpp14::make_unique<CLIm2ColKernel>()),
      _weights_reshape_kernel(support::cpp14::make_unique<CLWeightsReshapeKernel>()),
      _mm_kernel(support::cpp14::make_unique<CLLocallyConnectedMatrixMultiplyKernel>()),
      _output_col2im_kernel(support::cpp14::make_unique<CLCol2ImKernel>()),
      _input_im2col_reshaped(),
      _weights_reshaped(),
      _gemm_output(),
      _is_prepared(false),
      _original_weights(nullptr)
{
}

Status CLLocallyConnectedLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(2) != input->dimension(2));
    ARM_COMPUTE_RETURN_ERROR_ON(!conv_info.padding_is_symmetric());

    bool has_bias = (biases != nullptr);

    if(has_bias)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(3));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 2);
    }

    const unsigned int kernel_width  = weights->dimension(0);
    const unsigned int kernel_height = weights->dimension(1);

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(input->dimension(0), input->dimension(1), kernel_width, kernel_height,
                                                 conv_info);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((output->dimension(0) != conv_w) || (output->dimension(1) != conv_h), "Output shape does not match the expected one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(4) != (conv_w * conv_h), "Weights shape does not match the expected one");

    // Calculate intermediate buffer shapes
    TensorShape shape_wr;
    TensorShape shape_im2col;
    TensorShape shape_gemm;
    calculate_shapes(input, weights, biases, output, conv_info, shape_wr, shape_im2col, shape_gemm);

    TensorInfo weights_reshaped_info(shape_wr, 1, weights->data_type());
    TensorInfo input_im2col_reshaped_info(shape_im2col, 1, input->data_type());
    TensorInfo gemm_output_info(shape_gemm, 1, input->data_type());

    ARM_COMPUTE_RETURN_ON_ERROR(CLIm2ColKernel::validate(input, &input_im2col_reshaped_info, Size2D(kernel_width, kernel_height), conv_info, has_bias));
    ARM_COMPUTE_RETURN_ON_ERROR(CLWeightsReshapeKernel::validate(weights, biases, &weights_reshaped_info));
    ARM_COMPUTE_RETURN_ON_ERROR(CLLocallyConnectedMatrixMultiplyKernel::validate(&input_im2col_reshaped_info, &weights_reshaped_info, &gemm_output_info));
    ARM_COMPUTE_RETURN_ON_ERROR(CLCol2ImKernel::validate(&gemm_output_info, output, Size2D(conv_w, conv_h)));

    return Status{};
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
void CLLocallyConnectedLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info);
}
#pragma GCC diagnostic pop

void CLLocallyConnectedLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                        const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLLocallyConnectedLayer::validate(input->info(), weights->info(), biases == nullptr ? nullptr : biases->info(), output->info(), conv_info));

    bool _has_bias    = (biases != nullptr);
    _original_weights = weights;
    _is_prepared      = false;

    const unsigned int kernel_width  = weights->info()->dimension(0);
    const unsigned int kernel_height = weights->info()->dimension(1);

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1), kernel_width, kernel_height,
                                                 conv_info);

    // Calculate intermediate buffer shapes
    TensorShape shape_wr;
    TensorShape shape_im2col;
    TensorShape shape_gemm;
    calculate_shapes(input->info(), weights->info(), biases == nullptr ? nullptr : biases->info(), output->info(), conv_info, shape_wr, shape_im2col, shape_gemm);

    _weights_reshaped.allocator()->init(TensorInfo(shape_wr, 1, weights->info()->data_type()));
    _input_im2col_reshaped.allocator()->init(TensorInfo(shape_im2col, 1, input->info()->data_type()));
    _gemm_output.allocator()->init(TensorInfo(shape_gemm, 1, input->info()->data_type()));

    // Manage intermediate buffers
    _memory_group.manage(&_input_im2col_reshaped);
    _memory_group.manage(&_gemm_output);

    // Configure kernels
    _input_im2col_kernel->configure(compile_context, input, &_input_im2col_reshaped, Size2D(kernel_width, kernel_height), conv_info, _has_bias);
    _weights_reshape_kernel->configure(compile_context, weights, biases, &_weights_reshaped);
    _mm_kernel->configure(compile_context, &_input_im2col_reshaped, &_weights_reshaped, &_gemm_output);
    _output_col2im_kernel->configure(compile_context, &_gemm_output, output, Size2D(conv_w, conv_h));

    // Allocate intermediate tensors
    _input_im2col_reshaped.allocator()->allocate();
    _gemm_output.allocator()->allocate();

    CLScheduler::get().tune_kernel_static(*_input_im2col_kernel);
}

void CLLocallyConnectedLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Run input reshaping
    CLScheduler::get().enqueue(*_input_im2col_kernel);

    // Runs vector matrix multiply on reshaped matrices
    CLScheduler::get().enqueue(*_mm_kernel);

    // Reshape output matrix
    CLScheduler::get().enqueue(*_output_col2im_kernel.get(), false);
}

void CLLocallyConnectedLayer::prepare()
{
    if(!_is_prepared)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        // Run weights reshaping and mark original weights tensor as unused
        _weights_reshaped.allocator()->allocate();
        CLScheduler::get().enqueue(*_weights_reshape_kernel);
        _original_weights->mark_as_unused();

        CLScheduler::get().queue().finish();
        _is_prepared = true;
    }
}
