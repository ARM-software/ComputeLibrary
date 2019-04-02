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
#include "arm_compute/runtime/CL/functions/CLWinogradConvolutionLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

namespace
{
Size2D winograd_output_tile(const Size2D &input_dims, const Size2D &kernel_dims, DataLayout data_layout)
{
    Size2D output_tile = Size2D{};

    const unsigned int kernel_max_dim = std::max(kernel_dims.width, kernel_dims.height);

    // Check if the input spatial dimensions are smaller than 4
    const bool is_input_lt4_nchw = (input_dims.width <= 4 && input_dims.height <= 4) && (data_layout == DataLayout::NCHW);

    if(kernel_max_dim == 3U)
    {
        if(kernel_dims == Size2D(3U, 3U))
        {
            output_tile = is_input_lt4_nchw ? Size2D(2U, 2U) : Size2D(4U, 4U);
        }
        else if(kernel_dims == Size2D(3U, 1U))
        {
            output_tile = is_input_lt4_nchw ? Size2D(2U, 1U) : Size2D(4U, 1U);
        }
        else
        {
            output_tile = is_input_lt4_nchw ? Size2D(1U, 2U) : Size2D(1U, 4U);
        }
    }
    else if(kernel_max_dim == 5U)
    {
        output_tile = Size2D(kernel_dims.width == 1 ? 1U : 4U,
                             kernel_dims.height == 1 ? 1U : 4U);
    }
    else if(kernel_max_dim == 7U)
    {
        output_tile = Size2D(kernel_dims.width == 1 ? 1U : 2U,
                             kernel_dims.height == 1 ? 1U : 2U);
    }

    return output_tile;
}

bool check_support_fast_math(const Size2D &output_tile, const Size2D &kernel_size)
{
    // Check if we want to configure a Winograd configuration which requires fast math
    using WinogradConfiguration = std::pair<std::pair<int, int>, std::pair<int, int>>;

    std::vector<WinogradConfiguration> fast_math_winograd =
    {
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5)),
        WinogradConfiguration(std::pair<int, int>(2, 2), std::pair<int, int>(7, 7))
    };

    auto p = std::make_pair(std::pair<int, int>(output_tile.width, output_tile.height),
                            std::pair<int, int>(kernel_size.width, kernel_size.height));

    return std::find(fast_math_winograd.begin(), fast_math_winograd.end(), p) != fast_math_winograd.end();
}
} // namespace

CLWinogradConvolutionLayer::CLWinogradConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _batched_mm(memory_manager), _input_transform(), _filter_transform(), _output_transform(), _input0(), _input1(), _batched_mm_output(), _original_weights(nullptr),
      _is_prepared(false)
{
}

void CLWinogradConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info,
                                           bool enable_fast_math)
{
    // Get indices for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape, kernel size and output tile
    const Size2D input_dims  = Size2D(input->info()->tensor_shape()[idx_width], input->info()->tensor_shape()[idx_height]);
    const Size2D kernel_size = Size2D(weights->info()->tensor_shape()[idx_width], weights->info()->tensor_shape()[idx_height]);
    const Size2D output_tile = winograd_output_tile(input_dims, kernel_size, input->info()->data_layout());

    // Check if the Winograd configuration requires fast math
    if(!enable_fast_math)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32); //disable winograd for fp16 if fast math is false.
        ARM_COMPUTE_ERROR_ON_MSG(check_support_fast_math(output_tile, kernel_size), "This Winograd configuration requires enable_fast_math=true");
    }
    const WinogradInfo winograd_info = WinogradInfo(output_tile,
                                                    kernel_size,
                                                    input_dims,
                                                    conv_info,
                                                    input->info()->data_layout());

    _is_prepared      = false;
    _original_weights = weights;

    // Manage intermediate tensors
    _memory_group.manage(&_input0);
    _memory_group.manage(&_batched_mm_output);

    // Do not manage _input1 as it contains the weights

    // Configure input transform
    _input_transform.configure(input, &_input0, winograd_info);

    // Configure filter transform
    _filter_transform.configure(weights, &_input1, winograd_info);

    // Configure batched matrix multiply
    _batched_mm.configure(&_input0, &_input1, nullptr, &_batched_mm_output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/, 0, false, false, GEMMLowpOutputStageInfo(),
                                                                                                 (input->info()->data_type() == DataType::F16)));

    // Configure output transform
    _output_transform.configure(&_batched_mm_output, biases, output, winograd_info, act_info);

    // Allocate temporary tensors
    _input0.allocator()->allocate();
    _batched_mm_output.allocator()->allocate();
}

Status CLWinogradConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                            const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    // Get indeces for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape, kernel size and output tile
    const Size2D input_dims  = Size2D(input->tensor_shape()[idx_width], input->tensor_shape()[idx_height]);
    const Size2D kernel_size = Size2D(weights->tensor_shape()[idx_width], weights->tensor_shape()[idx_height]);
    const Size2D output_tile = winograd_output_tile(input_dims, kernel_size, input->data_layout());

    // Check if the Winograd configuration requires fast math
    if(!enable_fast_math)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32); //disable winograd for fp16 if fast math is false.
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(check_support_fast_math(output_tile, kernel_size), "This Winograd configuration requires enable_fast_math=true");
    }

    const WinogradInfo winograd_info = WinogradInfo(output_tile,
                                                    kernel_size,
                                                    input_dims,
                                                    conv_info,
                                                    input->data_layout());

    // Validate input transform
    const TensorShape input0_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*input, winograd_info);
    const TensorInfo  input0       = input->clone()->set_tensor_shape(input0_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLWinogradInputTransform::validate(input, &input0, winograd_info));

    // Validate filter transform
    const TensorShape input1_shape = misc::shape_calculator::compute_winograd_filter_transform_shape(*weights, winograd_info);
    const TensorInfo  input1       = weights->clone()->set_tensor_shape(input1_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLWinogradFilterTransformKernel::validate(weights, &input1, winograd_info));

    // Validate batched matrix multiply
    TensorShape batched_mm_output_shape = input0.tensor_shape();
    batched_mm_output_shape[0]          = input1.tensor_shape()[0];
    const TensorInfo batched_mm_output  = input0.clone()->set_tensor_shape(batched_mm_output_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(&input0, &input1, nullptr, &batched_mm_output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/, 0, false, false,
                                                                                                                     GEMMLowpOutputStageInfo(), (input->data_type() == DataType::F16))));

    // Configure output transform
    ARM_COMPUTE_RETURN_ON_ERROR(CLWinogradOutputTransformKernel::validate(&batched_mm_output, biases, output, winograd_info));

    // Validate Activation Layer
    if(act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(output, nullptr, act_info));
    }

    return Status{};
}

void CLWinogradConvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Run input transform
    _input_transform.run();

    // Run batched matrix multiplication
    _batched_mm.run();

    // Run output transform
    CLScheduler::get().enqueue(_output_transform);
}

void CLWinogradConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        // Run filter transform and mark original weights as unused
        _input1.allocator()->allocate();
        CLScheduler::get().enqueue(_filter_transform, false);
        _original_weights->mark_as_unused();

        // Prepare GEMM and release reshaped weights if marked unused by CLGEMM
        _batched_mm.prepare();
        if(!_input1.is_used())
        {
            _input1.allocator()->free();
        }

        CLScheduler::get().queue().finish();
        _is_prepared = true;
    }
}
