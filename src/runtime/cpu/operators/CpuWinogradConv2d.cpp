/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/runtime/cpu/operators/CpuWinogradConv2d.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/kernels/convolution/common/utils.hpp"
#include "src/core/NEON/kernels/convolution/winograd/winograd.hpp"
#include "src/core/cpu/kernels/CpuWinogradConv2dKernel.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/cpu/operators/CpuActivation.h"
#include "src/runtime/cpu/operators/CpuPermute.h"
#include "src/runtime/cpu/operators/CpuWinogradConv2d.h"
#include "src/runtime/cpu/utils/CpuAuxTensorHandler.h"

#include "support/Cast.h"

#include <set>

namespace arm_compute
{
namespace cpu
{
using namespace arm_compute::experimental;
using namespace arm_compute::utils::cast;

namespace
{
arm_gemm::Activation arm_gemm_activation_from_acl_activation(const ActivationLayerInfo &act_info)
{
    switch(act_info.activation())
    {
        case ActivationLayerInfo::ActivationFunction::RELU:
        {
            return arm_gemm::Activation(arm_gemm::Activation::Type::ReLU, act_info.a(), act_info.b());
        }
        case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
        {
            return arm_gemm::Activation(arm_gemm::Activation::Type::BoundedReLU, act_info.a(), act_info.b());
        }
        default:
        {
            return arm_gemm::Activation(arm_gemm::Activation::Type::None);
        }
    }
}

inline Status validate_kernel_3x3(const Size2D input_dims, const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    if(input->data_type() == DataType::F32)
    {
        if(input_dims.width > 4 && input_dims.height > 4)
        {
            ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<float, 4, 4, 3, 3>::validate(input, input0, winograd_info)));
            ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<float, 4, 4, 3, 3>::validate(weights, input1, winograd_info)));
            ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<float, 4, 4, 3, 3>::validate(batched_mm_output, biases, output, winograd_info)));
        }
        else
        {
            ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<float, 2, 2, 3, 3>::validate(input, input0, winograd_info)));
            ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<float, 2, 2, 3, 3>::validate(weights, input1, winograd_info)));
            ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<float, 2, 2, 3, 3>::validate(batched_mm_output, biases, output, winograd_info)));
        }
    }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else if(input->data_type() == DataType::F16)
    {
        ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<__fp16, 4, 4, 3, 3>::validate(input, input0, winograd_info)));
        ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<__fp16, 4, 4, 3, 3>::validate(weights, input1, winograd_info)));
        ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<__fp16, 4, 4, 3, 3>::validate(batched_mm_output, biases, output, winograd_info)));
    }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

    if(act_info.enabled())
    {
        CpuActivation::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_5x5(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<float, 2, 2, 5, 5>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<float, 2, 2, 5, 5>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<float, 2, 2, 5, 5>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        CpuActivation::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_3x1(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<float, 1, 6, 1, 3>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<float, 1, 6, 1, 3>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<float, 1, 6, 1, 3>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        CpuActivation::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_1x3(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<float, 6, 1, 3, 1>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<float, 6, 1, 3, 1>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<float, 6, 1, 3, 1>::validate(batched_mm_output, biases, output, winograd_info)));

    if(act_info.enabled())
    {
        CpuActivation::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_5x1(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<float, 1, 4, 1, 5>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<float, 1, 4, 1, 5>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<float, 1, 4, 1, 5>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        CpuActivation::validate(output, nullptr, act_info);
    }
    return Status{};
}
inline Status validate_kernel_1x5(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<float, 4, 1, 5, 1>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<float, 4, 1, 5, 1>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<float, 4, 1, 5, 1>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        CpuActivation::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_7x1(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<float, 1, 2, 1, 7>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<float, 1, 2, 1, 7>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<float, 1, 2, 1, 7>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        CpuActivation::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_1x7(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformInputKernel<float, 2, 1, 7, 1>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformWeightsKernel<float, 2, 1, 7, 1>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((CpuWinogradConv2dTransformOutputKernel<float, 2, 1, 7, 1>::validate(batched_mm_output, biases, output, winograd_info)));

    if(act_info.enabled())
    {
        CpuActivation::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Tensor4DShape internal_get_input_shape(const ITensorInfo *input)
{
    const DataLayout data_layout = input->data_layout();
    const int        in_width    = input->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH));
    const int        in_height   = input->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT));
    const int        in_channels = input->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL));
    const int        in_batches  = input->dimension(3);

    return Tensor4DShape{ in_batches, in_height, in_width, in_channels };
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.stride().first != 1 || conv_info.stride().second != 1, "Winograd layer only supports unit strides.");
    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }
    return ICpuWinogradConv2dTransformWeightsKernel::validate(input, weights);
}
Size2D winograd_output_tile(const Size2D &input_dims, const Size2D &kernel_dims, DataType data_type)
{
    Size2D output_tile = Size2D{};
    if(kernel_dims == Size2D(3U, 3U))
    {
        output_tile = (input_dims.width <= 4 || input_dims.height <= 4) ? Size2D(2U, 2U) : Size2D(4U, 4U);
        if(data_type == DataType::F16)
        {
            output_tile = Size2D(4U, 4U);
        }
    }
    else if(kernel_dims == Size2D(5U, 5U))
    {
        output_tile = Size2D(2U, 2U);
    }
    else if(kernel_dims == Size2D(1U, 3U))
    {
        output_tile = Size2D(1U, 6U);
    }
    else if(kernel_dims == Size2D(3U, 1U))
    {
        output_tile = Size2D(6U, 1U);
    }
    else if(kernel_dims == Size2D(1U, 5U))
    {
        output_tile = Size2D(1U, 4U);
    }
    else if(kernel_dims == Size2D(5U, 1U))
    {
        output_tile = Size2D(4U, 1U);
    }
    else if(kernel_dims == Size2D(7U, 1U))
    {
        output_tile = Size2D(2U, 1U);
    }
    else if(kernel_dims == Size2D(1U, 7U))
    {
        output_tile = Size2D(1U, 2U);
    }
    return output_tile;
}

bool check_support_fast_math(const Size2D &output_tile, const Size2D &kernel_size, DataType data_type)
{
    // Check if we want to configure a Winograd configuration which requires fast math
    using WinogradConfiguration = std::pair<std::pair<int, int>, std::pair<int, int>>;

    const std::vector<WinogradConfiguration> fast_math_winograd_f16 =
    {
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(3, 3))
    };

    const std::vector<WinogradConfiguration> fast_math_winograd_f32 =
    {
        WinogradConfiguration(std::pair<int, int>(2, 2), std::pair<int, int>(5, 5)),
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5))
    };

    auto p = std::make_pair(std::pair<int, int>(output_tile.width, output_tile.height),
                            std::pair<int, int>(kernel_size.width, kernel_size.height));

    switch(data_type)
    {
        case DataType::F16:
            return std::find(fast_math_winograd_f16.begin(), fast_math_winograd_f16.end(), p) != fast_math_winograd_f16.end();
        case DataType::F32:
            return std::find(fast_math_winograd_f32.begin(), fast_math_winograd_f32.end(), p) != fast_math_winograd_f32.end();
        default:
            return false;
    }
}

inline bool fuse_function_supported(const ActivationLayerInfo &act_info)
{
    return act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU || act_info.activation() == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU;
}

} // namespace

CpuWinogradConv2d::CpuWinogradConv2d()
    : _gemm_function(std::make_unique<CpuGemm>()),
      _activation_func(std::make_unique<CpuActivation>()),
      _permute_input(std::make_unique<CpuPermute>()),
      _permute_output(std::make_unique<CpuPermute>()),
      _permute_weights(std::make_unique<CpuPermute>()),
      _transform_input_kernel(nullptr),
      _transform_weights_kernel(nullptr),
      _transform_output_kernel(nullptr),
      _data_layout(),
      _aux_mem(AuxTensorIdx::Count),
      _input_nhwc(),
      _output_nhwc(),
      _input_workspace(),
      _kernel_storage(),
      _output_workspace(),
      _input_transformed(),
      _output_transformed(),
      _weights_hwio(),
      _run_activation(false),
      _is_prepared(false)
{
}

CpuWinogradConv2d::~CpuWinogradConv2d() = default;

void CpuWinogradConv2d::configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst,
                                  const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, biases, dst, conv_info));

    // Get indices for the width and height
    _data_layout                   = src->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);

    const Size2D   input_dims  = Size2D(src->dimension(width_idx), src->dimension(height_idx));
    const Size2D   kernel_size = Size2D(weights->dimension(width_idx), weights->dimension(height_idx));
    const DataType data_type   = src->data_type();
    const Size2D   output_tile = winograd_output_tile(input_dims, kernel_size, data_type);

    // Check if the Winograd configuration requires fast math
    if(!enable_fast_math)
    {
        ARM_COMPUTE_ERROR_ON_MSG(check_support_fast_math(output_tile, kernel_size, data_type),
                                 "This Winograd configuration requires enable_fast_math=true");
    }

    _is_prepared = false;

    std::unique_ptr<ICpuWinogradConv2dTransformInputKernel>   transform_input_kernel;
    std::unique_ptr<ICpuWinogradConv2dTransformWeightsKernel> transform_weights_kernel;
    std::unique_ptr<ICpuWinogradConv2dTransformOutputKernel>  transform_output_kernel;

    int n_gemms = 1;
    int N_BLOCK = 1; // Size of block used by GEMM.
    if(data_type == DataType::F32)
    {
        if(kernel_size == Size2D(3, 3))
        {
            if(src->dimension(width_idx) > 4 && src->dimension(height_idx) > 4)
            {
                using config             = CpuWinogradConv2dConfiguration<float, float, 4, 4, 3, 3>;
                transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
                transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
                transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
                n_gemms                  = config::WinogradBase::N_GEMMS;
                N_BLOCK                  = config::WinogradConv::N_BLOCK;
            }
            else
            {
                using config             = CpuWinogradConv2dConfiguration<float, float, 2, 2, 3, 3>;
                transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
                transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
                transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
                n_gemms                  = config::WinogradBase::N_GEMMS;
                N_BLOCK                  = config::WinogradConv::N_BLOCK;
            }
        }
        else if(kernel_size == Size2D(5, 5))
        {
            using config             = CpuWinogradConv2dConfiguration<float, float, 2, 2, 5, 5>;
            transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
        else if(kernel_size == Size2D(1, 3))
        {
            using config             = CpuWinogradConv2dConfiguration<float, float, 6, 1, 3, 1>;
            transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
        else if(kernel_size == Size2D(3, 1))
        {
            using config             = CpuWinogradConv2dConfiguration<float, float, 1, 6, 1, 3>;
            transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
        else if(kernel_size == Size2D(1, 5))
        {
            using config             = CpuWinogradConv2dConfiguration<float, float, 4, 1, 5, 1>;
            transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
        else if(kernel_size == Size2D(5, 1))
        {
            using config             = CpuWinogradConv2dConfiguration<float, float, 1, 4, 1, 5>;
            transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
        else if(kernel_size == Size2D(1, 7))
        {
            using config             = CpuWinogradConv2dConfiguration<float, float, 2, 1, 7, 1>;
            transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
        else if(kernel_size == Size2D(7, 1))
        {
            using config             = CpuWinogradConv2dConfiguration<float, float, 1, 2, 1, 7>;
            transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
        else
        {
            ARM_COMPUTE_ERROR("Not supported.");
        }
    }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else if(data_type == DataType::F16)
    {
        if(kernel_size == Size2D(3, 3))
        {
            using config             = CpuWinogradConv2dConfiguration<__fp16, __fp16, 4, 4, 3, 3>;
            transform_input_kernel   = std::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = std::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = std::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
        else
        {
            ARM_COMPUTE_ERROR("Not supported.");
        }
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else
    {
        ARM_COMPUTE_ERROR("Not supported.");
    }

    const PaddingType use_padding_type = (conv_info.pad_top() != 0u || conv_info.pad_left() != 0) ? PADDING_SAME : PADDING_VALID;
    const bool        use_same_padding = use_padding_type == PADDING_SAME;

    // Get convolved dimensions
    const int in_channels  = src->dimension(channel_idx);
    const int out_channels = dst->dimension(channel_idx);

    const Tensor4DShape in_shape(internal_get_input_shape(src));
    const size_t        data_type_size = src->element_size();
    // Get the memory required to instantiate a new Winograd operator.
    constexpr size_t storage_alignment = 64;

    // Kernel Storage
    const size_t kernel_storage_size = transform_weights_kernel->get_weight_storage_size(out_channels,
                                                                                         in_channels)
                                       * data_type_size;

    // Input storage
    const size_t input_storage_size = transform_input_kernel->get_input_storage_size(in_shape.n_batches, in_shape.n_channels, in_shape.n_rows, in_shape.n_cols,
                                                                                     use_same_padding)
                                      * data_type_size;

    // Output storage
    const size_t output_storage_size  = transform_output_kernel->get_output_storage_size(in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, out_channels) * data_type_size;
    const int    kernel_matrix_stride = transform_weights_kernel->get_matrix_stride(out_channels, in_channels);
    const int    output_matrix_stride = transform_output_kernel->get_matrix_stride(in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, out_channels);
    const auto   output_shape         = transform_output_kernel->get_output_shape(in_shape.n_rows, in_shape.n_cols, use_padding_type == PADDING_SAME);
    const int    input_matrix_stride  = transform_input_kernel->get_matrix_stride(in_shape.n_batches, in_channels, in_shape.n_rows, in_shape.n_cols, use_padding_type == PADDING_SAME);

    // Configure GEMM
    const int tile_rows                = iceildiv(output_shape.first, output_tile.height);
    const int tile_cols                = iceildiv(output_shape.second, output_tile.width);
    const int m                        = in_shape.n_batches * tile_rows * tile_cols;
    const int k                        = in_shape.n_channels;
    const int n                        = out_channels;
    const int kernel_matrix_row_stride = roundup(out_channels, N_BLOCK);
    const int output_matrix_row_stride = kernel_matrix_row_stride;

    TensorShape a_shape(k, m, 1, n_gemms);
    Strides     a_strides(data_type_size);
    a_strides.set(1, a_strides[0] * k);
    //a_strides.set(2, data_type_size * input_matrix_stride / n_gemms); FIXME: This is the real batch size, but RSH's code crashes if it's not 0.
    a_strides.set(2, 0);
    a_strides.set(3, data_type_size * input_matrix_stride);

    TensorShape b_shape(n, k, n_gemms);
    Strides     b_strides(data_type_size);
    b_strides.set(1, data_type_size * kernel_matrix_row_stride);
    b_strides.set(2, data_type_size * kernel_matrix_stride);

    TensorShape d_shape(n, m, 1, n_gemms);
    Strides     d_strides(data_type_size);
    d_strides.set(1, data_type_size * output_matrix_row_stride);
    //d_strides.set(2, data_type_size * output_matrix_stride / n_gemms); FIXME: This is the real batch size, but RSH's code crashes if it's not 0.
    d_strides.set(2, 0);
    d_strides.set(3, data_type_size * output_matrix_stride);

    TensorInfo a_info{};
    TensorInfo b_info{};
    TensorInfo d_info{};
    a_info.init(a_shape, 1, data_type, a_strides, 0, input_storage_size);
    b_info.init(b_shape, 1, data_type, b_strides, 0, kernel_storage_size);
    d_info.init(d_shape, 1, data_type, d_strides, 0, output_storage_size);

    _input_transformed  = a_info;
    _kernel_storage     = b_info;
    _output_transformed = d_info;

    // configure and allocate dst tensor to be used to convert from winograd domain to spatial domain when calling to reshape_output()
    TensorInfo info(TensorShape(dst->dimension(2), dst->dimension(0),
                                dst->dimension(1), dst->dimension(3)),
                    1, dst->data_type());
    _output_nhwc = info;

    const ITensorInfo *input_to_use  = src;
    ITensorInfo       *output_to_use = dst;
    PermutationVector  weights_permutation_vector(3U, 0U, 1U, 2U);
    const unsigned int max_num_threads = NEScheduler::get().num_threads();

    // Configure the kernel to transform the input tensor from NCHW -> NHWC
    if(_data_layout == DataLayout::NCHW)
    {
        _permute_input->configure(src, &_input_nhwc, PermutationVector(2U, 0U, 1U));
        _aux_mem[PermutedInput]    = MemoryInfo(offset_int_vec(PermutedInput), MemoryLifetime::Temporary, src->total_size());
        input_to_use               = &_input_nhwc;
        weights_permutation_vector = PermutationVector(3U, 2U, 0U, 1U);
    }

    // Configure input transform kernel
    transform_input_kernel->configure(input_to_use, in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, in_shape.n_channels, use_padding_type,
                                      &_input_transformed, input_matrix_stride, &_input_workspace);
    const size_t input_workspace_size = transform_input_kernel->get_working_space_size(max_num_threads);
    TensorInfo   input_workspace_info(TensorShape(input_workspace_size), 1, src->data_type());
    _input_workspace = input_workspace_info;

    // Re-order a weight tensor from [Output feature map x Input feature map x Height x Width] to [Height x Width x Input feature map x Output feature map]
    _permute_weights->configure(weights, &_weights_hwio, weights_permutation_vector);
    transform_weights_kernel->configure(&_weights_hwio, &_kernel_storage, kernel_matrix_stride, out_channels, in_channels);

    // Configure GEMM function
    _gemm_function->configure(&_input_transformed, &_kernel_storage, nullptr, &_output_transformed, 1.0f, 0.f);

    // Configure output transform function
    // The biases tensor has not been allocated at this point in time, the output transform will add the biases to the final result in the run() method
    if(_data_layout == DataLayout::NCHW)
    {
        output_to_use = &_output_nhwc;
    }
    const arm_gemm::Activation activation = arm_gemm_activation_from_acl_activation(act_info);

    transform_output_kernel->configure(biases,
                                       &_output_transformed,
                                       output_matrix_stride,
                                       output_to_use,
                                       in_shape.n_batches,
                                       output_shape.first,
                                       output_shape.second,
                                       out_channels,
                                       &_output_workspace,
                                       activation);

    const size_t output_workspace_size = transform_output_kernel->get_working_space_size(max_num_threads);
    TensorInfo   output_workspace_info(TensorShape(output_workspace_size), 1, dst->data_type());
    _output_workspace = output_workspace_info;

    // Reorder the convoluted output to ACL's ordering NCHW
    if(_data_layout == DataLayout::NCHW)
    {
        _permute_output->configure(&_output_nhwc, dst, PermutationVector(1U, 2U, 0U));
        _aux_mem[PermutedOutput] = MemoryInfo(offset_int_vec(PermutedOutput), MemoryLifetime::Temporary, dst->total_size());
    }

    _transform_input_kernel   = std::move(transform_input_kernel);
    _transform_weights_kernel = std::move(transform_weights_kernel);
    _transform_output_kernel  = std::move(transform_output_kernel);

    //Configure Activation Layer
    _run_activation = act_info.enabled() && !fuse_function_supported(act_info);
    if(_run_activation)
    {
        _activation_func->configure(dst, nullptr, act_info);
    }

    auto asm_mem_req         = _gemm_function->workspace();
    _aux_mem[GemmWorkspace]  = asm_mem_req[GemmWorkspace];
    _aux_mem[Pretranspose]   = asm_mem_req[Pretranspose];
    _aux_mem[InterleavedLHS] = asm_mem_req[InterleavedLHS];
    _aux_mem[TransposedRHS]  = asm_mem_req[TransposedRHS];
    _aux_mem[TempResult]     = asm_mem_req[TempResult];

    _aux_mem[InputTransformed] = MemoryInfo(offset_int_vec(InputTransformed), MemoryLifetime::Persistent, input_storage_size, storage_alignment);
    _aux_mem[InputWorkspace]   = MemoryInfo(offset_int_vec(InputWorkspace), MemoryLifetime::Persistent, input_workspace_size);
    if(_aux_mem[Pretranspose].size > 0)
    {
        // Release permuted weights at the of prepare as they are further transposed by the assembly dispatch
        _aux_mem[PermutedWeights] = MemoryInfo(offset_int_vec(PermutedWeights), MemoryLifetime::Prepare, _weights_hwio.total_size());
    }
    else
    {
        _aux_mem[PermutedWeights] = MemoryInfo(offset_int_vec(PermutedWeights), MemoryLifetime::Persistent, _weights_hwio.total_size());
    }
    _aux_mem[WeightsTransformed] = MemoryInfo(offset_int_vec(WeightsTransformed), MemoryLifetime::Persistent, kernel_storage_size, storage_alignment);
    _aux_mem[OutputTransformed]  = MemoryInfo(offset_int_vec(OutputTransformed), MemoryLifetime::Persistent, output_storage_size, storage_alignment);
    _aux_mem[OutputWorkspace]    = MemoryInfo(offset_int_vec(OutputWorkspace), MemoryLifetime::Persistent, output_workspace_size);
}

Status CpuWinogradConv2d::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                   const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, conv_info));

    // Get indices for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape, kernel size and output tile
    const Size2D   input_dims  = Size2D(input->dimension(idx_width), input->dimension(idx_height));
    const Size2D   kernel_size = Size2D(weights->dimension(idx_width), weights->dimension(idx_height));
    const DataType data_type   = input->data_type();
    const Size2D   output_tile = winograd_output_tile(input_dims, kernel_size, data_type);

    // Check if the Winograd configuration requires fast math
    if(!enable_fast_math)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(check_support_fast_math(output_tile, kernel_size, data_type),
                                        "This Winograd configuration requires enable_fast_math=true");
    }

    const WinogradInfo winograd_info = WinogradInfo(output_tile,
                                                    kernel_size,
                                                    input_dims,
                                                    conv_info,
                                                    input->data_layout());

    // Validate input transform
    const TensorShape input0_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*input, winograd_info);
    const TensorInfo  input0       = input->clone()->set_tensor_shape(input0_shape);
    // Validate filter transform
    const TensorShape input1_shape = misc::shape_calculator::compute_winograd_filter_transform_shape(*weights, winograd_info);
    const TensorInfo  input1       = weights->clone()->set_tensor_shape(input1_shape);
    // Validate batched matrix multiply
    TensorShape batched_mm_output_shape = input0.tensor_shape();
    batched_mm_output_shape[0]          = input1.tensor_shape()[0];
    const TensorInfo batched_mm_output  = input0.clone()->set_tensor_shape(batched_mm_output_shape);

    if(kernel_size == Size2D(3, 3))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != conv_info.pad_left(), "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != conv_info.pad_bottom(), "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != conv_info.pad_left(), "Only SAME or VALID padding supported");
        return validate_kernel_3x3(input_dims, input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(5, 5))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != conv_info.pad_left(), "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != conv_info.pad_bottom(), "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != conv_info.pad_left(), "Only SAME or VALID padding supported");
        return validate_kernel_5x5(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    if(kernel_size == Size2D(3, 1))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_bottom() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_3x1(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(1, 3))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_right() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_1x3(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(5, 1))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_bottom() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_5x1(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(1, 5))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_right() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_1x5(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(7, 1))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 3, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 3, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_bottom() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_7x1(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(1, 7))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 3, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 3, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_right() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_1x7(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Kernel shape not supported");
    }
}

void CpuWinogradConv2d::run(ITensorPack &tensors)
{
    prepare(tensors);

    auto a = tensors.get_const_tensor(ACL_SRC_0);
    auto c = tensors.get_const_tensor(ACL_SRC_2);
    auto d = tensors.get_tensor(ACL_DST);

    CpuAuxTensorHandler input_nhwc(offset_int_vec(PermutedInput), _input_nhwc, tensors, true);
    CpuAuxTensorHandler output_nhwc(offset_int_vec(PermutedOutput), _output_nhwc, tensors, true);
    CpuAuxTensorHandler input_transformed(offset_int_vec(InputTransformed), _input_transformed, tensors, true);
    CpuAuxTensorHandler input_workspace(offset_int_vec(InputWorkspace), _input_workspace, tensors, true);

    const bool is_nchw = _data_layout == DataLayout::NCHW;
    if(is_nchw)
    {
        //Bring channels to the front as Winograd code expects the tensor to be in the format NHWC
        ITensorPack pack{ { ACL_SRC, a }, { ACL_DST, input_nhwc.get() } };
        _permute_input->run(pack);
    }

    // Transform input tensor to the winograd domain
    ITensorPack transform_input_pack{ { ACL_SRC, is_nchw ? input_nhwc.get() : a }, { ACL_DST, input_transformed.get() }, { ACL_INT, input_workspace.get() } };
    NEScheduler::get().schedule_op(_transform_input_kernel.get(), Window::DimX, _transform_input_kernel->window(), transform_input_pack);

    CpuAuxTensorHandler output_transformed(offset_int_vec(OutputTransformed), _output_transformed, tensors, true);
    CpuAuxTensorHandler weights_transformed(offset_int_vec(WeightsTransformed), _kernel_storage, tensors, true);

    // Run 16 GEMMs in multiple threads, each kernel runs one or more GEMMs
    ITensorPack gemm_pack{ { ACL_SRC, input_transformed.get() }, { ACL_SRC_1, weights_transformed.get() }, { ACL_DST, output_transformed.get() } };
    _gemm_function->run(gemm_pack);

    // Transform output tensor to the spatial domain
    CpuAuxTensorHandler output_workspace(offset_int_vec(OutputWorkspace), _output_workspace, tensors, true);
    ITensorPack         transform_output_pack{ { ACL_SRC_0, c }, { ACL_SRC_1, output_transformed.get() }, { ACL_DST, is_nchw ? output_nhwc.get() : d }, { ACL_INT, output_workspace.get() } };
    NEScheduler::get().schedule_op(_transform_output_kernel.get(), Window::DimX, _transform_output_kernel->window(), transform_output_pack);

    if(is_nchw)
    {
        // Reorder the convoluted output to ACL's ordering NCHW
        ITensorPack pack{ { ACL_SRC, output_nhwc.get() }, { ACL_DST, d } };
        _permute_output->run(pack);
    }

    if(_run_activation)
    {
        ITensorPack pack{ { ACL_SRC, d }, { ACL_DST, d } };
        _activation_func->run(pack);
    }
}

void CpuWinogradConv2d::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        // Permute weights
        const ITensor *weights     = tensors.get_const_tensor(ACL_SRC_1);
        ITensor       *weights_aux = utils::cast::polymorphic_cast<ITensor *>(tensors.get_tensor(offset_int_vec(PermutedWeights)));
        ARM_COMPUTE_ERROR_ON_NULLPTR(weights, weights_aux);

        CpuAuxTensorHandler permuted_weights(_weights_hwio, *weights_aux);
        ITensorPack         permute_tensors{ { ACL_SRC, weights }, { ACL_DST, permuted_weights.get() } };
        _permute_weights->run(permute_tensors);

        // Transform weights
        ITensor *weights_transf = utils::cast::polymorphic_cast<ITensor *>(tensors.get_tensor(offset_int_vec(WeightsTransformed)));
        ARM_COMPUTE_ERROR_ON_NULLPTR(weights_transf);

        CpuAuxTensorHandler transformed_weights(_kernel_storage, *weights_transf);
        ITensorPack         transform_tensors{ { ACL_SRC, permuted_weights.get() }, { ACL_DST, transformed_weights.get() } };
        NEScheduler::get().schedule_op(_transform_weights_kernel.get(), Window::DimX, _transform_weights_kernel->window(), transform_tensors);

        CpuAuxTensorHandler input_transformed(offset_int_vec(InputTransformed), _input_transformed, tensors, true);
        CpuAuxTensorHandler output_transformed(offset_int_vec(OutputTransformed), _output_transformed, tensors, true);
        ITensorPack         gemm_pack = tensors;
        gemm_pack.add_const_tensor(ACL_SRC_0, input_transformed.get());
        gemm_pack.add_const_tensor(ACL_SRC_1, transformed_weights.get());
        _gemm_function->prepare(gemm_pack);

        _is_prepared = true;
    }
}

experimental::MemoryRequirements CpuWinogradConv2d::workspace() const
{
    return _aux_mem;
}
} // namespace cpu
} // namespace arm_compute