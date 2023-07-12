/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_WINOGRAD_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_WINOGRAD_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/Utils.h"
#include "tests/validation/reference/Winograd.h"
#include "utils/Utils.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename T1 = T, bool use_bias = true, bool mixed_layout = false>
class WinogradConvolutionLayerFastMathValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation,
               DataType data_type, ActivationLayerInfo act_info, const DataLayout &data_layout)

    {
        ARM_COMPUTE_UNUSED(dilation);
        _mixed_layout = mixed_layout;
        _target       = compute_target(input_shape, weights_shape, bias_shape, output_shape, info, data_type, act_info, data_layout);
        _reference    = compute_reference(input_shape, weights_shape, bias_shape, info, data_type, act_info);
    }

protected:
    void mix_layout(FunctionType &layer, TensorType &src, TensorType &dst)
    {
        const DataLayout data_layout = src.info()->data_layout();
        // Test Multi DataLayout graph cases, when the data layout changes after configure
        src.info()->set_data_layout(data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
        dst.info()->set_data_layout(data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);

        // Compute Convolution function
        layer.run();

        // Reinstating original data layout for the test suite to properly check the values
        src.info()->set_data_layout(data_layout);
        dst.info()->set_data_layout(data_layout);
    }

    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ float(min), float(max) };
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
            }
        }
    }

    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, const PadStrideInfo &info,
                              DataType data_type, ActivationLayerInfo act_info, const DataLayout data_layout)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType weights = create_tensor<TensorType>(weights_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType bias    = create_tensor<TensorType>(bias_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType dst     = create_tensor<TensorType>(output_shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType conv;
        ARM_COMPUTE_EXPECT(static_cast<bool>(conv.validate(src.info(), weights.info(), (use_bias) ? bias.info() : nullptr, dst.info(), info, act_info, true /* Enable fast math */)),
                           framework::LogLevel::ERRORS);
        conv.configure(&src, &weights, (use_bias) ? &bias : nullptr, &dst, info, act_info, true /* Enable fast math */);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        add_padding_x({ &src, &weights, &bias, &dst }, data_layout);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        dst.allocator()->allocate();
        bias.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0, -0.5f, 0.5f);
        fill(AccessorType(weights), 1, -0.5f, 0.5f);
        fill(AccessorType(bias), 2, -0.5f, 0.5f);

        if(_mixed_layout)
        {
            mix_layout(conv, src, dst);
        }
        else
        {
            // Compute function
            conv.run();
        }
        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const PadStrideInfo &info,
                                      DataType data_type, ActivationLayerInfo act_info)
    {
        // Create reference
        SimpleTensor<T> src_t{ input_shape, data_type, 1 };
        SimpleTensor<T> weights_t{ weights_shape, data_type, 1 };
        SimpleTensor<T> bias_t{ bias_shape, data_type, 1 };

        // Fill reference
        fill(src_t, 0, -0.5f, 0.5f);
        SimpleTensor<T1> src_t1(copy_tensor<T1, T>(src_t));

        fill(weights_t, 1, -0.5f, 0.5f);
        SimpleTensor<T1> weights_t1(copy_tensor<T1, T>(weights_t));
        if(use_bias)
        {
            fill(bias_t, 2, -0.5f, 0.5f);
        }
        else
        {
            fill(bias_t, 2, 0.f, 0.f);
        }
        SimpleTensor<T1> bias_t1(copy_tensor<T1, T>(bias_t));

        // Set output tile
        Size2D output_tile(4U, 4U);
        if(weights_shape[0] == 7 && weights_shape[1] == 1)
        {
            output_tile.width  = 2;
            output_tile.height = 1;
        }
        else if(weights_shape[0] == 1 && weights_shape[1] == 7)
        {
            output_tile.width  = 1;
            output_tile.height = 2;
        }
        else if(weights_shape[0] == 1)
        {
            output_tile.width = 1;
        }
        else if(weights_shape[1] == 1)
        {
            output_tile.height = 1;
        }

        WinogradInfo winograd_info(output_tile,
                                   Size2D(weights_shape[0], weights_shape[1]),
                                   Size2D(input_shape[0], input_shape[1]),
                                   info,
                                   src_t1.data_layout());

        // Compute tensor shapes for input, filter and output transforms
        TensorShape input_transform_shape  = compute_winograd_input_transform_shape(TensorInfo(input_shape, 1, data_type), winograd_info);
        TensorShape filter_transform_shape = compute_winograd_filter_transform_shape(TensorInfo(weights_shape, 1, data_type), winograd_info);
        TensorShape batched_gemm_shape     = input_transform_shape;
        batched_gemm_shape[0]              = filter_transform_shape[0];
        TensorShape output_transform_shape = compute_winograd_output_transform_shape(TensorInfo(batched_gemm_shape, 1, data_type), winograd_info);

        // Dummy matrix C to perform matrix multiplication
        SimpleTensor<T1> dummy_c{ batched_gemm_shape, data_type, 1 };

        // Compute Winograd-based convolution
        SimpleTensor<T1> input_transform_out = reference::winograd_input_transform<T1>(src_t1, input_transform_shape, winograd_info);

        SimpleTensor<T1> filter_transform_out = reference::winograd_filter_transform<T1>(weights_t1, filter_transform_shape, winograd_info);
        SimpleTensor<T1> batched_gemm         = reference::gemm<T1>(input_transform_out, filter_transform_out, dummy_c, 1.0f, 0.0f);
        SimpleTensor<T1> conv_out             = reference::winograd_output_transform<T1>(batched_gemm, bias_t1, output_transform_shape, winograd_info);
        SimpleTensor<T>  conv_out_t(std::move(copy_tensor<T, T1>(conv_out)));
        return (act_info.enabled()) ? reference::activation_layer<T>(conv_out_t, act_info) : conv_out_t;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    bool            _mixed_layout{ false };
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class WinogradInputTransformValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape input_shape, WinogradInfo winograd_info, DataLayout data_layout, DataType data_type)
    {
        TensorShape output_shape = compute_winograd_input_transform_shape(TensorInfo(input_shape, 1, data_type), winograd_info);
        _mixed_layout            = mixed_layout;
        _target                  = compute_target(input_shape, output_shape, winograd_info, data_layout, data_type);
        _reference               = compute_reference(input_shape, output_shape, winograd_info, data_type);
    }

protected:
    void mix_layout(FunctionType &layer, TensorType &src, TensorType &dst)
    {
        const DataLayout data_layout_src = src.info()->data_layout();
        const DataLayout data_layout_dst = dst.info()->data_layout();

        // Test Multi DataLayout graph cases, when the data layout changes after configure
        src.info()->set_data_layout(data_layout_src == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
        dst.info()->set_data_layout(data_layout_dst == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);

        // Compute Convolution function
        layer.run();

        // Reinstating original data layout for the test suite to properly check the values
        src.info()->set_data_layout(data_layout_src);
        dst.info()->set_data_layout(data_layout_dst);
    }

    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ float(min), float(max) };
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
            }
        }
    }

    TensorType compute_target(TensorShape input_shape, const TensorShape &output_shape, const WinogradInfo &winograd_info, DataLayout data_layout, DataType data_type)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
        }

        TensorType src = create_tensor<TensorType>(input_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1, QuantizationInfo());

        // Create and configure function
        FunctionType transf;
        transf.configure(&src, &dst, winograd_info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        add_padding_x({ &src, &dst }, data_layout);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0, -1.f, 1.f);

        if(_mixed_layout)
        {
            mix_layout(transf, src, dst);
        }
        else
        {
            // Compute Winograd input transform function
            transf.run();
        }
        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &output_shape, const WinogradInfo &winograd_info, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type, 1, QuantizationInfo() };

        // Fill reference
        fill(src, 0, -1.f, 1.f);

        return reference::winograd_input_transform<T>(src, output_shape, winograd_info);
    }

    bool            _mixed_layout{ false };
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class WinogradFilterTransformValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape input_shape, Size2D output_tile, DataLayout data_layout, DataType data_type)
    {
        WinogradInfo winograd_info(output_tile, Size2D(input_shape[0], input_shape[1]), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, DataLayout::NCHW /* Not needed */);
        TensorShape  output_shape = compute_winograd_filter_transform_shape(TensorInfo(input_shape, 1, data_type), winograd_info);

        _mixed_layout = mixed_layout;
        _target       = compute_target(input_shape, output_shape, winograd_info, data_layout, data_type);
        _reference    = compute_reference(input_shape, output_shape, winograd_info, data_type);
    }

protected:
    void mix_layout(FunctionType &layer, TensorType &src, TensorType &dst)
    {
        const DataLayout data_layout_src = src.info()->data_layout();
        const DataLayout data_layout_dst = dst.info()->data_layout();

        // Test Multi DataLayout graph cases, when the data layout changes after configure
        src.info()->set_data_layout(data_layout_src == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
        dst.info()->set_data_layout(data_layout_dst == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);

        // Compute Convolution function
        layer.run();

        // Reinstating original data layout for the test suite to properly check the values
        src.info()->set_data_layout(data_layout_src);
        dst.info()->set_data_layout(data_layout_dst);
    }

    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ float(min), float(max) };
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
            }
        }
    }

    TensorType compute_target(TensorShape input_shape, const TensorShape &output_shape, const WinogradInfo &winograd_info, DataLayout data_layout, DataType data_type)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1, QuantizationInfo());

        // Create and configure function
        FunctionType filter_transform;
        filter_transform.configure(&src, &dst, winograd_info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        add_padding_x({ &src, &dst }, data_layout);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0, -1.f, 1.f);

        if(_mixed_layout)
        {
            mix_layout(filter_transform, src, dst);
        }
        else
        {
            // Compute Winograd filter transform function
            filter_transform.run();
        }
        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &output_shape, const WinogradInfo &winograd_info, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type, 1, QuantizationInfo() };

        // Fill reference
        fill(src, 0, -1.f, 1.f);

        return reference::winograd_filter_transform<T>(src, output_shape, winograd_info);
    }

    bool            _mixed_layout{ false };
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class WinogradOutputTransformValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape input_shape, WinogradInfo winograd_info, DataType data_type, ActivationLayerInfo act_info = ActivationLayerInfo())
    {
        _target    = compute_target(input_shape, winograd_info, data_type, act_info);
        _reference = compute_reference(input_shape, winograd_info, data_type, act_info);
    }

protected:
    void mix_layout(FunctionType &layer, TensorType &src, TensorType &dst)
    {
        const DataLayout data_layout_src = src.info()->data_layout();
        const DataLayout data_layout_dst = dst.info()->data_layout();

        // Test Multi DataLayout graph cases, when the data layout changes after configure
        src.info()->set_data_layout(data_layout_src == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
        dst.info()->set_data_layout(data_layout_dst == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);

        // Compute Convolution function
        layer.run();

        // Reinstating original data layout for the test suite to properly check the values
        src.info()->set_data_layout(data_layout_src);
        dst.info()->set_data_layout(data_layout_dst);
    }

    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ float(min), float(max) };
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
            }
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const WinogradInfo &winograd_info, DataType data_type, ActivationLayerInfo act_info)
    {
        TensorShape output_shape = compute_winograd_output_transform_shape(TensorInfo(input_shape, 1, data_type), winograd_info);

        // Create tensors
        TensorType src  = create_tensor<TensorType>(input_shape, data_type);
        TensorType bias = create_tensor<TensorType>(output_shape[get_data_layout_dimension_index(winograd_info.output_data_layout, DataLayoutDimension::CHANNEL)], data_type);
        TensorType dst  = create_tensor<TensorType>(output_shape, data_type, 1, QuantizationInfo(), winograd_info.output_data_layout);

        // Create and configure function
        FunctionType output_transform;
        output_transform.configure(&src, &bias, &dst, winograd_info, act_info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        add_padding_x({ &src, &bias, &dst }, winograd_info.output_data_layout);

        // Allocate tensors
        src.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0, -1.f, 1.f);
        fill(AccessorType(bias), 1, -1.f, 1.f);

        if(_mixed_layout)
        {
            mix_layout(output_transform, src, dst);
        }
        else
        {
            // Compute Winograd output transform function
            output_transform.run();
        }
        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, WinogradInfo winograd_info, DataType data_type, ActivationLayerInfo act_info)
    {
        winograd_info.output_data_layout = DataLayout::NCHW;
        TensorShape output_shape         = compute_winograd_output_transform_shape(TensorInfo(input_shape, 1, data_type), winograd_info);

        // Create reference
        SimpleTensor<T> src{ input_shape, data_type };
        SimpleTensor<T> bias{ TensorShape(input_shape[0]), data_type };

        // Fill reference
        fill(src, 0, -1.f, 1.f);
        fill(bias, 1, -1.f, 1.f);

        const SimpleTensor<T> winograd_output = reference::winograd_output_transform<T>(src, bias, output_shape, winograd_info);

        return (act_info.enabled()) ? reference::activation_layer<T>(winograd_output, act_info) : winograd_output;
    }

    bool            _mixed_layout{ false };
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WINOGRAD_LAYER_FIXTURE */