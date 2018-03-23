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
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/Utils.h"
#include "tests/validation/reference/Winograd.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class WinogradConvolutionLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation, DataType data_type)
    {
        ARM_COMPUTE_UNUSED(dilation);

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info, data_type);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
                library->fill_tensor_uniform(tensor, i);
                break;
            }
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const PadStrideInfo &info,
                              DataType data_type)
    {
        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, data_type, 1);
        TensorType weights = create_tensor<TensorType>(weights_shape, data_type, 1);
        TensorType bias    = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType dst     = create_tensor<TensorType>(output_shape, data_type, 1);

        // Create and configure function
        FunctionType conv;
        conv.configure(&src, &weights, &bias, &dst, info);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        dst.allocator()->allocate();
        bias.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!weights.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0, -1.f, 1.f);
        fill(AccessorType(weights), 1, -1.f, 1.f);
        fill(AccessorType(bias), 2, -1.f, 1.f);

        // Compute Winograd Convolution function
        conv.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const PadStrideInfo &info,
                                      DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type, 1 };
        SimpleTensor<T> weights{ weights_shape, data_type, 1 };
        SimpleTensor<T> bias{ bias_shape, data_type, 1 };

        // Fill reference
        fill(src, 0, -1.f, 1.f);
        fill(weights, 1, -1.f, 1.f);
        fill(bias, 2, -1.f, 1.f);

        return reference::convolution_layer<T>(src, weights, bias, output_shape, info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class WinogradInputTransformValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, PadStrideInfo conv_info, Size2D kernel_dims, bool is_nchw_format, DataType data_type)
    {
        TensorShape output_shape = compute_winograd_input_transform_shape(TensorInfo(input_shape, 1, data_type), conv_info, kernel_dims);

        _target    = compute_target(input_shape, output_shape, conv_info, kernel_dims, is_nchw_format, data_type);
        _reference = compute_reference(input_shape, output_shape, conv_info, kernel_dims, is_nchw_format, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
                library->fill_tensor_uniform(tensor, i);
                break;
            }
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &output_shape, const PadStrideInfo &conv_info, const Size2D &kernel_dims, bool is_nchw_format, DataType data_type)
    {
        ARM_COMPUTE_UNUSED(is_nchw_format);

        TensorType src = create_tensor<TensorType>(input_shape, data_type);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type);

        // Create and configure function
        FunctionType transf;
        transf.configure(&src, &dst, conv_info, kernel_dims);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0, -1.f, 1.f);

        // Compute CLWinogradInputTransform function
        transf.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &output_shape, const PadStrideInfo &conv_info, const Size2D &kernel_dims, bool is_nchw_format, DataType data_type)
    {
        ARM_COMPUTE_UNUSED(is_nchw_format);

        // Create reference
        SimpleTensor<T> src{ input_shape, data_type };

        // Fill reference
        fill(src, 0, -1.f, 1.f);

        return reference::winograd_input_transform<T>(src, output_shape, conv_info, kernel_dims);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class WinogradFilterTransformValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, bool is_nchw_format, Size2D output_tile, DataType data_type)
    {
        TensorShape output_shape = compute_winograd_filter_transform_shape(TensorInfo(input_shape, 1, data_type), output_tile);

        _target    = compute_target(input_shape, output_shape, is_nchw_format, output_tile, data_type);
        _reference = compute_reference(input_shape, output_shape, is_nchw_format, output_tile, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
                library->fill_tensor_uniform(tensor, i);
                break;
            }
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &output_shape, bool is_nchw_format, const Size2D &output_tile, DataType data_type)
    {
        ARM_COMPUTE_UNUSED(is_nchw_format);

        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, data_type, 1);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1);

        // Create and configure function
        FunctionType filter_transform;
        filter_transform.configure(&src, &dst, output_tile);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0, -1.f, 1.f);

        filter_transform.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &output_shape, bool is_nchw_format, const Size2D &output_tile, DataType data_type)
    {
        ARM_COMPUTE_UNUSED(is_nchw_format);

        // Create reference
        SimpleTensor<T> src{ input_shape, data_type, 1 };

        // Fill reference
        fill(src, 0, -1.f, 1.f);

        return reference::winograd_filter_transform<T>(src, output_shape, output_tile);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class WinogradOutputTransformValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, Size2D kernel_dims, Size2D output_convolved_dims, Size2D num_tiles, DataLayout data_layout, DataType data_type)
    {
        TensorShape output_shape = compute_winograd_output_transform_shape(TensorInfo(input_shape, 1, data_type), output_convolved_dims, data_layout);

        _target    = compute_target(input_shape, output_shape, kernel_dims, output_convolved_dims, num_tiles, data_layout, data_type);
        _reference = compute_reference(input_shape, output_shape, kernel_dims, output_convolved_dims, num_tiles, data_layout, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
                library->fill_tensor_uniform(tensor, i);
                break;
            }
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &output_shape, const Size2D &kernel_dims, const Size2D &output_convolved_dims, Size2D &num_tiles, DataLayout data_layout,
                              DataType data_type)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, data_type, 1, 0, QuantizationInfo(), data_layout);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1, 0, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType output_transform;
        output_transform.configure(&src, nullptr, &dst, kernel_dims, output_convolved_dims, num_tiles);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0, -1.f, 1.f);

        output_transform.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &output_shape, const Size2D &kernel_dims, const Size2D &output_convolved_dims, Size2D &num_tiles,
                                      DataLayout data_layout,
                                      DataType   data_type)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type, 1, 0, QuantizationInfo(), data_layout };

        // Fill reference
        fill(src, 0, -1.f, 1.f);

        return reference::winograd_output_transform<T>(src, output_shape, kernel_dims, num_tiles);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WINOGRAD_LAYER_FIXTURE */
