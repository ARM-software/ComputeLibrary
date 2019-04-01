/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_POOLING_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_POOLING_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/PoolingLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingLayerInfo pool_info, DataType data_type, DataLayout data_layout)
    {
        std::mt19937                    gen(library->seed());
        std::uniform_int_distribution<> offset_dis(0, 20);
        const QuantizationInfo          input_qinfo(1.f / 255.f, offset_dis(gen));
        const QuantizationInfo          output_qinfo(1.f / 255.f, offset_dis(gen));

        _pool_info = pool_info;
        _target    = compute_target(shape, pool_info, data_type, data_layout, input_qinfo, output_qinfo);
        _reference = compute_reference(shape, pool_info, data_type, input_qinfo, output_qinfo);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if(!is_data_type_quantized(tensor.data_type()))
        {
            std::uniform_real_distribution<> distribution(-1.f, 1.f);
            library->fill(tensor, distribution, 0);
        }
        else // data type is quantized_asymmetric
        {
            library->fill_tensor_uniform(tensor, 0);
        }
    }

    TensorType compute_target(TensorShape shape, PoolingLayerInfo info,
                              DataType data_type, DataLayout data_layout, QuantizationInfo input_qinfo, QuantizationInfo output_qinfo)
    {
        // Change shape in case of NHWC.
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType        src       = create_tensor<TensorType>(shape, data_type, 1, input_qinfo, data_layout);
        const TensorShape dst_shape = misc::shape_calculator::compute_pool_shape(*(src.info()), info);
        TensorType        dst       = create_tensor<TensorType>(dst_shape, data_type, 1, output_qinfo, data_layout);

        // Create and configure function
        FunctionType pool_layer;
        pool_layer.configure(&src, &dst, info);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        pool_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, PoolingLayerInfo info, DataType data_type, QuantizationInfo input_qinfo, QuantizationInfo output_qinfo)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 1, input_qinfo };

        // Fill reference
        fill(src);

        return reference::pooling_layer<T>(src, info, output_qinfo);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    PoolingLayerInfo _pool_info{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerValidationFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, Size2D pool_size, PadStrideInfo pad_stride_info, bool exclude_padding, DataType data_type, DataLayout data_layout)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, pool_size, pad_stride_info, exclude_padding),
                                                                                               data_type, data_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerValidationQuantizedFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, Size2D pool_size, PadStrideInfo pad_stride_info, bool exclude_padding, DataType data_type, DataLayout data_layout = DataLayout::NCHW)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, pool_size, pad_stride_info, exclude_padding),
                                                                                               data_type, data_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SpecialPoolingLayerValidationFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape src_shape, PoolingLayerInfo pool_info, DataType data_type)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(src_shape, pool_info, data_type, DataLayout::NCHW);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class GlobalPoolingLayerValidationFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, DataType data_type, DataLayout data_layout = DataLayout::NCHW)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type), data_type, DataLayout::NCHW);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_POOLING_LAYER_FIXTURE */
