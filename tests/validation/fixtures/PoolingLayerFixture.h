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
    void setup(TensorShape shape, PoolingLayerInfo pool_info, DataType data_type, DataLayout data_layout, bool indices = false)
    {
        std::mt19937                    gen(library->seed());
        std::uniform_int_distribution<> offset_dis(0, 20);
        const float                     scale     = data_type == DataType::QASYMM8_SIGNED ? 1.f / 127.f : 1.f / 255.f;
        const int                       scale_in  = data_type == DataType::QASYMM8_SIGNED ? -offset_dis(gen) : offset_dis(gen);
        const int                       scale_out = data_type == DataType::QASYMM8_SIGNED ? -offset_dis(gen) : offset_dis(gen);
        const QuantizationInfo          input_qinfo(scale, scale_in);
        const QuantizationInfo          output_qinfo(scale, scale_out);

        _pool_info = pool_info;
        _target    = compute_target(shape, pool_info, data_type, data_layout, input_qinfo, output_qinfo, indices);
        _reference = compute_reference(shape, pool_info, data_type, data_layout, input_qinfo, output_qinfo, indices);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if(tensor.data_type() == DataType::F32)
        {
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, 0);
        }
        else if(tensor.data_type() == DataType::F16)
        {
            arm_compute::utils::uniform_real_distribution_fp16 distribution{ half(-1.0f), half(1.0f) };
            library->fill(tensor, distribution, 0);
        }
        else // data type is quantized_asymmetric
        {
            library->fill_tensor_uniform(tensor, 0);
        }
    }

    TensorType compute_target(TensorShape shape, PoolingLayerInfo info,
                              DataType data_type, DataLayout data_layout,
                              QuantizationInfo input_qinfo, QuantizationInfo output_qinfo,
                              bool indices)
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
        _target_indices             = create_tensor<TensorType>(dst_shape, DataType::U32, 1, output_qinfo, data_layout);

        // Create and configure function
        FunctionType pool_layer;
        pool_layer.configure(&src, &dst, info, (indices) ? &_target_indices : nullptr);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_target_indices.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        _target_indices.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!_target_indices.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        pool_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(TensorShape shape, PoolingLayerInfo info, DataType data_type, DataLayout data_layout,
                                      QuantizationInfo input_qinfo, QuantizationInfo output_qinfo, bool indices)
    {
        // Create reference
        SimpleTensor<T> src(shape, data_type, 1, input_qinfo);
        // Fill reference
        fill(src);
        return reference::pooling_layer<T>(src, info, output_qinfo, indices ? &_ref_indices : nullptr, data_layout);
    }

    TensorType             _target{};
    SimpleTensor<T>        _reference{};
    PoolingLayerInfo       _pool_info{};
    TensorType             _target_indices{};
    SimpleTensor<uint32_t> _ref_indices{};
};
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerIndicesValidationFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, Size2D pool_size, PadStrideInfo pad_stride_info, bool exclude_padding, DataType data_type, DataLayout data_layout)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, pool_size, data_layout, pad_stride_info, exclude_padding),
                                                                                               data_type, data_layout, true);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerValidationFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, Size2D pool_size, PadStrideInfo pad_stride_info, bool exclude_padding, DataType data_type, DataLayout data_layout)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, pool_size, data_layout, pad_stride_info, exclude_padding),
                                                                                               data_type, data_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerValidationMixedPrecisionFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, Size2D pool_size, PadStrideInfo pad_stride_info, bool exclude_padding, DataType data_type, DataLayout data_layout, bool fp_mixed_precision = false)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, pool_size, data_layout, pad_stride_info, exclude_padding, fp_mixed_precision),
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
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, pool_size, data_layout, pad_stride_info, exclude_padding),
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
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, data_layout), data_type, data_layout);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_POOLING_LAYER_FIXTURE */
