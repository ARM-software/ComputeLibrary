/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_POOLING_3D_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_POOLING_3D_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Pooling3dLayer.h"
#include <random>
namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class Pooling3dLayerValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, Pooling3dLayerInfo pool_info, DataType data_type)
    {
        _target    = compute_target(shape, pool_info, data_type);
        _reference = compute_reference(shape, pool_info, data_type);
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
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -1.0f, 1.0f };
            library->fill(tensor, distribution, 0);
        }
        else // data type is quantized_asymmetric
        {
            ARM_COMPUTE_ERROR("Passed Type Not Supported");
        }
    }

    TensorType compute_target(TensorShape shape, Pooling3dLayerInfo info,
                              DataType data_type)
    {
        // Create tensors
        TensorType        src       = create_tensor<TensorType>(shape, data_type, 1, QuantizationInfo(), DataLayout::NDHWC);
        const TensorShape dst_shape = misc::shape_calculator::compute_pool3d_shape((src.info()->tensor_shape()), info);
        TensorType        dst       = create_tensor<TensorType>(dst_shape, data_type, 1, QuantizationInfo(), DataLayout::NDHWC);

        // Create and configure function
        FunctionType pool_layer;
        pool_layer.validate(src.info(), dst.info(), info);
        pool_layer.configure(&src, &dst, info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        pool_layer.run();
        return dst;
    }

    SimpleTensor<T> compute_reference(TensorShape shape, Pooling3dLayerInfo info, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src(shape, data_type, 1, QuantizationInfo(), DataLayout::NDHWC);
        // Fill reference
        fill(src);
        return reference::pooling_3d_layer<T>(src, info);
    }

    TensorType             _target{};
    SimpleTensor<T>        _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class Pooling3dLayerValidationFixture : public Pooling3dLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, Size3D pool_size, Size3D stride, Padding3D padding, bool exclude_padding, DataType data_type)
    {
        Pooling3dLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, Pooling3dLayerInfo(pool_type, pool_size, stride, padding, exclude_padding),
                                                                                                 data_type);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class Pooling3dLayerGlobalValidationFixture : public Pooling3dLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, DataType data_type)
    {
        Pooling3dLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, Pooling3dLayerInfo(pool_type), data_type);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SpecialPooling3dLayerValidationFixture : public Pooling3dLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape src_shape, Pooling3dLayerInfo pool_info, DataType data_type)
    {
        Pooling3dLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(src_shape, pool_info, data_type);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_POOLING_3D_LAYER_FIXTURE */
