/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_REMAP_FIXTURE
#define ARM_COMPUTE_TEST_REMAP_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Remap.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RemapValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, InterpolationPolicy policy, DataType data_type, BorderMode border_mode, DataLayout data_layout = DataLayout::NCHW)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> distribution(0, 255);
        PixelValue                             constant_border_value{ static_cast<T>(distribution(gen)) };

        _data_layout = data_layout;
        _target      = compute_target(shape, policy, data_type, border_mode, constant_border_value);
        _reference   = compute_reference(shape, policy, data_type, border_mode, constant_border_value);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, int min, int max)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                // map_x,y as integer values
                std::uniform_int_distribution<int> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution(static_cast<float>(min), static_cast<float>(max));
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::U8:
            {
                std::uniform_int_distribution<uint8_t> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                ARM_COMPUTE_ERROR("DataType for Remap not supported");
        }
    }

    TensorType compute_target(TensorShape shape, InterpolationPolicy policy, DataType data_type, BorderMode border_mode, PixelValue constant_border_value)
    {
        if(_data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src   = create_tensor<TensorType>(shape, data_type, 1, QuantizationInfo(), _data_layout);
        TensorType map_x = create_tensor<TensorType>(shape, DataType::F32, 1, QuantizationInfo(), _data_layout);
        TensorType map_y = create_tensor<TensorType>(shape, DataType::F32, 1, QuantizationInfo(), _data_layout);
        TensorType dst   = create_tensor<TensorType>(shape, data_type, 1, QuantizationInfo(), _data_layout);

        // Create and configure function
        FunctionType remap;
        remap.configure(&src, &map_x, &map_y, &dst, policy, border_mode, constant_border_value);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(map_x.info()->is_resizable());
        ARM_COMPUTE_ASSERT(map_y.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        map_x.allocator()->allocate();
        map_y.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!map_x.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!map_y.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        int max_val = std::max({ shape.x(), shape.y(), shape.z() });

        fill(AccessorType(src), 0, 0, 255);
        fill(AccessorType(map_x), 1, -5, max_val);
        fill(AccessorType(map_y), 2, -5, max_val);

        // Compute function
        remap.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape shape, InterpolationPolicy policy, DataType data_type, BorderMode border_mode, PixelValue constant_border_value)
    {
        ARM_COMPUTE_ERROR_ON(data_type != DataType::U8 && data_type != DataType::F16);

        // Create reference
        SimpleTensor<T>     src{ shape, data_type };
        SimpleTensor<float> map_x{ shape, DataType::F32 };
        SimpleTensor<float> map_y{ shape, DataType::F32 };
        T                   border_value{};
        constant_border_value.get(border_value);

        // Create the valid mask Tensor
        _valid_mask = SimpleTensor<T> { shape, data_type };

        // Fill reference
        int max_val = std::max({ shape.x(), shape.y(), shape.z() });

        fill(src, 0, 0, 255);
        fill(map_x, 1, -5, max_val);
        fill(map_y, 2, -5, max_val);

        // Compute reference
        return reference::remap<T>(src, map_x, map_y, _valid_mask, policy, border_mode, border_value);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    SimpleTensor<T> _valid_mask{};
    DataLayout      _data_layout{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RemapValidationFixture : public RemapValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, InterpolationPolicy policy, DataType data_type, BorderMode border_mode)
    {
        RemapValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, policy, data_type, border_mode);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RemapValidationMixedLayoutFixture : public RemapValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, InterpolationPolicy policy, DataType data_type, BorderMode border_mode, DataLayout data_layout)
    {
        RemapValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, policy, data_type, border_mode, data_layout);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_REMAP_FIXTURE */
