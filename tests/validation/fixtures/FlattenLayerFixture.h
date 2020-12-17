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
#ifndef ARM_COMPUTE_TEST_FLATTEN_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_FLATTEN_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/FlattenLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FlattenLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        TensorShape shape_flatten;
        TensorInfo  input_info(shape, 1, data_type);
        shape_flatten = compute_flatten_shape(&input_info);

        _target    = compute_target(shape, shape_flatten, data_type);
        _reference = compute_reference(shape, shape_flatten, data_type);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(_target.info()->tensor_shape(), _reference.shape());
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_fp16, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, 0);
    }

    TensorType compute_target(const TensorShape &shape, const TensorShape &shape_flatten, DataType data_type)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type, 1);
        TensorType dst = create_tensor<TensorType>(shape_flatten, data_type, 1);

        // Create and configure function
        FunctionType flatten_layer;
        flatten_layer.configure(&src, &dst);

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
        flatten_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, const TensorShape &shape_flatten, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 1 };

        // Fill reference
        fill(src);

        return reference::flatten_layer<T>(src, shape_flatten);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FLATTEN_LAYER_FIXTURE */
