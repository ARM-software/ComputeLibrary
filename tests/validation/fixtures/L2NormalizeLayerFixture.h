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
#ifndef ARM_COMPUTE_TEST_L2NORMALIZE_FIXTURE
#define ARM_COMPUTE_TEST_L2NORMALIZE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/L2NormalizeLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr int max_input_tensor_dim = 3;
} // namespace
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class L2NormalizeLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, DataLayout data_layout, int axis, float epsilon)
    {
        _target    = compute_target(shape, data_type, data_layout, axis, epsilon);
        _reference = compute_reference(shape, data_type, data_layout, axis, epsilon);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        std::uniform_real_distribution<> distribution(1.f, 2.f);
        library->fill(tensor, distribution, 0);
    }

    TensorType compute_target(TensorShape shape, DataType data_type, DataLayout data_layout, int axis, float epsilon)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType dst = create_tensor<TensorType>(shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType l2_norm_func;
        l2_norm_func.configure(&src, &dst, axis, epsilon);

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
        l2_norm_func.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type, DataLayout data_layout, int axis, float epsilon)
    {
        uint32_t actual_axis = wrap_around(axis, max_input_tensor_dim);
        if(data_layout == DataLayout::NHWC)
        {
            switch(actual_axis)
            {
                case 0:
                    actual_axis = 2;
                    break;
                case 1:
                    actual_axis = 0;
                    break;
                case 2:
                    actual_axis = 1;
                    break;
                default:
                    break;
            }
        }
        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src);

        return reference::l2_normalize<T>(src, actual_axis, epsilon);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_L2NORMALIZE_FIXTURE */
