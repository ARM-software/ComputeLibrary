/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_MEAN_STDDEV_NORMALIZATION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_MEAN_STDDEV_NORMALIZATION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/MeanStdDevNormalizationLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class MeanStdDevNormalizationLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType dt, bool in_place, float epsilon = 1e-8f)
    {
        _data_type = dt;
        _target    = compute_target(shape, dt, in_place, epsilon);
        _reference = compute_reference(shape, dt, epsilon);
    }

protected:
    template <typename U>
    void fill(U &&src_tensor)
    {
        const float                      min_bound = -1.f;
        const float                      max_bound = 1.f;
        std::uniform_real_distribution<> distribution(min_bound, max_bound);
        library->fill(src_tensor, distribution, 0);
    }

    TensorType compute_target(TensorShape shape, DataType dt, bool in_place, float epsilon)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, dt, 1);
        TensorType dst;

        TensorType *dst_ptr = in_place ? &src : &dst;

        // Create and configure function
        FunctionType norm;
        norm.configure(&src, dst_ptr, epsilon);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);

        if(!in_place)
        {
            dst.allocator()->allocate();
            ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        norm.run();

        if(in_place)
        {
            return src;
        }
        else
        {
            return dst;
        }
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType dt, float epsilon)
    {
        // Create reference
        SimpleTensor<T> ref_src{ shape, dt, 1 };

        // Fill reference
        fill(ref_src);

        return reference::mean_std_normalization_layer(ref_src, epsilon);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    DataType        _data_type{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MEAN_STDDEV_NORMALIZATION_LAYER_FIXTURE */
