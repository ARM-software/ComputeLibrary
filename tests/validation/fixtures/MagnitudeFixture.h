/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_MAGNITUDE_FIXTURE
#define ARM_COMPUTE_TEST_MAGNITUDE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Magnitude.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class MagnitudeValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, Format format, MagnitudeType magnitude_type, bool use_fp16)
    {
        _target         = compute_target(shape, format, magnitude_type, use_fp16);
        _reference      = compute_reference(shape, format, magnitude_type);
        _magnitude_type = magnitude_type;
    }

protected:
    template <typename U>
    void fill(U &&tensor, std::random_device::result_type seed_offset)
    {
        library->fill_tensor_uniform(tensor, seed_offset);
    }

    TensorType compute_target(const TensorShape &shape, Format format, MagnitudeType magnitude_type, bool use_fp16)
    {
        DataType data_type = data_type_from_format(format);

        // Create tensors
        TensorType src1 = create_tensor<TensorType>(shape, data_type);
        src1.info()->set_format(format);

        TensorType src2 = create_tensor<TensorType>(shape, data_type);
        src2.info()->set_format(format);

        TensorType dst = create_tensor<TensorType>(shape, data_type);
        dst.info()->set_format(format);

        // Create and configure function
        FunctionType magnitude;
        magnitude.configure(&src1, &src2, &dst, magnitude_type, use_fp16);

        ARM_COMPUTE_EXPECT(src1.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(src2.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src1.allocator()->allocate();
        src2.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src1.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!src2.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src1), 0);
        fill(AccessorType(src2), 1);

        // Compute function
        magnitude.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, Format format, MagnitudeType magnitude_type)
    {
        DataType data_type = data_type_from_format(format);

        // Create reference
        SimpleTensor<T> src1{ shape, data_type };
        SimpleTensor<T> src2{ shape, data_type };

        // Fill reference
        fill(src1, 0);
        fill(src2, 1);

        return reference::magnitude<T>(src1, src2, magnitude_type);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    MagnitudeType   _magnitude_type{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MAGNITUDE_FIXTURE */
