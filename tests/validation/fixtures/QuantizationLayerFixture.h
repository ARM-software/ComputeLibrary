/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_QUANTIZATION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_QUANTIZATION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/QuantizationLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename Tin, typename Tout>
class QuantizationValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo, QuantizationInfo qinfo_in)
    {
        _target    = compute_target(shape, data_type_in, data_type_out, qinfo, qinfo_in);
        _reference = compute_reference(shape, data_type_in, data_type_out, qinfo, qinfo_in);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo, QuantizationInfo qinfo_in)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type_in, 1, qinfo_in);
        TensorType dst = create_tensor<TensorType>(shape, data_type_out, 1, qinfo);

        // Create and configure function
        FunctionType quantization_layer;
        quantization_layer.configure(&src, &dst);

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
        quantization_layer.run();

        return dst;
    }

    SimpleTensor<Tout> compute_reference(const TensorShape &shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo, QuantizationInfo qinfo_in)
    {
        // Create reference
        SimpleTensor<Tin> src{ shape, data_type_in, 1, qinfo_in };

        // Fill reference
        fill(src);

        return reference::quantization_layer<Tin, Tout>(src, data_type_out, qinfo);
    }

    TensorType         _target{};
    SimpleTensor<Tout> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename Tin, typename Tout>
class QuantizationValidationFixture : public QuantizationValidationGenericFixture<TensorType, AccessorType, FunctionType, Tin, Tout>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo)
    {
        QuantizationValidationGenericFixture<TensorType, AccessorType, FunctionType, Tin, Tout>::setup(shape, data_type_in, data_type_out, qinfo, QuantizationInfo());
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_QUANTIZATION_LAYER_FIXTURE */
