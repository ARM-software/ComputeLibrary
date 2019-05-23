/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_PADLAYER_FIXTURE
#define ARM_COMPUTE_TEST_PADLAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/PadLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PaddingFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, const PaddingList &padding, const PaddingMode mode)
    {
        PaddingList clamped_padding = padding;
        if(mode != PaddingMode::CONSTANT)
        {
            // Clamp padding to prevent applying more than is possible.
            for(uint32_t i = 0; i < padding.size(); ++i)
            {
                if(mode == PaddingMode::REFLECT)
                {
                    clamped_padding[i].first  = std::min(static_cast<uint64_t>(padding[i].first), static_cast<uint64_t>(shape[i] - 1));
                    clamped_padding[i].second = std::min(static_cast<uint64_t>(padding[i].second), static_cast<uint64_t>(shape[i] - 1));
                }
                else
                {
                    clamped_padding[i].first  = std::min(static_cast<uint64_t>(padding[i].first), static_cast<uint64_t>(shape[i]));
                    clamped_padding[i].second = std::min(static_cast<uint64_t>(padding[i].second), static_cast<uint64_t>(shape[i]));
                }
            }
        }
        _target    = compute_target(shape, data_type, clamped_padding, mode);
        _reference = compute_reference(shape, data_type, clamped_padding, mode);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(const TensorShape &shape,
                              DataType           data_type,
                              const PaddingList &paddings,
                              const PaddingMode  mode)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);
        TensorType dst;

        TensorType const_val = create_tensor<TensorType>(TensorShape(1), data_type);
        const_val.allocator()->allocate();
        fill(AccessorType(const_val), 1);
        T const_value = *static_cast<T *>(AccessorType(const_val)(Coordinates(0)));

        // Create and configure function
        FunctionType padding;
        padding.configure(&src, &dst, paddings, const_value, mode);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);

        // Compute function
        padding.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type,
                                      const PaddingList &paddings, const PaddingMode mode)
    {
        // Create reference tensor
        SimpleTensor<T> src{ shape, data_type };
        SimpleTensor<T> const_val{ TensorShape(1), data_type };

        // Fill reference tensor
        fill(src, 0);
        fill(const_val, 1);

        return reference::pad_layer(src, paddings, const_val[0], mode);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_PADLAYER_FIXTURE */
