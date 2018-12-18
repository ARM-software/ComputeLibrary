/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_RANGE_FIXTURE
#define ARM_COMPUTE_TEST_RANGE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Range.h"

#include <algorithm>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
size_t num_of_elements_in_range(float start, float end, float step)
{
    ARM_COMPUTE_ERROR_ON(step == 0);
    return size_t(std::ceil((end - start) / step));
}
}

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RangeFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(const DataType data_type0, float start, float step, const QuantizationInfo qinfo0 = QuantizationInfo())
    {
        _target    = compute_target(data_type0, qinfo0, start, step);
        _reference = compute_reference(data_type0, qinfo0, start, step);
    }

protected:
    float get_random_end(const DataType output_data_type, const QuantizationInfo qinfo_out, float start, float step)
    {
        std::uniform_real_distribution<> distribution(1, 100);
        std::mt19937                     gen(library->seed());
        float                            end = start;
        switch(output_data_type)
        {
            case DataType::U8:
                end += std::max((uint8_t)1, static_cast<uint8_t>(distribution(gen))) * step;
                return utility::clamp<float, uint8_t>(end);
            case DataType::U16:
                end += std::max((uint16_t)1, static_cast<uint16_t>(distribution(gen))) * step;
                return utility::clamp<float, uint16_t>(end);
            case DataType::U32:
                end += std::max((uint32_t)1, static_cast<uint32_t>(distribution(gen))) * step;
                return utility::clamp<float, uint32_t>(end);
            case DataType::S8:
                end += std::max((int8_t)1, static_cast<int8_t>(distribution(gen))) * step;
                return utility::clamp<float, int8_t>(end);
            case DataType::S16:
                end += std::max((int16_t)1, static_cast<int16_t>(distribution(gen))) * step;
                return utility::clamp<float, int16_t>(end);
            case DataType::S32:
                end += std::max((int32_t)1, static_cast<int32_t>(distribution(gen))) * step;
                return utility::clamp<float, int32_t>(end);
            case DataType::F32:
                end += std::max(1.0f, static_cast<float>(distribution(gen))) * step;
                return end;
            case DataType::F16:
                end += std::max(half(1.0f), static_cast<half>(distribution(gen))) * step;
                return utility::clamp<float, half>(end);
            case DataType::QASYMM8:
                return utility::clamp<float, uint8_t>(end + (float)distribution(gen) * step, qinfo_out.dequantize(0), qinfo_out.dequantize(std::numeric_limits<uint8_t>::max()));
            default:
                return 0;
        }
    }

    TensorType compute_target(const DataType output_data_type, const QuantizationInfo qinfo_out, float start, float step)
    {
        float  end             = get_random_end(output_data_type, qinfo_out, start, step);
        size_t num_of_elements = num_of_elements_in_range(start, end, step);
        // Create tensor
        TensorType dst = create_tensor<TensorType>(TensorShape(num_of_elements), output_data_type, 1, qinfo_out);
        // Create and configure function
        FunctionType range_func;
        range_func.configure(&dst, start, end, step);

        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        // Allocate tensors
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Compute function
        range_func.run();
        return dst;
    }

    SimpleTensor<T> compute_reference(const DataType output_data_type, const QuantizationInfo qinfo_out, float start, float step)
    {
        // Create tensor
        const float     end             = get_random_end(output_data_type, qinfo_out, start, step);
        size_t          num_of_elements = num_of_elements_in_range(start, end, step);
        SimpleTensor<T> ref_dst{ TensorShape(num_of_elements ? num_of_elements : 1), output_data_type, 1, qinfo_out };
        return reference::range<T>(ref_dst, start, num_of_elements, step);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_RANGE_FIXTURE */
