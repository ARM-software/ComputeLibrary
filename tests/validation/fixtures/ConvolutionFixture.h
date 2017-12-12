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
#ifndef ARM_COMPUTE_TEST_CONVOLUTION_FIXTURE
#define ARM_COMPUTE_TEST_CONVOLUTION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Convolution.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, const unsigned int filter_size>
class ConvolutionValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, BorderMode border_mode)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> distribution(0, 255);
        const uint8_t                          constant_border_value = distribution(gen);

        // Generate random scale value between 0 and 255.
        const uint32_t scale = distribution(gen);

        switch(filter_size)
        {
            case 3:
            case 5:
            case 7:
            case 9:
                int16_t conv[filter_size * filter_size];
                create_conv(conv);

                _target    = compute_target(shape, data_type, conv, scale, border_mode, constant_border_value);
                _reference = compute_reference(shape, data_type, conv, scale, border_mode, constant_border_value);
                break;
            default:
                ARM_COMPUTE_ERROR("Filter Size Not Supported");
        }
    }

protected:
    void create_conv(int16_t *conv)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<int16_t> distribution_int16(-32768, 32767);

        for(unsigned int i = 0; i < filter_size * filter_size; ++i)
        {
            conv[i] = distribution_int16(gen);
        }
    }

    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);
        TensorType dst = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        FunctionType convolution;
        convolution.configure(&src, &dst, conv, scale, border_mode, constant_border_value);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(dst), 1);

        // Compute function
        convolution.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value)
    {
        ARM_COMPUTE_ERROR_ON(data_type != DataType::U8);

        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src, 0);

        // Compute reference
        return reference::convolution<T>(src, conv, scale, border_mode, constant_border_value, filter_size);
    }

    BorderMode      _border_mode{};
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CONVOLUTION_FIXTURE */
