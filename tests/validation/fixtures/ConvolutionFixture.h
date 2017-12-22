/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ConvolutionValidationFixture : public framework::Fixture
{
protected:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, BorderMode border_mode, const unsigned int width, const unsigned int height, const bool is_separable = false)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> distribution(0, 255);
        const uint8_t                          constant_border_value = distribution(gen);

        // Generate random scale value between 1 and 255.
        std::uniform_int_distribution<uint8_t> distribution_scale(1, 255);
        const uint32_t                         scale = distribution_scale(gen);

        ARM_COMPUTE_ERROR_ON(3 != width && 5 != width && 7 != width && 9 != width);
        ARM_COMPUTE_ERROR_ON(3 != height && 5 != height && 7 != height && 9 != height);

        int16_t conv[width * height];

        _width  = width;
        _height = height;

        if(is_separable)
        {
            create_separable_conv(conv);
        }
        else
        {
            create_conv(conv);
        }

        _target    = compute_target(shape, data_type, conv, scale, border_mode, constant_border_value);
        _reference = compute_reference(shape, data_type, conv, scale, border_mode, constant_border_value);
    }

    void
    create_conv(int16_t *conv)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<int16_t> distribution_int16(-32768, 32767);

        for(unsigned int i = 0; i < _width * _height; ++i)
        {
            conv[i] = distribution_int16(gen);
        }
    }

    void
    create_separable_conv(int16_t *conv)
    {
        std::mt19937 gen(library->seed());
        // Set it between -128 and 127 to ensure the matrix does not overflow
        std::uniform_int_distribution<int16_t> distribution_int16(-128, 127);

        int16_t conv_row[_width];
        int16_t conv_col[_height];

        conv_row[0] = conv_col[0] = 1;
        for(unsigned int i = 1; i < _width; ++i)
        {
            conv_row[i] = distribution_int16(gen);
            conv_col[i] = distribution_int16(gen);
        }

        // Multiply two matrices
        for(unsigned int i = 0; i < _width; ++i)
        {
            for(unsigned int j = 0; j < _height; ++j)
            {
                conv[i * _width + j] = conv_col[i] * conv_row[j];
            }
        }
    }

    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value)
    {
        ARM_COMPUTE_ERROR_ON(data_type != DataType::U8);

        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src, 0);

        // Compute reference
        return reference::convolution<T>(src, conv, scale, border_mode, constant_border_value, _width, _height);
    }

    virtual TensorType compute_target(const TensorShape &shape, DataType data_type, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value) = 0;

    BorderMode      _border_mode{};
    TensorType      _target{};
    SimpleTensor<T> _reference{};
    unsigned int    _width{};
    unsigned int    _height{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ConvolutionSquareValidationFixture : public ConvolutionValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, BorderMode border_mode, const unsigned int width)
    {
        ConvolutionValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, border_mode, width, width);
    }

protected:
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
        this->fill(AccessorType(src), 0);
        this->fill(AccessorType(dst), 1);

        // Compute function
        convolution.run();

        return dst;
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ConvolutionSeparableValidationFixture : public ConvolutionValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, BorderMode border_mode, const unsigned int width)
    {
        ConvolutionValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, border_mode, width, width, true);
    }

protected:
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
        this->fill(AccessorType(src), 0);
        this->fill(AccessorType(dst), 1);

        // Compute function
        convolution.run();

        return dst;
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ConvolutionRectangleValidationFixture : public ConvolutionValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, BorderMode border_mode, const unsigned int width, const unsigned int height)
    {
        ConvolutionValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, border_mode, width, height);
    }

protected:
    TensorType compute_target(const TensorShape &shape, DataType data_type, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);
        TensorType dst = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        FunctionType convolution;
        convolution.configure(&src, &dst, conv, this->_width, this->_height, scale, border_mode, constant_border_value);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        this->fill(AccessorType(src), 0);
        this->fill(AccessorType(dst), 1);

        // Compute function
        convolution.run();

        return dst;
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CONVOLUTION_FIXTURE */
