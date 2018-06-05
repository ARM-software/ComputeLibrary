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
#ifndef ARM_COMPUTE_TEST_CONVOLUTIONFIXTURE
#define ARM_COMPUTE_TEST_CONVOLUTIONFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
/** Parent fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class ConvolutionFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape src_shape, DataType output_data_type, BorderMode border_mode, unsigned int width, unsigned int height, bool is_separable = false)
    {
        std::mt19937  gen(library->seed());
        const uint8_t constant_border_value = 0;

        // Generate random scale value between 1 and 255.
        std::uniform_int_distribution<uint8_t> distribution_scale(1, 255);
        const uint32_t                         scale = distribution_scale(gen);

        ARM_COMPUTE_ERROR_ON(3 != width && 5 != width && 7 != width && 9 != width);
        ARM_COMPUTE_ERROR_ON(3 != height && 5 != height && 7 != height && 9 != height);

        std::vector<int16_t> conv(width * height);

        _width  = width;
        _height = height;

        if(is_separable)
        {
            init_separable_conv(conv.data(), width, height, seed);
        }
        else
        {
            init_conv(conv.data(), width, height, seed);
        }

        // Create tensors
        src = create_tensor<TensorType>(src_shape, DataType::U8);
        dst = create_tensor<TensorType>(src_shape, output_data_type);

        // Configure function
        configure_target(convolution_func, src, dst, conv.data(), scale, border_mode, constant_border_value);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill tensors
        library->fill_tensor_uniform(Accessor(src), 0);
        library->fill_tensor_uniform(Accessor(dst), 1);
    }

    void run()
    {
        convolution_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

protected:
    virtual void configure_target(Function &func, TensorType &src, TensorType &dst, const int16_t *conv, uint32_t scale,
                                  BorderMode border_mode, uint8_t border_value) = 0;

protected:
    unsigned int _width{};
    unsigned int _height{};
    Function     convolution_func{};

private:
    const std::random_device::result_type seed = 0;
    TensorType                            src{};
    TensorType                            dst{};
};

/** Child fixture used for square convolutions */
template <typename TensorType, typename Function, typename Accessor>
class ConvolutionSquareFixture : public ConvolutionFixture<TensorType, Function, Accessor>
{
public:
    template <typename...>
    void setup(TensorShape src_shape, DataType output_data_type, BorderMode border_mode, unsigned int width)
    {
        ConvolutionFixture<TensorType, Function, Accessor>::setup(src_shape, output_data_type, border_mode, width, width);
    }

protected:
    void configure_target(Function &func, TensorType &src, TensorType &dst, const int16_t *conv, uint32_t scale,
                          BorderMode border_mode, uint8_t constant_border_value)
    {
        this->convolution_func.configure(&src, &dst, conv, scale, border_mode, constant_border_value);
    }
};

/** Child fixture used for rectangular convolutions */
template <typename TensorType, typename Function, typename Accessor>
class ConvolutionRectangleFixture : public ConvolutionFixture<TensorType, Function, Accessor>
{
public:
    template <typename...>
    void setup(TensorShape src_shape, DataType output_data_type, BorderMode border_mode, unsigned int width, unsigned int height)
    {
        ConvolutionFixture<TensorType, Function, Accessor>::setup(src_shape, output_data_type, border_mode, width, height);
    }

protected:
    void configure_target(Function &func, TensorType &src, TensorType &dst, const int16_t *conv, uint32_t scale,
                          BorderMode border_mode, uint8_t constant_border_value)
    {
        this->convolution_func.configure(&src, &dst, conv, this->_width, this->_height, scale, border_mode, constant_border_value);
    }
};

/** Child fixture used for separable convolutions */
template <typename TensorType, typename Function, typename Accessor>
class ConvolutionSeperableFixture : public ConvolutionFixture<TensorType, Function, Accessor>
{
public:
    template <typename...>
    void setup(TensorShape src_shape, DataType output_data_type, BorderMode border_mode, unsigned int width)
    {
        ConvolutionFixture<TensorType, Function, Accessor>::setup(src_shape, output_data_type, border_mode, width, width, true);
    }

protected:
    void configure_target(Function &func, TensorType &src, TensorType &dst, const int16_t *conv, uint32_t scale,
                          BorderMode border_mode, uint8_t constant_border_value)
    {
        this->convolution_func.configure(&src, &dst, conv, scale, border_mode, constant_border_value);
    }
};

} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CONVOLUTIONFIXTURE */
