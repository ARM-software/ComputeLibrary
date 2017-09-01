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
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/CPP/ConvolutionLayer.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DirectConvolutionValidationFixedPointFixture : public ConvolutionValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, int stride_x, int stride_y, int pad_x, int pad_y, unsigned int kernel_size, unsigned int num_kernels, DataType data_type, int fractional_bits)
    {
        const TensorShape   weights_shape(kernel_size, kernel_size, input_shape.z(), num_kernels);
        const TensorShape   bias_shape(num_kernels);
        const PadStrideInfo info(stride_x, stride_y, pad_x, pad_y, DimensionRoundingType::FLOOR);
        const TensorShape   output_shape = get_output_shape(input_shape, weights_shape, info);

        ConvolutionValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, data_type, fractional_bits);
    }

private:
    TensorShape get_output_shape(TensorShape in_shape, TensorShape kernel_shape, const PadStrideInfo &info)
    {
        TensorShape out_shape(in_shape);
        const std::pair<unsigned int, unsigned int> scaled_dims = scaled_dimensions(in_shape.x(),
                                                                                    in_shape.y(),
                                                                                    kernel_shape.x(),
                                                                                    kernel_shape.y(),
                                                                                    info);
        out_shape.set(0, scaled_dims.first);
        out_shape.set(1, scaled_dims.second);
        out_shape.set(2, kernel_shape[3]);
        return out_shape;
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DirectConvolutionValidationFixture : public DirectConvolutionValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, int stride_x, int stride_y, int pad_x, int pad_y, unsigned int kernel_size, unsigned int num_kernels, DataType data_type)
    {
        DirectConvolutionValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, stride_x, stride_y, pad_x, pad_y, kernel_size, num_kernels, data_type, 0);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
