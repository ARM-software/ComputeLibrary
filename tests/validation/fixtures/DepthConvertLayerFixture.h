/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_DEPTH_CONVERT_FIXTURE
#define ARM_COMPUTE_TEST_DEPTH_CONVERT_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/DepthConvertLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
/*  This function ignores the scale and zeroPoint of quanized tensors, i.e. QASYMM8 input is treated as uint8 values.*/
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class DepthConvertLayerValidationBaseFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift, QuantizationInfo quantization_info)
    {
        _shift             = shift;
        _quantization_info = quantization_info;
        _target            = compute_target(shape, dt_in, dt_out, policy, shift);
        _reference         = compute_reference(shape, dt_in, dt_out, policy, shift);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, DataType dt_in, DataType dt_out)
    {
        if(is_data_type_quantized(tensor.data_type()))
        {
            std::pair<int, int> bounds = get_quantized_bounds(tensor.quantization_info(), -1.0f, 1.0f);
            std::uniform_int_distribution<uint8_t> distribution(bounds.first, bounds.second);

            library->fill(tensor, distribution, i);
        }
        else
        {
            // When converting S32 to F16, both reference and Neon implementations are + or - infinity outside the F16 range.
            if(dt_in == DataType::S32 && dt_out == DataType::F16)
            {
                std::uniform_int_distribution<int32_t> distribution_s32(-65504, 65504);
                library->fill(tensor, distribution_s32, i);
            }
            else
            {
                library->fill_tensor_uniform(tensor, i);
            }
        }
    }

    TensorType compute_target(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, dt_in, 1, _quantization_info);
        TensorType dst = create_tensor<TensorType>(shape, dt_out, 1, _quantization_info);

        // Create and configure function
        FunctionType depth_convert;
        depth_convert.configure(&src, &dst, policy, shift);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0, dt_in, dt_out);

        // Compute function
        depth_convert.run();

        return dst;
    }

    SimpleTensor<T2> compute_reference(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift)
    {
        // Create reference
        SimpleTensor<T1> src{ shape, dt_in, 1, _quantization_info };

        // Fill reference
        fill(src, 0, dt_in, dt_out);

        return reference::depth_convert<T1, T2>(src, dt_out, policy, shift);
    }

    TensorType       _target{};
    SimpleTensor<T2> _reference{};
    int              _shift{};
    QuantizationInfo _quantization_info{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class DepthConvertLayerValidationFixture : public DepthConvertLayerValidationBaseFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift)
    {
        DepthConvertLayerValidationBaseFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, dt_in, dt_out, policy,
                                                                                                      shift, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class DepthConvertLayerValidationQuantizedFixture : public DepthConvertLayerValidationBaseFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift, QuantizationInfo quantization_info)
    {
        DepthConvertLayerValidationBaseFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, dt_in, dt_out, policy,
                                                                                                      shift, quantization_info);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTH_CONVERT_FIXTURE */
