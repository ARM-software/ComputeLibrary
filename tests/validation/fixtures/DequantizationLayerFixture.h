/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_DEQUANTIZATION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_DEQUANTIZATION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/DequantizationLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DequantizationValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType src_data_type, DataType dst_datatype, DataLayout data_layout)
    {
        _quantization_info = generate_quantization_info(src_data_type, shape.z());
        _target            = compute_target(shape, src_data_type, dst_datatype, data_layout);
        _reference         = compute_reference(shape, src_data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(TensorShape shape, DataType src_data_type, DataType dst_datatype, DataLayout data_layout)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, src_data_type, 1, _quantization_info, data_layout);
        TensorType dst = create_tensor<TensorType>(shape, dst_datatype, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType dequantization_layer;
        dequantization_layer.configure(&src, &dst);

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
        dequantization_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType src_data_type)
    {
        switch(src_data_type)
        {
            case DataType::QASYMM8:
            {
                SimpleTensor<uint8_t> src{ shape, src_data_type, 1, _quantization_info };
                fill(src);
                return reference::dequantization_layer<T>(src);
            }
            case DataType::QSYMM8_PER_CHANNEL:
            case DataType::QSYMM8:
            {
                SimpleTensor<int8_t> src{ shape, src_data_type, 1, _quantization_info };
                fill(src);
                return reference::dequantization_layer<T>(src);
            }
            case DataType::QSYMM16:
            {
                SimpleTensor<int16_t> src{ shape, src_data_type, 1, _quantization_info };
                fill(src);
                return reference::dequantization_layer<T>(src);
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

protected:
    QuantizationInfo generate_quantization_info(DataType data_type, int32_t num_channels)
    {
        std::mt19937                    gen(library.get()->seed());
        std::uniform_int_distribution<> distribution_scale_q8(1, 255);
        std::uniform_int_distribution<> distribution_offset_q8(1, 127);
        std::uniform_int_distribution<> distribution_scale_q16(1, 32768);

        switch(data_type)
        {
            case DataType::QSYMM16:
                return QuantizationInfo(1.f / distribution_scale_q16(gen));
            case DataType::QSYMM8:
                return QuantizationInfo(1.f / distribution_scale_q8(gen));
            case DataType::QSYMM8_PER_CHANNEL:
            {
                std::vector<float> scale(num_channels);
                for(int32_t i = 0; i < num_channels; ++i)
                {
                    scale[i] = 1.f / distribution_offset_q8(gen);
                }
                return QuantizationInfo(scale);
            }
            case DataType::QASYMM8:
                return QuantizationInfo(1.f / distribution_scale_q8(gen), distribution_offset_q8(gen));
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

protected:
    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    QuantizationInfo _quantization_info{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEQUANTIZATION_LAYER_FIXTURE */
