/*
 * Copyright (c) 2017-2021, 2023-2025 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_QUANTIZATIONLAYERFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_QUANTIZATIONLAYERFIXTURE_H

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
    void setup(TensorShape shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo_out, QuantizationInfo qinfo_in)
    {
        if(std::is_same<TensorType, Tensor>::value &&  // Cpu
            (data_type_in == DataType::F16 || data_type_out == DataType::F16) && !CPUInfo::get().has_fp16())
        {
            return;
        }

        QuantizationInfo output_qinfo( (data_type_out == DataType::QSYMM8_PER_CHANNEL)? generate_quantization_info(data_type_out, shape.z()) : qinfo_out );
        _target    = compute_target(shape, data_type_in, data_type_out, output_qinfo, qinfo_in);
        _reference = compute_reference(shape, data_type_in, data_type_out, output_qinfo, qinfo_in);
    }

protected:
    QuantizationInfo generate_quantization_info(DataType data_type, int32_t num_channels)
    {
        std::mt19937                    gen(library.get()->seed());
        std::uniform_int_distribution<> distribution_offset_q8(1, 127);

        switch(data_type)
        {
            case DataType::QSYMM8_PER_CHANNEL:
            {
                std::vector<float> scale(num_channels);
                for(int32_t i = 0; i < num_channels; ++i)
                {
                    scale[i] = 1.f / distribution_offset_q8(gen);
                }
                return QuantizationInfo(scale);
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo_out, QuantizationInfo qinfo_in)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type_in, 1, qinfo_in);
        TensorType dst = create_tensor<TensorType>(shape, data_type_out, 1, qinfo_out);

        // Create and configure function
        FunctionType quantization_layer;
        quantization_layer.configure(&src, &dst);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        quantization_layer.run();

        return dst;
    }

    SimpleTensor<Tout> compute_reference(const TensorShape &shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo_out, QuantizationInfo qinfo_in)
    {
        // Create reference
        SimpleTensor<Tin> src{ shape, data_type_in, 1, qinfo_in };

        // Fill reference
        fill(src);

        return reference::quantization_layer<Tin, Tout>(src, data_type_out, qinfo_out);
    }

    TensorType         _target{};
    SimpleTensor<Tout> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename Tin, typename Tout>
class QuantizationValidationFixture : public QuantizationValidationGenericFixture<TensorType, AccessorType, FunctionType, Tin, Tout>
{
public:
    void setup(TensorShape shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo_out)
    {
        QuantizationValidationGenericFixture<TensorType, AccessorType, FunctionType, Tin, Tout>::setup(shape, data_type_in, data_type_out, qinfo_out, QuantizationInfo());
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_QUANTIZATIONLAYERFIXTURE_H
