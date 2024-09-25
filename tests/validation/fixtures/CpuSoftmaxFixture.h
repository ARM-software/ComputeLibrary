/*
 * Copyright (c) 2017-2021, 2023-2024 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUSOFTMAXFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUSOFTMAXFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/SoftmaxLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool IS_LOG = false>
class CpuSoftmaxValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, DataType data_type, float beta, size_t axis)
    {
        if(std::is_same<TensorType, Tensor>::value &&  // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        quantization_info_ = QuantizationInfo();

        reference_ = compute_reference(shape, data_type, quantization_info_, beta, axis);
        target_    = compute_target(shape, data_type, quantization_info_, beta, axis);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if(tensor.data_type() == DataType::F32)
        {
            std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
            library->fill(tensor, distribution, 0);
        }
        else if(tensor.data_type() == DataType::F16)
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -10.0f, 10.0f };
            library->fill(tensor, distribution, 0);
        }
        else if(!is_data_type_quantized(tensor.data_type()))
        {
            std::uniform_int_distribution<> distribution(0, 100);
            library->fill(tensor, distribution, 0);
        }
        else
        {
            library->fill_tensor_uniform(tensor, 0);
        }
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type,
                              QuantizationInfo quantization_info, float beta, int32_t axis)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type, 1, quantization_info);
        TensorType dst = create_tensor<TensorType>(shape, data_type, 1, get_softmax_output_quantization_info(data_type, IS_LOG));

        // Create and configure function
        FunctionType softmax;
        softmax.configure(src.info(), dst.info(), beta, axis);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src));

        ITensorPack run_pack{ { arm_compute::TensorType::ACL_SRC_0, &src }};
        run_pack.add_tensor(arm_compute::TensorType::ACL_DST, &dst);
        auto mg = MemoryGroup{};
        auto ws = manage_workspace<Tensor>(softmax.workspace(), mg, run_pack);

        // Compute function
        softmax.run(run_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type,
                                      QuantizationInfo quantization_info, float beta, int32_t axis)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 1, quantization_info };

        // Fill reference
        fill(src);

        return reference::softmax_layer<T>(src, beta, axis, IS_LOG);
    }

    TensorType       target_{};
    SimpleTensor<T>  reference_{};
    QuantizationInfo quantization_info_{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUSOFTMAXFIXTURE_H
