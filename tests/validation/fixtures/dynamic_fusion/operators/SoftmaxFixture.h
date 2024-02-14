/*
* Copyright (c) 2023-2024 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_SOFTMAXFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_SOFTMAXFIXTURE_H

#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/SoftmaxAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"

#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/SimpleTensor.h"
#include "tests/validation/reference/SoftmaxLayer.h"
#include "tests/validation/Validation.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionSoftmaxValidationGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, DataType data_type, float beta, size_t axis, bool is_log)
    {
        _reference = compute_reference(shape, data_type, beta, axis, is_log);
        _target    = compute_target(shape, data_type, beta, axis, is_log);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if (tensor.data_type() == DataType::F32)
        {
            std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
            library->fill(tensor, distribution, 0);
        }
        else if (tensor.data_type() == DataType::F16)
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution{-10.0f, 10.0f};
            library->fill(tensor, distribution, 0);
        }
        else if (!is_data_type_quantized(tensor.data_type()))
        {
            std::uniform_int_distribution<> distribution(0, 100);
            library->fill(tensor, distribution, 0);
        }
        else
        {
            library->fill_tensor_uniform(tensor, 0);
        }
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type, float beta, int32_t axis, bool is_log)
    {
        // Create a new workload sketch
        CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        GpuWorkloadContext context        = GpuWorkloadContext{&cl_compile_ctx};
        GpuWorkloadSketch  sketch{&context};

        SoftmaxAttributes softmax_attr{};
        softmax_attr.axis(axis).beta(beta).is_log_softmax(is_log);
        ITensorInfo *src_info = context.create_tensor_info(shape, 1, data_type);
        ITensorInfo *dst_info = context.create_tensor_info(shape, 1, data_type);
        FunctionType::create_op(sketch, src_info, dst_info, softmax_attr);

        // Configure runtime
        ClWorkloadRuntime runtime;
        runtime.configure(sketch);

        // (Important) Allocate auxiliary tensor memory if there are any
        // Instead of using ACL allocated memory, the user can choose to import memory into the tensors
        for (auto &data : runtime.get_auxiliary_tensors())
        {
            CLTensor     *tensor      = std::get<0>(data);
            TensorInfo    info        = std::get<1>(data);
            AuxMemoryInfo aux_mem_req = std::get<2>(data);
            tensor->allocator()->init(info, aux_mem_req.alignment);
            tensor->allocator()->allocate(); // Use ACL allocated memory
        }
        // Construct user tensors
        TensorType src{};
        TensorType dst{};

        // Initialize user tensors
        src.allocator()->init(*src_info);
        dst.allocator()->init(*dst_info);

        // Allocate and fill user tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        fill(AccessorType(src));

        // Run runtime
        runtime.run({&src, &dst});

        return dst;
    }

    SimpleTensor<T>
    compute_reference(const TensorShape &shape, DataType data_type, float beta, int32_t axis, bool is_log)
    {
        // Create reference
        SimpleTensor<T> src{shape, data_type, 1};

        // Fill reference
        fill(src);

        return reference::softmax_layer<T>(src, beta, axis, is_log);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionSoftmaxValidationFixture
    : public DynamicFusionSoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, DataType data_type, float beta, size_t axis, bool is_log)
    {
        DynamicFusionSoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, data_type, beta, axis, is_log);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_SOFTMAXFIXTURE_H
