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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_RESHAPEFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_RESHAPEFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/ReshapeAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuReshape.h"

#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/validation/reference/ReshapeLayer.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionGpuReshapeLayerValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape input_shape, TensorShape output_shape, DataType data_type)
    {
        _target    = compute_target(input_shape, output_shape, data_type);
        _reference = compute_reference(input_shape, output_shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(TensorShape &input_shape, TensorShape &output_shape, DataType data_type)
    {
        // Check if indeed the input shape can be reshape to the output one
        ARM_COMPUTE_ASSERT(input_shape.total_size() == output_shape.total_size());

        // Create a new workload sketch
        auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        auto              context        = GpuWorkloadContext{&cl_compile_ctx};
        GpuWorkloadSketch sketch{&context};

        // Create sketch tensors
        ITensorInfo      *src_info = context.create_tensor_info(TensorInfo(input_shape, 1, data_type));
        ITensorInfo      *dst_info = context.create_tensor_info(TensorInfo(output_shape, 1, data_type));
        ReshapeAttributes attributes;
        attributes.shape(output_shape);

        ITensorInfo *ans_info = FunctionType::create_op(sketch, src_info, attributes);
        GpuOutput::create_op(sketch, ans_info, dst_info);

        // Configure runtime
        ClWorkloadRuntime runtime;
        runtime.configure(sketch);

        // (Important) Allocate auxiliary tensor memory if there are any
        for (auto &data : runtime.get_auxiliary_tensors())
        {
            CLTensor     *tensor      = std::get<0>(data);
            TensorInfo    info        = std::get<1>(data);
            AuxMemoryInfo aux_mem_req = std::get<2>(data);
            tensor->allocator()->init(info, aux_mem_req.alignment);
            tensor->allocator()->allocate(); // Use ACL allocated memory
        }

        // Construct user tensors
        TensorType t_src{};
        TensorType t_dst{};
        // Initialize user tensors
        t_src.allocator()->init(*src_info);
        t_dst.allocator()->init(*dst_info);

        // Allocate and fill user tensors
        t_src.allocator()->allocate();
        t_dst.allocator()->allocate();

        fill(AccessorType(t_src), 0);

        // Run runtime
        runtime.run({&t_src, &t_dst});

        return t_dst;
    }

    SimpleTensor<T>
    compute_reference(const TensorShape &input_shape, const TensorShape &output_shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{input_shape, data_type};

        // Fill reference
        fill(src, 0);

        return reference::reshape_layer<T>(src, output_shape);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
/** [ReshapeLayer fixture] **/
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_RESHAPEFIXTURE_H
