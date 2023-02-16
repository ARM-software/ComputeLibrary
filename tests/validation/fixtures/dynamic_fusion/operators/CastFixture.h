/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_CASTFIXTURE
#define TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_CASTFIXTURE

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/CastAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"

#include "tests/framework/Fixture.h"
#include "tests/validation/reference/DepthConvertLayer.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class DynamicFusionCastValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType dt_in, DataType dt_out, ConvertPolicy policy)
    {
        _target    = compute_target(shape, dt_in, dt_out, policy);
        _reference = compute_reference(shape, dt_in, dt_out, policy);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, DataType dt_in, DataType dt_out)
    {
        // Restricting range to avoid inf values
        if(dt_out == DataType::F16)
        {
            constexpr int signed_min   = -32000;
            constexpr int signed_max   = 32000;
            constexpr int unsigned_min = 0;
            constexpr int unsigned_max = 65000;

            switch(dt_in)
            {
                case DataType::U8:
                case DataType::QASYMM8:
                case DataType::QASYMM8_SIGNED:
                case DataType::S8:
                case DataType::F32:
                {
                    library->fill_tensor_uniform(tensor, i);
                    break;
                }
                case DataType::U16:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<uint16_t>(unsigned_min), static_cast<uint16_t>(unsigned_max));
                    break;
                }
                case DataType::S16:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<int16_t>(signed_min), static_cast<int16_t>(signed_max));
                    break;
                }
                case DataType::U32:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<uint32_t>(unsigned_min), static_cast<uint32_t>(unsigned_max));
                    break;
                }
                case DataType::S32:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<int32_t>(signed_min), static_cast<int32_t>(signed_max));
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("NOT SUPPORTED!");
            }
        }
        else
        {
            library->fill_tensor_uniform(tensor, i);
        }
    }

    // Given input is in nchw format
    TensorType compute_target(const TensorShape &shape, const DataType dt_in, const DataType dt_out, const ConvertPolicy policy)
    {
        // Create a new workload sketch
        auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        auto              gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
        GpuWorkloadSketch sketch{ &gpu_ctx };

        // Create sketch tensors
        TensorInfo src_info = sketch.create_tensor_info(TensorInfo(shape, 1, dt_in, DataLayout::NCHW)); // layout is not important
        TensorInfo dst_info = sketch.create_tensor_info();

        CastAttributes attributes;
        attributes.convert_policy(policy).data_type(dt_out);

        ITensorInfo *ans_info = FunctionType::create_op(sketch, &src_info, attributes);
        GpuOutput::create_op(sketch, ans_info, &dst_info);

        // Configure runtime
        ClWorkloadRuntime runtime;
        runtime.configure(sketch);

        // (Important) Allocate auxiliary tensor memory if there are any
        for(auto &data : runtime.get_auxiliary_tensors())
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
        t_src.allocator()->init(src_info);
        t_dst.allocator()->init(dst_info);

        // Allocate and fill user tensors
        t_src.allocator()->allocate();
        t_dst.allocator()->allocate();

        fill(AccessorType(t_src), 0, dt_in, dt_out);

        // Run runtime
        runtime.run({ &t_src, &t_dst });
        return t_dst;
    }

    SimpleTensor<T2> compute_reference(const TensorShape &shape, const DataType dt_in, const DataType dt_out, const ConvertPolicy policy)
    {
        // Create reference
        SimpleTensor<T1> src{ shape, dt_in, 1 };

        // Fill reference
        fill(src, 0, dt_in, dt_out);

        return reference::depth_convert<T1, T2>(src, dt_out, policy, 0);
    }

    TensorType       _target{};
    SimpleTensor<T2> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_CASTFIXTURE */
