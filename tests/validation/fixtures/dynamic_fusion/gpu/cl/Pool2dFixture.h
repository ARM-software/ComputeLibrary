/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_GPU_CL_POOL2DFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_GPU_CL_POOL2DFIXTURE_H

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/Pool2dAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuPool2d.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"
#include "src/dynamic_fusion/utils/Utils.h"

#include "tests/CL/CLAccessor.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/PoolingLayer.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionGpuPool2dValidationGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape input_shape, const Pool2dAttributes &pool_attr, DataType data_type, bool mixed_precision)
    {
        _target    = compute_target(input_shape, pool_attr, data_type, mixed_precision);
        _reference = compute_reference(input_shape, convert_pool_attr_to_pool_info(pool_attr, mixed_precision), data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -1.0f, 1.0f };
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    // Given input is in nchw format
    TensorType compute_target(TensorShape input_shape, const Pool2dAttributes &pool_attr, const DataType data_type, bool mixed_precision)
    {
        CLScheduler::get().default_reinit();

        // Change shape due to NHWC data layout, test shapes are NCHW
        permute(input_shape, PermutationVector(2U, 0U, 1U));

        // Create a new workload sketch
        auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        auto              context        = GpuWorkloadContext{ &cl_compile_ctx };
        GpuWorkloadSketch sketch{ &context };

        // Create sketch tensors
        auto input_info = context.create_tensor_info(TensorInfo(input_shape, 1, data_type, DataLayout::NHWC));
        auto dst_info   = context.create_tensor_info();

        // Create Pool2dSettings
        GpuPool2dSettings pool_settings = GpuPool2dSettings().mixed_precision(mixed_precision);

        ITensorInfo *ans_info = FunctionType::create_op(sketch, &input_info, pool_attr, pool_settings);
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
        TensorType t_input{};
        TensorType t_dst{};

        // Initialize user tensors
        t_input.allocator()->init(input_info);
        t_dst.allocator()->init(dst_info);

        // Allocate and fill user tensors
        t_input.allocator()->allocate();
        t_dst.allocator()->allocate();

        fill(AccessorType(t_input), 0);

        // Run runtime
        runtime.run({ &t_input, &t_dst });
        return t_dst;
    }

    SimpleTensor<T> compute_reference(TensorShape shape, PoolingLayerInfo pool_info, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src(shape, data_type, 1, QuantizationInfo());
        // Fill reference
        fill(src, 0);
        return reference::pooling_layer<T>(src, pool_info, QuantizationInfo(), nullptr, DataLayout::NCHW);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionGpuPool2dValidationFixture : public DynamicFusionGpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape input_shape, PoolingType pool_type, Size2D pool_size, Padding2D pad, Size2D stride, bool exclude_padding, DataType data_type)
    {
        DynamicFusionGpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape,
                                                                                                         Pool2dAttributes().pool_type(pool_type).pool_size(pool_size).pad(pad).stride(stride).exclude_padding(exclude_padding),
                                                                                                         data_type, false);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionGpuPool2dMixedPrecisionValidationFixture : public DynamicFusionGpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape input_shape, PoolingType pool_type, Size2D pool_size, Padding2D pad, Size2D stride, bool exclude_padding, DataType data_type, bool mixed_precision)
    {
        DynamicFusionGpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape,
                                                                                                         Pool2dAttributes().pool_type(pool_type).pool_size(pool_size).pad(pad).stride(stride).exclude_padding(exclude_padding),
                                                                                                         data_type, mixed_precision);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionGpuPool2dSpecialValidationFixture : public DynamicFusionGpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape input_shape, Pool2dAttributes pool_attr, DataType data_type)
    {
        DynamicFusionGpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, pool_attr, data_type, false);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_GPU_CL_POOL2DFIXTURE_H
