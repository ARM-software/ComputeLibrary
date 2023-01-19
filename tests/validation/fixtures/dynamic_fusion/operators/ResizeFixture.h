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
#ifndef TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_RESIZEFIXTURE
#define TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_RESIZEFIXTURE

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/ResizeAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"

#include "tests/CL/CLAccessor.h"
#include "tests/SimpleTensor.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/Scale.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionResizeGenericValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, QuantizationInfo quantization_info, DataLayout data_layout,
               InterpolationPolicy interpolation_policy, SamplingPolicy sampling_policy,
               bool align_corners, QuantizationInfo output_quantization_info)
    {
        _shape                    = shape;
        _interpolation_policy     = interpolation_policy;
        _sampling_policy          = sampling_policy;
        _data_type                = data_type;
        _input_quantization_info  = quantization_info;
        _output_quantization_info = output_quantization_info;
        _align_corners            = align_corners;
        _data_layout              = data_layout;

        ARM_COMPUTE_ERROR_ON(data_layout != DataLayout::NHWC); // Dynamic fusion resize supports only NHWC layout

        generate_scale(shape);

        std::mt19937                            generator(library->seed());
        std::uniform_int_distribution<uint32_t> distribution_u8(0, 255);

        _target    = compute_target(shape);
        _reference = compute_reference(shape);
    }

protected:
    void generate_scale(const TensorShape &shape)
    {
        static constexpr float _min_scale{ 0.25f };
        static constexpr float _max_scale{ 3.f };

        constexpr float max_width{ 8192.0f };
        constexpr float max_height{ 6384.0f };
        constexpr float min_width{ 1.f };
        constexpr float min_height{ 1.f };

        std::mt19937                          generator(library->seed());
        std::uniform_real_distribution<float> distribution_float(_min_scale, _max_scale);

        auto generate = [&](size_t input_size, float min_output, float max_output) -> int
        {
            const float generated_scale = distribution_float(generator);
            const int   output_size     = static_cast<int>(utility::clamp(static_cast<float>(input_size) * generated_scale, min_output, max_output));
            return output_size;
        };

        // Input shape is always given in NCHW layout. NHWC is dealt by permute in compute_target()
        const int idx_width  = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::HEIGHT);

        _output_width  = generate(shape[idx_width], min_width, max_width);
        _output_height = generate(shape[idx_height], min_height, max_height);
    }

    template <typename U>
    void fill(U &&tensor)
    {
        if(tensor.data_type() == DataType::F32)
        {
            std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
            library->fill(tensor, distribution, 0);
        }
        else if(tensor.data_type() == DataType::F16)
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -5.0f, 5.0f };
            library->fill(tensor, distribution, 0);
        }
        else if(is_data_type_quantized(tensor.data_type()))
        {
            std::uniform_int_distribution<> distribution(0, 100);
            library->fill(tensor, distribution, 0);
        }
        else
        {
            library->fill_tensor_uniform(tensor, 0);
        }
    }

    TensorType compute_target(TensorShape shape)
    {
        // Our test shapes are assumed in NCHW data layout, thus the permutation
        permute(shape, PermutationVector(2U, 0U, 1U));

        // Create a new workload sketch
        CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        GpuWorkloadContext gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
        GpuWorkloadSketch  sketch{ &gpu_ctx };

        // Create sketch tensors
        TensorInfo src_info = sketch.create_tensor_info(TensorInfo(shape, 1, _data_type, _data_layout));
        src_info.set_quantization_info(_input_quantization_info);
        TensorInfo dst_info = sketch.create_tensor_info();

        ResizeAttributes attributes;
        attributes.align_corners(_align_corners).sampling_policy(_sampling_policy).interpolation_policy(_interpolation_policy).output_width(_output_width).output_height(_output_height);

        ITensorInfo *scale_result_info = FunctionType::create_op(sketch, &src_info, attributes);
        GpuOutput::create_op(sketch, scale_result_info, &dst_info);

        // Configure runtime
        ClWorkloadRuntime runtime;
        runtime.configure(sketch);

        // (Important) Allocate auxiliary tensor memory if there are any
        for(auto &data : runtime.get_auxiliary_tensors())
        {
            auto       tensor      = data.first;
            const auto aux_mem_req = data.second;
            tensor->allocator()->init(*data.first->info(), aux_mem_req.alignment);
            tensor->allocator()->allocate();
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

        fill(AccessorType(t_src));

        // Run runtime
        runtime.run({ &t_src, &t_dst });

        return t_dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape)
    {
        // Create reference
        SimpleTensor<T> src{ shape, _data_type, 1, _input_quantization_info };

        // Reference code is NCHW, so the input shapes are NCHW
        const int idx_width  = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::HEIGHT);

        const float scale_x = static_cast<float>(_output_width) / shape[idx_width];
        const float scale_y = static_cast<float>(_output_height) / shape[idx_height];

        // Fill reference
        fill(src);

        return reference::scale<T>(src, scale_x, scale_y, _interpolation_policy,
                                   BorderMode::REPLICATE, static_cast<T>(0), _sampling_policy, /* ceil_policy_scale */ false,
                                   _align_corners, _output_quantization_info);
    }

    TensorType          _target{};
    SimpleTensor<T>     _reference{};
    TensorShape         _shape{};
    InterpolationPolicy _interpolation_policy{};
    SamplingPolicy      _sampling_policy{};
    DataType            _data_type{};
    DataLayout          _data_layout{};
    QuantizationInfo    _input_quantization_info{};
    QuantizationInfo    _output_quantization_info{};
    bool                _align_corners{ false };
    int                 _output_width{ 0 };
    int                 _output_height{ 0 };
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionResizeValidationFixture : public DynamicFusionResizeGenericValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, DataLayout data_layout, InterpolationPolicy policy, SamplingPolicy sampling_policy, bool align_corners)
    {
        DynamicFusionResizeGenericValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape,
                                                                                                      data_type,
                                                                                                      QuantizationInfo(),
                                                                                                      data_layout,
                                                                                                      policy,
                                                                                                      sampling_policy,
                                                                                                      align_corners,
                                                                                                      QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class DynamicFusionResizeQuantizedValidationFixture : public DynamicFusionResizeGenericValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, QuantizationInfo quantization_info, DataLayout data_layout, InterpolationPolicy policy, SamplingPolicy sampling_policy,
               bool align_corners)
    {
        DynamicFusionResizeGenericValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape,
                                                                                                      data_type,
                                                                                                      quantization_info,
                                                                                                      data_layout,
                                                                                                      policy,
                                                                                                      sampling_policy,
                                                                                                      align_corners,
                                                                                                      quantization_info);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif /* TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_RESIZEFIXTURE */
