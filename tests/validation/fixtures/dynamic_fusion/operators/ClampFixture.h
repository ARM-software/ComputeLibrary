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
#ifndef TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_CLAMPFIXTURE
#define TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_CLAMPFIXTURE

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"

#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ActivationLayer.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionClampValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, ClampAttributes attributes, bool fuse, DataType data_type)
    {
        // CLAMP is implemented as LU_BOUNDED_RELU with the alpha and beta variables swapped.
        ActivationLayerInfo act_info{ ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, attributes.max_val(), attributes.min_val() };

        _fuse       = fuse;
        _attributes = attributes;
        _data_type  = data_type;
        _target     = compute_target(shape, attributes);
        _reference  = compute_reference(shape, act_info);
    }

protected:
    std::vector<T> get_boundary_values(T min, T max)
    {
        // This function will return a vector filled with the following values that can
        // represent two partitions derived from equivalent partitioning.
        // * Lower partition: min, min + delta, lower quarter (nominal), center - delta
        // * Upper partition: center, center + delta, upper quarter (nominal), max - delta, max
        const auto delta         = is_data_type_float(_data_type) ? T(0.1f) : T(1);
        const auto center_value  = (min + max) / 2;
        const auto lower_quarter = (min + center_value) / 2;
        const auto upper_quarter = (center_value + max) / 2;

        std::vector<T> boundary_values{};

        // To ensure all the inserted values are within the given range after subtracing/adding delta
        auto insert_values = [&boundary_values, &min, &max](const std::initializer_list<T> &new_values)
        {
            for(auto &v : new_values)
            {
                if(v >= min && v <= max)
                {
                    boundary_values.emplace_back(v);
                }
            }
        };

        insert_values({ min, static_cast<T>(min + delta), static_cast<T>(lower_quarter), static_cast<T>(center_value - delta) });                               // lower partition
        insert_values({ static_cast<T>(center_value), static_cast<T>(center_value + delta), static_cast<T>(upper_quarter), static_cast<T>(max - delta), max }); // upper partition

        return boundary_values;
    }

    template <typename U>
    void fill(U &&tensor)
    {
        float min_bound = 0;
        float max_bound = 0;
        std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<T>(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, _data_type);
        library->fill_static_values(tensor, get_boundary_values(static_cast<T>(min_bound), static_cast<T>(max_bound)));
    }

    TensorType compute_target(const TensorShape &shape, ClampAttributes attributes)
    {
        // Create a new workload sketch
        CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        GpuWorkloadContext context{ &cl_compile_ctx };
        GpuWorkloadSketch  sketch{ &context };

        // Create sketch tensors
        TensorInfo src_info = context.create_tensor_info(TensorInfo(shape, 1, _data_type));
        TensorInfo dst_info = context.create_tensor_info(TensorInfo(shape, 1, _data_type));

        ITensorInfo *ans_0_info = FunctionType::create_op(sketch, &src_info, attributes);
        if(_fuse)
        {
            ITensorInfo *ans_1_info = FunctionType::create_op(sketch, ans_0_info, attributes);
            GpuOutput::create_op(sketch, ans_1_info, &dst_info);
        }
        else
        {
            GpuOutput::create_op(sketch, ans_0_info, &dst_info);
        }

        // Configure runtime
        ClWorkloadRuntime runtime;
        runtime.configure(sketch);

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

    SimpleTensor<T> compute_reference(const TensorShape &shape, ActivationLayerInfo act_info)
    {
        // Create reference
        SimpleTensor<T> src{ shape, _data_type, 1, _quantization_info };

        // Fill reference
        fill(src);

        auto dst = reference::activation_layer<T>(src, act_info, _quantization_info);
        return dst;
    }

protected:
    QuantizationInfo _quantization_info{};
    ClampAttributes  _attributes{};
    bool             _fuse{ false };
    DataType         _data_type{};
    TensorType       _target{};
    SimpleTensor<T>  _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_CLAMPFIXTURE */
