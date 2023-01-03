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
#ifndef TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_GPU_CL_DIRECTCONV2DFIXTURE
#define TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_GPU_CL_DIRECTCONV2DFIXTURE

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/OperatorAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuConv2d.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"

#include "tests/CL/CLAccessor.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/Permute.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionGpuConv2dValidationGenericFixture : public framework::Fixture
{
public:
    using TBias = typename std::conditional < std::is_same<typename std::decay<T>::type, uint8_t>::value
                  || std::is_same<typename std::decay<T>::type, int8_t>::value,
                  int32_t, T >::type; // If T: uint8_t or int8_t then TBias: int32_t, otherwise TBias: T

    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, const PadStrideInfo &info, const Size2D &dilation, DataType data_type,
               DataLayout data_layout, QuantizationInfo quantization_info, QuantizationInfo weight_quantization_info)
    {
        ARM_COMPUTE_ERROR_ON(data_layout != DataLayout::NHWC); // Dynamic fusion conv2d only supports NHWC layout
        const Conv2dAttributes conv2d_attr = convert_pad_stride_info_to_conv_attr(info, dilation);
        _data_type                         = data_type;
        _data_layout                       = data_layout;
        _is_quantized                      = is_data_type_quantized_asymmetric(data_type);
        _quantization_info                 = quantization_info;
        _weight_quantization_info          = weight_quantization_info;
        _bias_data_type                    = _is_quantized ? DataType::S32 : data_type;
        _target                            = compute_target(input_shape, weights_shape, bias_shape, conv2d_attr);
        _reference                         = compute_reference(input_shape, weights_shape, bias_shape, output_shape, conv2d_attr);
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
    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, const TensorShape &bias_shape, Conv2dAttributes conv2d_attr)
    {
        ARM_COMPUTE_ERROR_ON(_data_layout != DataLayout::NHWC);
        permute(input_shape, PermutationVector(2U, 0U, 1U));
        permute(weights_shape, PermutationVector(2U, 0U, 1U));
        CLScheduler::get().default_reinit();

        // Create a new workload sketch
        auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        auto              gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
        GpuWorkloadSketch sketch{ &gpu_ctx };

        // Create sketch tensors
        TensorInfo input_info  = sketch.create_tensor_info(TensorInfo(input_shape, 1, _data_type, _data_layout));
        TensorInfo weight_info = sketch.create_tensor_info(TensorInfo(weights_shape, 1, _data_type, _data_layout));
        TensorInfo bias_info   = sketch.create_tensor_info(TensorInfo(bias_shape, 1, _data_type, _data_layout));
        TensorInfo dst_info    = sketch.create_tensor_info();

        ITensorInfo *ans_info = FunctionType::create_op(sketch, &input_info, &weight_info, &bias_info, conv2d_attr);
        GpuOutput::create_op(sketch, ans_info, &dst_info);

        // Configure runtime
        ClWorkloadRuntime runtime;
        runtime.configure(sketch);
        // (Important) Allocate auxiliary tensor memory if there are any
        for(auto &data : runtime.get_auxiliary_tensors())
        {
            auto       tensor      = data.first;
            const auto aux_mem_req = data.second;
            tensor->allocator()->init(*data.first->info(), aux_mem_req.alignment);
            tensor->allocator()->allocate(); // Use ACL allocated memory
        }
        // Construct user tensors
        TensorType t_input{};
        TensorType t_weight{};
        TensorType t_bias{};
        TensorType t_dst{};

        // Initialize user tensors
        t_input.allocator()->init(input_info);
        t_weight.allocator()->init(weight_info);
        t_bias.allocator()->init(bias_info);
        t_dst.allocator()->init(dst_info);

        // Allocate and fill user tensors
        t_input.allocator()->allocate();
        t_weight.allocator()->allocate();
        t_bias.allocator()->allocate();
        t_dst.allocator()->allocate();

        fill(AccessorType(t_input), 0);
        fill(AccessorType(t_weight), 1);
        fill(AccessorType(t_bias), 2);

        // Run runtime
        runtime.run({ &t_input, &t_weight, &t_bias, &t_dst });
        return t_dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape,
                                      const TensorShape &output_shape, Conv2dAttributes conv2d_attr)
    {
        // Create reference
        SimpleTensor<T>     src{ input_shape, _data_type, 1, _quantization_info };
        SimpleTensor<T>     weight{ weights_shape, _data_type, 1, _weight_quantization_info };
        SimpleTensor<TBias> bias{ bias_shape, _data_type, 1, _quantization_info };

        fill(src, 0);
        fill(weight, 1);
        fill(bias, 2);

        auto src_nchw          = src;
        auto weights_nchw      = weight;
        auto bias_nchw         = bias;
        auto output_shape_nchw = output_shape;

        PadStrideInfo legacy_pad_stride(conv2d_attr.stride().x(), conv2d_attr.stride().y(), conv2d_attr.pad().left, conv2d_attr.pad().right, conv2d_attr.pad().top, conv2d_attr.pad().bottom,
                                        DimensionRoundingType{});
        auto dst_nchw = reference::convolution_layer(src_nchw, weights_nchw, bias_nchw, output_shape_nchw, legacy_pad_stride, conv2d_attr.dilation());
        return dst_nchw;
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    DataType         _data_type{};
    DataType         _bias_data_type{};
    DataLayout       _data_layout{};
    QuantizationInfo _quantization_info{};
    QuantizationInfo _weight_quantization_info{};
    bool             _is_quantized = false;
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionGpuConv2dValidationFixture : public DynamicFusionGpuConv2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape output_shape, TensorShape bias_shape,
               const PadStrideInfo &info, const Size2D &dialation, DataType data_type, DataLayout data_layout, QuantizationInfo quantization_info)
    {
        DynamicFusionGpuConv2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, weights_shape, output_shape, bias_shape, info, dialation,
                                                                                                         data_type, data_layout, quantization_info, quantization_info);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_GPU_CL_DIRECTCONV2DFIXTURE */
