/*
 * Copyright (c) 2022 Arm Limited.
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

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/OperatorAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuConv2d.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuAdd.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"

#include "tests/CL/CLAccessor.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/dynamic_fusion/Utils.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/ElementwiseOperations.h"

using namespace arm_compute::experimental::dynamic_fusion;
using namespace arm_compute::test::validation::utils;

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(INTEGRATION)
TEST_SUITE(DYNAMIC_FUSION)
TEST_CASE(Conv2d, framework::DatasetMode::ALL)
{
    /* Computation:
     * out = conv2d1x1(direct_conv)(input, weights, bias)
     */
    CLScheduler::get().default_reinit();

    const auto data_type      = DataType::F32;
    const auto data_layout    = DataLayout::NHWC;
    const auto t_input_shape  = TensorShape(384, 12, 12);
    const auto t_weight_shape = TensorShape(384, 1, 1, 16);
    const auto t_dst_shape    = TensorShape(16, 12, 12);

    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &gpu_ctx };

    // Fuse conv2d
    Conv2dAttributes conv2d_attr{};
    auto             input_info  = sketch.create_tensor_info(t_input_shape, 1, data_type, data_layout);
    auto             weight_info = sketch.create_tensor_info(TensorInfo(t_weight_shape, 1, data_type, data_layout));
    auto             ans_info    = sketch.create_tensor_info();
    GpuConv2d::create_op(sketch, &input_info, &weight_info, nullptr, &ans_info, conv2d_attr);

    auto dst_info = sketch.create_tensor_info();
    GpuOutput::create_op(sketch, &ans_info, &dst_info);

    // Configure runtime
    ClWorkloadRuntime runtime;
    runtime.configure(sketch);

    // (Important) Allocate auxiliary tensor memory if there are any
    // Instead of using ACL allocated memory, the user can choose to import memory into the tensors
    for(auto &data : runtime.get_auxiliary_tensors())
    {
        CLTensor     *tensor      = data.first;
        AuxMemoryInfo aux_mem_req = data.second;
        tensor->allocator()->init(*data.first->info(), aux_mem_req.alignment);
        tensor->allocator()->allocate(); // Use ACL allocated memory
        // auto buf = cl::Buffer();
        // tensor->allocator()->import_memory(buf);  // Or, import external memory
    }

    // Construct user tensors
    CLTensor t_input{};
    CLTensor t_weight{};
    CLTensor t_dst{};

    // Initialize user tensors
    t_input.allocator()->init(input_info);
    t_weight.allocator()->init(weight_info);
    t_dst.allocator()->init(dst_info);

    // Allocate and fill user tensors
    // Instead of using ACL allocator, the user can choose to import memory into the tensors
    t_input.allocator()->allocate();
    t_weight.allocator()->allocate();
    t_dst.allocator()->allocate();
    fill<float>(CLAccessor(t_input), 0, library.get());
    fill<float>(CLAccessor(t_weight), 1, library.get());

    // Run runtime
    runtime.run({ &t_input, &t_weight, &t_dst });

    // Create reference
    SimpleTensor<float> ref_t_input{ t_input_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_t_weight{ t_weight_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_t_bias_placeholder{ t_dst_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };

    // Fill reference
    fill<float>(ref_t_input, 0, library.get());
    fill<float>(ref_t_weight, 1, library.get());

    auto ref_t_input_nchw            = reference::permute(ref_t_input, PermutationVector(1U, 2U, 0U));
    auto ref_t_weight_nchw           = reference::permute(ref_t_weight, PermutationVector(1U, 2U, 0U));
    auto ref_t_bias_placeholder_nchw = reference::permute(ref_t_bias_placeholder, PermutationVector(1U, 2U, 0U));
    auto t_dst_shape_nchw            = t_dst_shape;
    permute(t_dst_shape_nchw, PermutationVector(1U, 2U, 0U));

    PadStrideInfo legacy_pad_stride(conv2d_attr.stride().x(), conv2d_attr.stride().y(), conv2d_attr.pad().left, conv2d_attr.pad().right, conv2d_attr.pad().top, conv2d_attr.pad().bottom,
                                    DimensionRoundingType{});
    auto       ref_t_dst_nchw = reference::convolution_layer(ref_t_input_nchw, ref_t_weight_nchw, ref_t_bias_placeholder_nchw, t_dst_shape_nchw, legacy_pad_stride, conv2d_attr.dilation());
    const auto ref_t_dst      = reference::permute(ref_t_dst_nchw, PermutationVector(2U, 0U, 1U));

    RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
    validate(CLAccessor(t_dst), ref_t_dst_nchw, tolerance_f32);
}
TEST_CASE(Add_Output_Add_Output, framework::DatasetMode::ALL)
{
    /* Computation:
     *   out_0 = in_0 + in_1
     *   out_1 = out_0 + in_2
     */
    CLScheduler::get().default_reinit();

    const auto data_type      = DataType::F32;
    const auto t_input_shape  = TensorShape(8, 2, 1);

    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &gpu_ctx };

    auto in_0_info = sketch.create_tensor_info(t_input_shape, 1, data_type);
    auto in_1_info = sketch.create_tensor_info(t_input_shape, 1, data_type);
    auto in_2_info = sketch.create_tensor_info(t_input_shape, 1, data_type);

    auto out_0_info = sketch.create_tensor_info();
    auto out_1_info = sketch.create_tensor_info();

    auto ans_0_info = sketch.create_tensor_info();
    auto ans_1_info = sketch.create_tensor_info();

    GpuAdd::create_op(sketch, &in_0_info, &in_1_info, &ans_0_info);
    GpuOutput::create_op(sketch, &ans_0_info, &out_0_info);
    GpuAdd::create_op(sketch, &ans_0_info, &in_2_info, &ans_1_info);
    GpuOutput::create_op(sketch, &ans_1_info, &out_1_info);

    // Configure runtime
    ClWorkloadRuntime runtime;
    runtime.configure(sketch);

    // (Important) Allocate auxiliary tensor memory if there are any
    // Instead of using ACL allocated memory, the user can choose to import memory into the tensors
    for(auto &data : runtime.get_auxiliary_tensors())
    {
        CLTensor     *tensor      = data.first;
        AuxMemoryInfo aux_mem_req = data.second;
        tensor->allocator()->init(*data.first->info(), aux_mem_req.alignment);
        tensor->allocator()->allocate(); // Use ACL allocated memory
        // auto buf = cl::Buffer();
        // tensor->allocator()->import_memory(buf);  // Or, import external memory
    }

    // Construct user tensors
    CLTensor t_in_0{};
    CLTensor t_in_1{};
    CLTensor t_in_2{};

    CLTensor t_out_0{};
    CLTensor t_out_1{};

    // Initialize user tensors
    t_in_0.allocator()->init(in_0_info);
    t_in_1.allocator()->init(in_1_info);
    t_in_2.allocator()->init(in_2_info);

    t_out_0.allocator()->init(out_0_info);
    t_out_1.allocator()->init(out_1_info);

    // Allocate and fill user tensors
    // Instead of using ACL allocator, the user can choose to import memory into the tensors
    t_in_0.allocator()->allocate();
    t_in_1.allocator()->allocate();
    t_in_2.allocator()->allocate();

    t_out_0.allocator()->allocate();
    t_out_1.allocator()->allocate();

    fill<float>(CLAccessor(t_in_0), 0, library.get());
    fill<float>(CLAccessor(t_in_1), 1, library.get());
    fill<float>(CLAccessor(t_in_2), 2, library.get());

    // Run runtime
    runtime.run({ &t_in_0, &t_in_1, &t_in_2, &t_out_0, &t_out_1 });

    // Create reference
    SimpleTensor<float> ref_t_in_0{ t_input_shape, data_type, 1, QuantizationInfo() };
    SimpleTensor<float> ref_t_in_1{ t_input_shape, data_type, 1, QuantizationInfo() };
    SimpleTensor<float> ref_t_in_2{ t_input_shape, data_type, 1, QuantizationInfo() };

    SimpleTensor<float> ref_t_out_0{ t_input_shape, data_type, 1, QuantizationInfo() };
    SimpleTensor<float> ref_t_out_1{ t_input_shape, data_type, 1, QuantizationInfo() };

    // Fill reference
    fill<float>(ref_t_in_0, 0, library.get());
    fill<float>(ref_t_in_1, 1, library.get());
    fill<float>(ref_t_in_2, 2, library.get());

    reference::arithmetic_operation(ArithmeticOperation::ADD, ref_t_in_0, ref_t_in_1, ref_t_out_0, ConvertPolicy::WRAP);
    reference::arithmetic_operation(ArithmeticOperation::ADD, ref_t_out_0, ref_t_in_2, ref_t_out_1, ConvertPolicy::WRAP);

    RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
    validate(CLAccessor(t_out_0), ref_t_out_0, tolerance_f32);
    validate(CLAccessor(t_out_1), ref_t_out_1, tolerance_f32);
}
TEST_SUITE(Invalid_Fusion_Should_Fail)
TEST_CASE(Multiple_Complex_Ops_0, framework::DatasetMode::ALL)
{
    /* Computation:
     * out = conv2d(conv2d(l0_input, l0_weight), l1_weight)
     */
    CLScheduler::get().default_reinit();

    const auto data_type      = DataType::F32;
    const auto data_layout    = DataLayout::NHWC;
    const auto t_input_shape  = TensorShape(384, 12, 12);
    const auto t_weight_shape = TensorShape(384, 1, 1, 16);
    auto       t_input_info   = TensorInfo(t_input_shape, 1, data_type, data_layout);
    auto       t_weight_info  = TensorInfo(t_weight_shape, 1, data_type, data_layout);
    auto       t_dst_info     = TensorInfo();

    Conv2dAttributes conv2d_attr{};

    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &gpu_ctx };

    // Create tensor infos
    auto input_info  = sketch.create_tensor_info(t_input_shape, 1, data_type, data_layout);
    auto weight_info = sketch.create_tensor_info(TensorInfo(t_weight_shape, 1, data_type, data_layout));
    auto dst_info    = sketch.create_tensor_info();

    // Fuse conv2d into the workload
    {
        // Validate operator
        const auto success = GpuConv2d::validate_op(sketch, &input_info, &weight_info, nullptr, &dst_info, conv2d_attr);
        ARM_COMPUTE_EXPECT(bool(success), framework::LogLevel::ERRORS);

        GpuConv2d::create_op(sketch, &input_info, &weight_info, nullptr, &dst_info, conv2d_attr);
    }

    // Create tensor infos
    auto weight_info_2 = sketch.create_tensor_info(t_weight_info);
    auto dst_info_2    = sketch.create_tensor_info();

    // Fuse conv2d into the workload
    {
        // Validate operator, should fail
        const auto success = GpuConv2d::validate_op(sketch, &dst_info, &weight_info_2, nullptr, &dst_info_2, conv2d_attr);
        ARM_COMPUTE_EXPECT(!bool(success), framework::LogLevel::ERRORS);
    }
}
TEST_SUITE_END() // Invalid_Fusion_Should_Fail
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // INTEGRATION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
