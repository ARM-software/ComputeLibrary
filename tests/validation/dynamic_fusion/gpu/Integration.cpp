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

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/CastAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/Conv2dAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/DepthwiseConv2dAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuAdd.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuCast.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuConv2d.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuDepthwiseConv2d.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuMul.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"

#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuSigmoid.h"
#include "tests/CL/CLAccessor.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/dynamic_fusion/Utils.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/DepthConvertLayer.h"
#include "tests/validation/reference/DepthwiseConvolutionLayer.h"
#include "tests/validation/reference/ElementwiseOperations.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"

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
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF // Conv2d is not ported to ckw yet. COMPMID-6259
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
    auto              context        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &context };

    // Fuse conv2d
    Conv2dAttributes conv2d_attr{};
    TensorInfo       input_info  = context.create_tensor_info(t_input_shape, 1, data_type, data_layout);
    TensorInfo       weight_info = context.create_tensor_info(TensorInfo(t_weight_shape, 1, data_type, data_layout));

    ITensorInfo *conv_out_info = GpuConv2d::create_op(sketch, &input_info, &weight_info, nullptr, conv2d_attr);

    TensorInfo dst_info = context.create_tensor_info();
    GpuOutput::create_op(sketch, conv_out_info, &dst_info);

    // Configure runtime
    ClWorkloadRuntime runtime;
    runtime.configure(sketch);

    // (Important) Allocate auxiliary tensor memory if there are any
    // Instead of using ACL allocated memory, the user can choose to import memory into the tensors
    for(auto &data : runtime.get_auxiliary_tensors())
    {
        CLTensor     *tensor      = std::get<0>(data);
        TensorInfo    info        = std::get<1>(data);
        AuxMemoryInfo aux_mem_req = std::get<2>(data);
        tensor->allocator()->init(info, aux_mem_req.alignment);
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
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
TEST_CASE(Add_Output_Add_Output, framework::DatasetMode::ALL)
{
    /* Computation:
     *   out_0 = in_0 + in_1
     *   out_1 = out_0 + in_2
     */
    CLScheduler::get().default_reinit();

    const auto data_type     = DataType::F32;
    const auto t_input_shape = TensorShape(33, 3, 2);

    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              context        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &context };

    TensorInfo in_0_info = context.create_tensor_info(t_input_shape, 1, data_type);
    TensorInfo in_1_info = context.create_tensor_info(t_input_shape, 1, data_type);
    TensorInfo in_2_info = context.create_tensor_info(t_input_shape, 1, data_type);

    TensorInfo out_0_info = context.create_tensor_info();
    TensorInfo out_1_info = context.create_tensor_info();

    ITensorInfo *ans_0_info = GpuAdd::create_op(sketch, &in_0_info, &in_1_info);
    GpuOutput::create_op(sketch, ans_0_info, &out_0_info);
    ITensorInfo *ans_1_info = GpuAdd::create_op(sketch, ans_0_info, &in_2_info);
    GpuOutput::create_op(sketch, ans_1_info, &out_1_info);

    // Configure runtime
    ClWorkloadRuntime runtime;
    runtime.configure(sketch);

    // (Important) Allocate auxiliary tensor memory if there are any
    // Instead of using ACL allocated memory, the user can choose to import memory into the tensors
    for(auto &data : runtime.get_auxiliary_tensors())
    {
        CLTensor     *tensor      = std::get<0>(data);
        TensorInfo    info        = std::get<1>(data);
        AuxMemoryInfo aux_mem_req = std::get<2>(data);
        tensor->allocator()->init(info, aux_mem_req.alignment);
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
TEST_CASE(Add_Output_Add_Cast_Cast_Output, framework::DatasetMode::ALL)
{
    /* Computation:
     *   out_0 = in_0 + in_1
     *   out_1 = float(int32_t(out_0 + in_2))
     */
    CLScheduler::get().default_reinit();

    const auto data_type     = DataType::F32;
    const auto t_input_shape = TensorShape(3, 8, 5);

    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              context        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &context };

    TensorInfo in_0_info = context.create_tensor_info(t_input_shape, 1, data_type);
    TensorInfo in_1_info = context.create_tensor_info(t_input_shape, 1, data_type);
    TensorInfo in_2_info = context.create_tensor_info(t_input_shape, 1, data_type);

    TensorInfo out_0_info = context.create_tensor_info();
    TensorInfo out_1_info = context.create_tensor_info();

    CastAttributes cast_0_attr;
    cast_0_attr.data_type(DataType::S32).convert_policy(ConvertPolicy::SATURATE);

    CastAttributes cast_1_attr;
    cast_1_attr.data_type(DataType::F32).convert_policy(ConvertPolicy::SATURATE);

    ITensorInfo *ans_0_info = GpuAdd::create_op(sketch, &in_0_info, &in_1_info);
    GpuOutput::create_op(sketch, ans_0_info, &out_0_info);
    ITensorInfo *ans_1_info = GpuAdd::create_op(sketch, ans_0_info, &in_2_info);
    ITensorInfo *ans_2_info = GpuCast::create_op(sketch, ans_1_info, cast_0_attr);
    ITensorInfo *ans_3_info = GpuCast::create_op(sketch, ans_2_info, cast_1_attr);
    GpuOutput::create_op(sketch, ans_3_info, &out_1_info);

    // Configure runtime
    ClWorkloadRuntime runtime;
    runtime.configure(sketch);

    // (Important) Allocate auxiliary tensor memory if there are any
    // Instead of using ACL allocated memory, the user can choose to import memory into the tensors
    for(auto &data : runtime.get_auxiliary_tensors())
    {
        CLTensor     *tensor      = std::get<0>(data);
        TensorInfo    info        = std::get<1>(data);
        AuxMemoryInfo aux_mem_req = std::get<2>(data);
        tensor->allocator()->init(info, aux_mem_req.alignment);
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
    SimpleTensor<float> ref_t_ans_1{ t_input_shape, data_type, 1, QuantizationInfo() };

    // Fill reference
    fill<float>(ref_t_in_0, 0, library.get());
    fill<float>(ref_t_in_1, 1, library.get());
    fill<float>(ref_t_in_2, 2, library.get());

    reference::arithmetic_operation(ArithmeticOperation::ADD, ref_t_in_0, ref_t_in_1, ref_t_out_0, ConvertPolicy::WRAP);
    reference::arithmetic_operation(ArithmeticOperation::ADD, ref_t_out_0, ref_t_in_2, ref_t_ans_1, ConvertPolicy::WRAP);
    const auto ref_t_ans_2 = reference::depth_convert<float, int32_t>(ref_t_ans_1, DataType::S32, ConvertPolicy::SATURATE, 0);
    const auto ref_t_out_1 = reference::depth_convert<int32_t, float>(ref_t_ans_2, DataType::F32, ConvertPolicy::SATURATE, 0);

    RelativeTolerance<float> tolerance_add_f32(0.001f);
    AbsoluteTolerance<float> tolerance_cast_f32(1.0f);
    validate(CLAccessor(t_out_0), ref_t_out_0, tolerance_add_f32);
    validate(CLAccessor(t_out_1), ref_t_out_1, tolerance_cast_f32);
}

#ifndef ACL_INTERNAL_TEST_CKW_IN_DF // Conv2d is not ported to ckw yet. COMPMID-6259
TEST_CASE(Conv2d_Sigmoid_DepthwiseConv2d_Mul, framework::DatasetMode::ALL)
{
    //   (tensor0)
    //       |
    // ======|============================================== Sketch 0
    //       |     (tensor1)     +---- (tensor2)
    //       |         |         |         |
    // +-- input -- weights -- biases --+  |
    // |                                |  |
    // |            Conv2d              |  |
    // |                                |  |
    // +----------- output -------------+  |
    //                |                    |
    //          +-- input ---+             |
    //          |            |             |
    //          |  Sigmoid   |             |
    //          |            |             |
    //          +-- output --+             |
    //                |                    |
    //          +-- input ---+             |
    //          |            |             |
    //          |   Output   |             |
    //          |            |             |
    //          +-- output --+             |
    //                |                    |
    //            (tensor5)                |
    //                |                    |
    //       +--------+                    |
    // ======|=============================|================ Sketch 1
    //       |     (tensor3) (tensor4)     |
    //       |         |         |         |
    // +-- input -- weights -- biases --+  |
    // |                                |  |
    // |        DepthwiseConv2d         |  |
    // |                                |  |
    // +----------- output -------------+  |
    //                |                    |
    //             +--+   +----------------+
    //             |      |
    //        +-- lhs -- rhs --+
    //        |                |
    //        |    Multiply    |
    //        |                |
    //        +---- output ----+
    //                |
    //          +-- input ---+
    //          |            |
    //          |   Output   |
    //          |            |
    //          +-- output --+
    //                |
    //            (tensor6)

    TensorShape conv2d_src_shape(10, 20, 30);
    TensorShape conv2d_wei_shape(10, 3, 3, 5);
    TensorShape conv2d_bia_shape(5);
    TensorShape conv2d_dst_shape(5, 18, 28);
    TensorShape dwc_wei_shape(5, 3, 3);
    TensorShape dwc_bia_shape(5);
    TensorShape dwc_dst_shape(5, 16, 26);

    // Initialize the context.
    CLScheduler::get().default_reinit();

    auto               cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    GpuWorkloadContext context(&cl_compile_ctx);

    auto tensor0_info = context.create_tensor_info(conv2d_src_shape, 1, DataType::F32, DataLayout::NHWC);

    // Create the first sketch: conv2d + cast + output.
    GpuWorkloadSketch sketch0(&context);

    Conv2dAttributes conv2d_attr;
    auto             tensor1_info = context.create_tensor_info(conv2d_wei_shape, 1, DataType::F32, DataLayout::NHWC);
    auto             tensor2_info = context.create_tensor_info(conv2d_bia_shape, 1, DataType::F32, DataLayout::NHWC);
    ARM_COMPUTE_EXPECT(GpuConv2d::validate_op(sketch0, &tensor0_info, &tensor1_info, &tensor2_info, conv2d_attr), framework::LogLevel::ERRORS);
    auto ans_info = GpuConv2d::create_op(sketch0, &tensor0_info, &tensor1_info, &tensor2_info, conv2d_attr);

    ARM_COMPUTE_EXPECT(GpuSigmoid::validate_op(sketch0, ans_info), framework::LogLevel::ERRORS);
    ans_info = GpuSigmoid::create_op(sketch0, ans_info);

    DepthwiseConv2dAttributes dwc_attr;
    auto                      tensor3_info = context.create_tensor_info(dwc_wei_shape, 1, DataType::F32, DataLayout::NHWC);
    auto                      tensor4_info = context.create_tensor_info(dwc_bia_shape, 1, DataType::F32, DataLayout::NHWC);
    ARM_COMPUTE_EXPECT(!GpuDepthwiseConv2d::validate_op(sketch0, ans_info, &tensor3_info, &tensor4_info, dwc_attr), framework::LogLevel::ERRORS);

    auto tensor5_info = context.create_tensor_info();
    ARM_COMPUTE_EXPECT(GpuOutput::validate_op(sketch0, ans_info, &tensor5_info), framework::LogLevel::ERRORS);
    GpuOutput::create_op(sketch0, ans_info, &tensor5_info);

    // Create the first workload runtime.
    ClWorkloadRuntime runtime0;
    runtime0.configure(sketch0);

    // Create the second sketch: dwc + sigmoid + output.
    GpuWorkloadSketch sketch1(&context);

    ARM_COMPUTE_EXPECT(GpuDepthwiseConv2d::validate_op(sketch1, &tensor5_info, &tensor3_info, &tensor4_info, dwc_attr), framework::LogLevel::ERRORS);
    ans_info = GpuDepthwiseConv2d::create_op(sketch1, &tensor5_info, &tensor3_info, &tensor4_info, dwc_attr);

    ARM_COMPUTE_EXPECT(GpuMul::validate_op(sketch1, ans_info, &tensor2_info), framework::LogLevel::ERRORS);
    ans_info = GpuMul::create_op(sketch1, ans_info, &tensor2_info);

    auto tensor6_info = context.create_tensor_info();
    ARM_COMPUTE_EXPECT(GpuOutput::validate_op(sketch1, ans_info, &tensor6_info), framework::LogLevel::ERRORS);
    GpuOutput::create_op(sketch1, ans_info, &tensor6_info);

    // Create the second workload runtime.
    ClWorkloadRuntime runtime1;
    runtime1.configure(sketch1);

    // Create the user tensors.
    CLTensor tensor0;
    CLTensor tensor1;
    CLTensor tensor2;
    CLTensor tensor3;
    CLTensor tensor4;
    CLTensor tensor5;
    CLTensor tensor6;

    tensor0.allocator()->init(tensor0_info);
    tensor1.allocator()->init(tensor1_info);
    tensor2.allocator()->init(tensor2_info);
    tensor3.allocator()->init(tensor3_info);
    tensor4.allocator()->init(tensor4_info);
    tensor5.allocator()->init(tensor5_info);
    tensor6.allocator()->init(tensor6_info);

    tensor0.allocator()->allocate();
    tensor1.allocator()->allocate();
    tensor2.allocator()->allocate();
    tensor3.allocator()->allocate();
    tensor4.allocator()->allocate();
    tensor5.allocator()->allocate();
    tensor6.allocator()->allocate();

    // Allocate the auxiliary tensors.
    for(auto &data : runtime0.get_auxiliary_tensors())
    {
        auto  tensor      = std::get<0>(data);
        auto &tensor_info = std::get<1>(data);
        auto  mem_req     = std::get<2>(data);

        tensor->allocator()->init(tensor_info, mem_req.alignment);
        tensor->allocator()->allocate();
    }

    for(auto &data : runtime1.get_auxiliary_tensors())
    {
        auto  tensor      = std::get<0>(data);
        auto &tensor_info = std::get<1>(data);
        auto  mem_req     = std::get<2>(data);

        tensor->allocator()->init(tensor_info, mem_req.alignment);
        tensor->allocator()->allocate();
    }

    // Fill the input tensors with random data.
    fill<float>(CLAccessor(tensor0), 0, library.get());
    fill<float>(CLAccessor(tensor1), 1, library.get());
    fill<float>(CLAccessor(tensor2), 2, library.get());
    fill<float>(CLAccessor(tensor3), 3, library.get());
    fill<float>(CLAccessor(tensor4), 4, library.get());

    // Run each runtime.
    runtime0.run({ &tensor0, &tensor1, &tensor2, &tensor5 });
    runtime1.run({ &tensor5, &tensor3, &tensor4, &tensor2, &tensor6 });

    // Compute the reference result.
    SimpleTensor<float> ref_conv2d_src(conv2d_src_shape, DataType::F32, 1, QuantizationInfo(), DataLayout::NHWC);
    SimpleTensor<float> ref_conv2d_wei(conv2d_wei_shape, DataType::F32, 1, QuantizationInfo(), DataLayout::NHWC);
    SimpleTensor<float> ref_conv2d_bia(conv2d_bia_shape, DataType::F32, 1, QuantizationInfo(), DataLayout::NHWC);
    SimpleTensor<float> ref_dwc_wei(dwc_wei_shape, DataType::F32, 1, QuantizationInfo(), DataLayout::NHWC);
    SimpleTensor<float> ref_dwc_bia(dwc_bia_shape, DataType::F32, 1, QuantizationInfo(), DataLayout::NHWC);

    fill<float>(ref_conv2d_src, 0, library.get());
    fill<float>(ref_conv2d_wei, 1, library.get());
    fill<float>(ref_conv2d_bia, 2, library.get());
    fill<float>(ref_dwc_wei, 3, library.get());
    fill<float>(ref_dwc_bia, 4, library.get());

    PermutationVector nhwc_to_nchw(1, 2, 0);

    auto conv2d_dst_shape_nchw = conv2d_dst_shape;
    permute(conv2d_dst_shape_nchw, nhwc_to_nchw);
    const auto ref_conv2d_src_nchw = reference::permute(ref_conv2d_src, nhwc_to_nchw);
    const auto ref_conv2d_wei_nchw = reference::permute(ref_conv2d_wei, nhwc_to_nchw);
    const auto ref_conv2d_bia_nchw = reference::permute(ref_conv2d_bia, nhwc_to_nchw);
    const auto ref_conv2d_dst_nchw = reference::convolution_layer(ref_conv2d_src_nchw, ref_conv2d_wei_nchw, ref_conv2d_bia_nchw, conv2d_dst_shape_nchw, PadStrideInfo());

    const auto ref_sigmoid_dst_nchw = reference::activation_layer(ref_conv2d_dst_nchw, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));

    auto dwc_dst_shape_nchw = dwc_dst_shape;
    permute(dwc_dst_shape_nchw, nhwc_to_nchw);
    const auto ref_dwc_wei_nchw = reference::permute(ref_dwc_wei, nhwc_to_nchw);
    const auto ref_dwc_bia_nchw = reference::permute(ref_dwc_bia, nhwc_to_nchw);
    const auto ref_dwc_dst_nchw = reference::depthwise_convolution(ref_sigmoid_dst_nchw, ref_dwc_wei_nchw, ref_dwc_bia_nchw, dwc_dst_shape_nchw, PadStrideInfo(), 1);

    const auto ref_mul_dst_nchw = reference::pixel_wise_multiplication<float, float, float>(ref_dwc_dst_nchw, ref_conv2d_bia_nchw, 1.0, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_UP,
                                                                                            DataType::F32);

    constexpr RelativeTolerance<float> tolerance(0.001f);
    validate(CLAccessor(tensor6), ref_mul_dst_nchw, tolerance);
}
#endif // ACL_INTERNAL_TEST_CKW_IN_DF

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
    auto              context        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &context };

    // Create tensor infos
    TensorInfo   input_info  = context.create_tensor_info(t_input_shape, 1, data_type, data_layout);
    TensorInfo   weight_info = context.create_tensor_info(TensorInfo(t_weight_shape, 1, data_type, data_layout));
    ITensorInfo *dst_info;

    // Fuse conv2d into the workload
    {
        // Validate operator
        const Status success = GpuConv2d::validate_op(sketch, &input_info, &weight_info, nullptr, conv2d_attr);
        ARM_COMPUTE_EXPECT(bool(success), framework::LogLevel::ERRORS);

        dst_info = GpuConv2d::create_op(sketch, &input_info, &weight_info, nullptr, conv2d_attr);
    }

    // Create tensor infos
    TensorInfo weight_info_2 = context.create_tensor_info(t_weight_info);

    // Fuse conv2d into the workload
    {
        // Validate operator, should fail
        const Status success            = GpuConv2d::validate_op(sketch, dst_info, &weight_info_2, nullptr, conv2d_attr);
        const auto   expected_error_str = "Operator fusion test failed. This operator cannot be fused into the workload";

        ARM_COMPUTE_EXPECT(!bool(success), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT((success.error_description().find(expected_error_str) != std::string::npos), framework::LogLevel::ERRORS);
    }
}
TEST_SUITE_END() // Invalid_Fusion_Should_Fail
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // INTEGRATION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
