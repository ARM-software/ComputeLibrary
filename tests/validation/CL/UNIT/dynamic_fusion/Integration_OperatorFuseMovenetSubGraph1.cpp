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

#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#include "arm_compute/core/TensorInfo.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/experimental/ClWorkload.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/experimental/ClCompositeOperator.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClKernelDescriptors.h"
#include "src/gpu/cl/operators/ClAdd.h"
#include "src/gpu/cl/operators/ClConv2d.h"
#include "tests/CL/CLAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/CL/UNIT/dynamic_fusion/Utils.h"
#include "tests/validation/Validation.h"

#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/ElementwiseOperations.h"
#include "tests/validation/reference/Permute.h"

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
#include "tests/SimpleTensorPrinter.h"
#endif /* ARM_COMPUTE_ASSERTS_ENABLED */

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
TEST_CASE(Operator_Fuse_Movenet_SubGraph_1_F32, framework::DatasetMode::ALL)
{
    // Please refer to: https://confluence.arm.com/pages/viewpage.action?pageId=886243697
    /* Computation:
     * out = add_desc(addend, conv2d1x1(direct_conv)(input, weights, bias))
     */
    const auto data_type     = DataType::F32;
    const auto data_layout   = DataLayout::NHWC;
    const auto t_input_shape = TensorShape(384, 12, 12);
    // const auto t_weight_shape   = TensorShape(384, 1, 1, 64);
    // const auto t_dst_shape      = TensorShape(64, 12, 12);
    const auto t_weight_shape   = TensorShape(384, 1, 1, 16);
    const auto t_dst_shape      = TensorShape(16, 12, 12);
    auto       t_input_info     = TensorInfo(t_input_shape, 1, data_type, data_layout);
    auto       t_weight_info    = TensorInfo(t_weight_shape, 1, data_type, data_layout);
    auto       t_l1_addend_info = TensorInfo(t_dst_shape, 1, data_type, data_layout);
    auto       t_acc_info       = TensorInfo(); // Intermediate tensor for cond3
    auto       t_dst_info       = TensorInfo();

    Conv2dDescriptor conv2d_desc{};
    AddDescriptor    add_desc{};

    // Create reference
    SimpleTensor<float> ref_t_input{ t_input_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_t_weight{ t_weight_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_t_bias_placeholder{ t_dst_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_t_l1_addend{ t_dst_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };

    // Fill reference
    fill<float>(ref_t_input, 0, library.get());
    fill<float>(ref_t_weight, 1, library.get());
    fill<float>(ref_t_l1_addend, 2, library.get());

    auto ref_t_input_nchw            = reference::permute(ref_t_input, PermutationVector(1U, 2U, 0U));
    auto ref_t_weight_nchw           = reference::permute(ref_t_weight, PermutationVector(1U, 2U, 0U));
    auto ref_t_bias_placeholder_nchw = reference::permute(ref_t_bias_placeholder, PermutationVector(1U, 2U, 0U));
    auto ref_t_l1_addend_nchw        = reference::permute(ref_t_l1_addend, PermutationVector(1U, 2U, 0U));
    auto t_dst_shape_nchw            = t_dst_shape;
    permute(t_dst_shape_nchw, PermutationVector(1U, 2U, 0U));

    PadStrideInfo legacy_pad_stride(conv2d_desc.stride.x(), conv2d_desc.stride.y(), conv2d_desc.pad.left, conv2d_desc.pad.right, conv2d_desc.pad.top, conv2d_desc.pad.bottom, DimensionRoundingType{});
    auto          ref_t_dst_nchw = reference::arithmetic_operation(
                                       ArithmeticOperation::ADD,
                                       ref_t_l1_addend_nchw,
                                       reference::convolution_layer(ref_t_input_nchw, ref_t_weight_nchw, ref_t_bias_placeholder_nchw, t_dst_shape_nchw, legacy_pad_stride, conv2d_desc.dilation),
                                       data_type,
                                       ConvertPolicy{});
    const auto ref_t_dst = reference::permute(ref_t_dst_nchw, PermutationVector(2U, 0U, 1U));

    CLScheduler::get().default_reinit();
    const auto    cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    OperatorGraph op_graph;

    const auto op_t_input     = add_tensor(op_graph, t_input_info);
    const auto op_t_weight    = add_tensor(op_graph, t_weight_info);
    const auto op_t_l1_addend = add_tensor(op_graph, t_l1_addend_info);
    const auto op_t_acc       = add_tensor(op_graph, t_acc_info); // temp accumulator; TensorInfo to be inferred
    const auto op_t_dst       = add_tensor(op_graph, t_dst_info);

    auto conv2d = add_op_conv2d(op_graph, conv2d_desc, op_t_input, op_t_weight, op_t_acc);
    force_conv2d_method(op_graph, conv2d, ConvolutionMethod::DIRECT);
    add_op_elementwise_add(op_graph, add_desc, op_t_acc, op_t_l1_addend, op_t_dst);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    build(workload, op_graph, workload_ctx);

    ClCompositeOperator op;
    op.configure(cl_compile_ctx, workload);

    // Construct tensors
    CLTensor t_input{};
    CLTensor t_weight{};
    CLTensor t_l1_addend{};
    CLTensor t_dst{};

    // Init tensors
    t_input.allocator()->init(t_input_info);
    t_weight.allocator()->init(t_weight_info);
    t_l1_addend.allocator()->init(t_dst_info);
    t_dst.allocator()->init(t_dst_info);

    // Allocate and fill tensors
    t_input.allocator()->allocate();
    t_weight.allocator()->allocate();
    t_l1_addend.allocator()->allocate();
    t_dst.allocator()->allocate();
    fill<float>(CLAccessor(t_input), 0, library.get());
    fill<float>(CLAccessor(t_weight), 1, library.get());
    fill<float>(CLAccessor(t_l1_addend), 2, library.get());
    // "Pack" tensors
    OpTensorBinding bp_tensors({ { op_t_input, &t_input },
        { op_t_weight, &t_weight },
        { op_t_l1_addend, &t_l1_addend },
        { op_t_dst, &t_dst }
    });

    // Populate prepare and run pack-maps (including allocating aux tensors)
    ClAuxTensorData aux_tensor_data{};
    TensorPackMap   prepare_pack_map{};
    TensorPackMap   run_pack_map{};
    bind_tensors(aux_tensor_data, prepare_pack_map, run_pack_map, workload, bp_tensors);

    op.prepare(prepare_pack_map);
    op.run(run_pack_map);
    RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
    validate(CLAccessor(t_dst), ref_t_dst_nchw, tolerance_f32);
}
TEST_SUITE(Unsupported)
TEST_CASE(DataType_QASYMM8, framework::DatasetMode::ALL)
{
    const auto data_type        = DataType::QASYMM8;
    const auto data_layout      = DataLayout::NHWC;
    const auto t_input_shape    = TensorShape(384, 12, 12);
    const auto t_weight_shape   = TensorShape(384, 1, 1, 64);
    const auto t_dst_shape      = TensorShape(64, 12, 12);
    auto       t_input_info     = TensorInfo(t_input_shape, 1, data_type, data_layout);
    auto       t_weight_info    = TensorInfo(t_weight_shape, 1, data_type, data_layout);
    auto       t_l1_addend_info = TensorInfo(t_dst_shape, 1, data_type, data_layout);
    auto       t_acc_info       = TensorInfo(t_dst_shape, 1, data_type, data_layout);
    auto       t_dst_info       = TensorInfo(t_dst_shape, 1, data_type, data_layout);

    Conv2dDescriptor conv2d_desc{};
    AddDescriptor    add_desc{};

    OperatorGraph op_graph;

    const auto op_t_input     = add_tensor(op_graph, t_input_info);
    const auto op_t_weight    = add_tensor(op_graph, t_weight_info);
    const auto op_t_l1_addend = add_tensor(op_graph, t_l1_addend_info);
    const auto op_t_acc       = add_tensor(op_graph, t_acc_info); // temp accumulator; TensorInfo to be inferred
    const auto op_t_dst       = add_tensor(op_graph, t_dst_info);

    auto conv2d = add_op_conv2d(op_graph, conv2d_desc, op_t_input, op_t_weight, op_t_acc);
    add_op_elementwise_add(op_graph, add_desc, op_t_acc, op_t_l1_addend, op_t_dst);
    force_conv2d_method(op_graph, conv2d, ConvolutionMethod::DIRECT);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    const auto              success = build(workload, op_graph, workload_ctx);

    ARM_COMPUTE_EXPECT(!bool(success), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!bool(ClCompositeOperator::validate(workload)), framework::LogLevel::ERRORS);
}
TEST_CASE(DataLayout_NCHW, framework::DatasetMode::ALL)
{
    const auto data_type      = DataType::F32;
    const auto data_layout    = DataLayout::NCHW;
    const auto t_input_shape  = TensorShape(384, 12, 12);
    const auto t_weight_shape = TensorShape(384, 1, 1, 64);
    const auto t_dst_shape    = TensorShape(64, 12, 12);
    auto       t_input_info   = TensorInfo(t_input_shape, 1, data_type, data_layout);
    auto       t_weight_info  = TensorInfo(t_weight_shape, 1, data_type, data_layout);
    auto       t_dst_info     = TensorInfo(t_dst_shape, 1, data_type, data_layout);

    Conv2dDescriptor conv2d_desc{};

    OperatorGraph op_graph;

    const auto op_t_input  = add_tensor(op_graph, t_input_info);
    const auto op_t_weight = add_tensor(op_graph, t_weight_info);
    const auto op_t_dst    = add_tensor(op_graph, t_dst_info);

    auto conv2d = add_op_conv2d(op_graph, conv2d_desc, op_t_input, op_t_weight, op_t_dst);
    force_conv2d_method(op_graph, conv2d, ConvolutionMethod::DIRECT);
    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    const auto              success = build(workload, op_graph, workload_ctx);

    ARM_COMPUTE_EXPECT(!bool(success), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!bool(ClCompositeOperator::validate(workload)), framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Unsupported

TEST_SUITE(Invalid)
TEST_CASE(Multiple_Complex_Ops_0, framework::DatasetMode::ALL)
{
    /* Computation:
     * out = conv2d(conv2d(l0_input, l0_weight), l1_weight)
     */
    const auto data_type         = DataType::F32;
    const auto data_layout       = DataLayout::NHWC;
    const auto t_l0_input_shape  = TensorShape(1024, 56, 56);
    const auto t_l0_weight_shape = TensorShape(512, 1024, 1, 1);
    const auto t_l1_weight_shape = TensorShape(512, 256, 1, 1);

    auto t_l0_input_info  = TensorInfo(t_l0_input_shape, 1, data_type, data_layout);
    auto t_l0_weight_info = TensorInfo(t_l0_weight_shape, 1, data_type, data_layout);
    auto t_l1_weight_info = TensorInfo(t_l1_weight_shape, 1, data_type, data_layout);
    auto t_l0_dst_info    = TensorInfo();
    auto t_dst_info       = TensorInfo();

    OperatorGraph op_graph;
    const auto    conv2d_desc = Conv2dDescriptor{};

    const auto op_t_l0_input  = add_tensor(op_graph, t_l0_input_info);
    const auto op_t_l0_weight = add_tensor(op_graph, t_l0_weight_info);
    const auto op_t_l1_weight = add_tensor(op_graph, t_l1_weight_info);
    const auto op_t_l0_dst    = add_tensor(op_graph, t_l0_dst_info); // temp accumulator; TensorInfo to be inferred
    const auto op_t_dst       = add_tensor(op_graph, t_dst_info);

    add_op_conv2d(op_graph, conv2d_desc, op_t_l0_input, op_t_l0_weight, op_t_l0_dst);
    add_op_conv2d(op_graph, conv2d_desc, op_t_l0_dst, op_t_l1_weight, op_t_dst);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    const auto              success = build(workload, op_graph, workload_ctx);

    ARM_COMPUTE_EXPECT(!bool(success), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!bool(ClCompositeOperator::validate(workload)), framework::LogLevel::ERRORS);
}
TEST_CASE(Enlarging_Execution_Space, framework::DatasetMode::ALL)
{
    /* Computation:
     * out = add(l2_lhs, add(add(l0_lhs, l0_rhs), l1_rhs))
     */
    const auto data_type      = DataType::F32;
    const auto data_layout    = DataLayout::NHWC;
    const auto t_l0_lhs_shape = TensorShape(1, 256, 3);
    const auto t_l0_rhs_shape = TensorShape(1, 256, 3);
    const auto t_l1_rhs_shape = TensorShape(1, 1, 3);
    const auto t_l2_lhs_shape = TensorShape(1024, 1, 3);

    auto t_l0_lhs_info = TensorInfo(t_l0_lhs_shape, 1, data_type, data_layout);
    auto t_l0_rhs_info = TensorInfo(t_l0_rhs_shape, 1, data_type, data_layout);
    auto t_l1_rhs_info = TensorInfo(t_l1_rhs_shape, 1, data_type, data_layout);
    auto t_l2_lhs_info = TensorInfo(t_l2_lhs_shape, 1, data_type, data_layout);
    auto t_l0_dst_info = TensorInfo();
    auto t_l1_dst_info = TensorInfo();
    auto t_dst_info    = TensorInfo();

    OperatorGraph op_graph;
    const auto    add_desc = AddDescriptor{};

    const auto op_t_l0_lhs = add_tensor(op_graph, t_l0_lhs_info);
    const auto op_t_l0_rhs = add_tensor(op_graph, t_l0_rhs_info);
    const auto op_t_l1_rhs = add_tensor(op_graph, t_l1_rhs_info);
    const auto op_t_l2_lhs = add_tensor(op_graph, t_l2_lhs_info);
    const auto op_t_l0_dst = add_tensor(op_graph, t_l0_dst_info); // temp accumulator; TensorInfo to be inferred
    const auto op_t_l1_dst = add_tensor(op_graph, t_l1_dst_info); // temp accumulator; TensorInfo to be inferred
    const auto op_t_dst    = add_tensor(op_graph, t_dst_info);

    add_op_elementwise_add(op_graph, add_desc, op_t_l0_lhs, op_t_l0_rhs, op_t_l0_dst);
    add_op_elementwise_add(op_graph, add_desc, op_t_l0_dst, op_t_l1_rhs, op_t_l1_dst);
    add_op_elementwise_add(op_graph, add_desc, op_t_l1_dst, op_t_l2_lhs, op_t_dst);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    const auto              success = build(workload, op_graph, workload_ctx);

    ARM_COMPUTE_EXPECT(!bool(success), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!bool(ClCompositeOperator::validate(workload)), framework::LogLevel::ERRORS);
}
TEST_CASE(Root_Simple_And_Complex, framework::DatasetMode::ALL)
{
    /* Computation:
     * out = add(conv(l0_0_input, l0_0_weight), add(l0_1_lhs, l0_1_rhs))
     */
    const auto data_type   = DataType::F32;
    const auto data_layout = DataLayout::NHWC;

    const auto t_l0_0_input_shape  = TensorShape(128, 21, 21);
    const auto t_l0_0_weight_shape = TensorShape(144, 128, 1, 1);
    const auto t_l0_1_lhs_shape    = TensorShape(144, 21, 21);
    const auto t_l0_1_rhs_shape    = TensorShape(1, 1, 21);

    auto t_l0_0_input_info  = TensorInfo(t_l0_0_input_shape, 1, data_type, data_layout);
    auto t_l0_0_weight_info = TensorInfo(t_l0_0_weight_shape, 1, data_type, data_layout);
    auto t_l0_1_lhs_info    = TensorInfo(t_l0_1_lhs_shape, 1, data_type, data_layout);
    auto t_l0_1_rhs_info    = TensorInfo(t_l0_1_rhs_shape, 1, data_type, data_layout);
    auto t_l0_0_dst_info    = TensorInfo();
    auto t_l0_1_dst_info    = TensorInfo();
    auto t_dst_info         = TensorInfo();

    OperatorGraph op_graph;
    const auto    conv2d_desc = Conv2dDescriptor{};
    const auto    add_desc    = AddDescriptor{};

    const auto op_t_l0_0_input  = add_tensor(op_graph, t_l0_0_input_info);
    const auto op_t_l0_0_weight = add_tensor(op_graph, t_l0_0_weight_info);
    const auto op_t_l0_1_lhs    = add_tensor(op_graph, t_l0_1_lhs_info);
    const auto op_t_l0_1_rhs    = add_tensor(op_graph, t_l0_1_rhs_info);
    const auto op_t_l0_0_dst    = add_tensor(op_graph, t_l0_0_dst_info); // temp accumulator; TensorInfo to be inferred
    const auto op_t_l0_1_dst    = add_tensor(op_graph, t_l0_1_dst_info); // temp accumulator; TensorInfo to be inferred
    const auto op_t_dst         = add_tensor(op_graph, t_dst_info);

    add_op_conv2d(op_graph, conv2d_desc, op_t_l0_0_input, op_t_l0_0_weight, op_t_l0_0_dst);
    add_op_elementwise_add(op_graph, add_desc, op_t_l0_1_lhs, op_t_l0_1_rhs, op_t_l0_1_dst);
    add_op_elementwise_add(op_graph, add_desc, op_t_l0_0_dst, op_t_l0_1_dst, op_t_dst);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    const auto              success = build(workload, op_graph, workload_ctx);

    ARM_COMPUTE_EXPECT(!bool(success), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!bool(ClCompositeOperator::validate(workload)), framework::LogLevel::ERRORS);
}
TEST_CASE(Loop, framework::DatasetMode::ALL)
{
    /* Computation:
     * tensor state0;
     * state1 = add(l0_lhs, state0)
     * state0 = add(l1_lhs, state1)
     */
    const auto data_type   = DataType::F32;
    const auto data_layout = DataLayout::NHWC;

    const auto t_shape = TensorShape(13, 21);

    auto t_l0_lhs_info = TensorInfo(t_shape, 1, data_type, data_layout);
    auto t_l1_lhs_info = TensorInfo(t_shape, 1, data_type, data_layout);
    auto state0_info   = TensorInfo(t_shape, 1, data_type, data_layout);
    auto state1_info   = TensorInfo();

    OperatorGraph op_graph;
    const auto    conv2d_desc = Conv2dDescriptor{};
    const auto    add_desc    = AddDescriptor{};

    const auto op_t_l0_lhs = add_tensor(op_graph, t_l0_lhs_info);
    const auto op_t_l1_lhs = add_tensor(op_graph, t_l1_lhs_info);
    const auto op_t_state0 = add_tensor(op_graph, state0_info);
    const auto op_t_state1 = add_tensor(op_graph, state1_info);

    add_op_conv2d(op_graph, conv2d_desc, op_t_l0_lhs, op_t_state0, op_t_state1);
    add_op_elementwise_add(op_graph, add_desc, op_t_l1_lhs, op_t_state1, op_t_state0);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    const auto              success = build(workload, op_graph, workload_ctx);

    ARM_COMPUTE_EXPECT(!bool(success), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!bool(ClCompositeOperator::validate(workload)), framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Invalid

TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // INTEGRATION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */